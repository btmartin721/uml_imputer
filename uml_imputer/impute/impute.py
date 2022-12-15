# Standard library imports
import errno
import gc
import os
import pprint
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, Any, Callable

# Third party imports
import numpy as np
import pandas as pd

# Scikit-learn imports

from sklearn_genetic.space import Continuous, Categorical, Integer

# Custom module imports
try:
    from ..read_input.read_input import GenotypeData
    from ..utils.misc import get_processor_name
    from ..utils.misc import timer

except (ModuleNotFoundError, ValueError):
    from read_input.read_input import GenotypeData
    from utils.misc import get_processor_name
    from utils.misc import timer

# Requires scikit-learn-intellex package
if get_processor_name().strip().startswith("Intel"):
    try:
        from sklearnex import patch_sklearn

        patch_sklearn()
        intelex = True
    except ImportError:
        print(
            "Warning: Intel CPU detected but scikit-learn-intelex is not "
            "installed. We recommend installing it to speed up computation."
        )
        intelex = False
else:
    intelex = False


class Impute:
    """Class to impute missing data from the provided classifier.

    The Impute class will run the provided classifier. The settings for the provided estimator should be provided as the ``kwargs`` argument as a dictionary object with the estimator's keyword arguments as the keys and the corresponding values. E.g., ``kwargs={"n_jobs", 4, "initial_strategy": "populations"}``\. ``clf_type`` just specifies either "classifier" or "regressor". "regressor" is primarily just for quick and dirty testing.

    Once the Impute class is initialized, the imputation should be performed with ``fit_predict()``\.

    The imputed data can then be written to a file with ``write_imputed()``

    Args:
        clf (str or Callable estimator object): The estimator object to use. The provided value should be SAE, UBP, or NLPCA.

        clf_type (str): Specify whether to use a "classifier" or "regressor". The "regressor" option is just for quick and dirty testing, and "classifier" should almost always be used.

        kwargs (Dict[str, Any]): Settings to use with the estimator. The keys should be the estimator's keywords, and the values should be their corresponding settings.

    Raises:
        TypeError: Check whether the ``gridparams`` values are of the correct format if ``gridsearch_method == "genetic_algorithm"``\.

    Examples:
        # Don't use parentheses after estimator object.
        >>>imputer = Impute(uml_impute.impute.neural_network_imputer.UBP),
        "classifier",
        {"n_jobs": 4, "grid_iter": 25, "gridsearch_method": "genetic_algorithm"})
        >>>self.imputed, self.best_params = imputer.fit_predict(df)
        >>>imputer.write_imputed(self.imputed)
        >>>print(self.imputed)
        [[0, 1, 1, 2],
        [0, 1, 1, 2],
        [0, 2, 2, 2],
        [2, 2, 2, 2]]
    """

    def __init__(
        self, clf: Union[str, Callable], clf_type: str, kwargs: Dict[str, Any]
    ) -> None:
        self.clf = clf
        self.clf_type = clf_type
        self.original_num_cols = None

        try:
            self.pops = kwargs["genotype_data"].populations
        except AttributeError:
            self.pops = None

        self.genotype_data = kwargs["genotype_data"]
        self.verbose = kwargs["verbose"]

        # Separate local variables into settings objects
        (
            self.imp_kwargs,
            self.ga_kwargs,
            self.verbose,
            self.n_jobs,
            self.prefix,
            self.column_subset,
            self.disable_progressbar,
            self.do_gridsearch,
            self.testing,
        ) = self._gather_impute_settings(kwargs)

        if self.do_gridsearch:
            for v in kwargs["gridparams"].values():
                if (
                    isinstance(v, (Categorical, Integer, Continuous))
                    and kwargs["gridsearch_method"].lower()
                    != "genetic_algorithm"
                ):
                    raise TypeError(
                        "gridsearch_method argument must equal 'genetic_algorithm' if gridparams values are of type sklearn_genetic.space"
                    )

        self.logfilepath = os.path.join(
            f"{self.prefix}_output", "logs", "imputer_progress_log.txt"
        )

        self.invalid_indexes = None

        # Remove logfile if exists
        try:
            os.remove(self.logfilepath)
        except OSError:
            pass

        # Make output file paths.
        Path(os.path.join(f"{self.prefix}_output", "plots")).mkdir(
            parents=True, exist_ok=True
        )

        Path(os.path.join(f"{self.prefix}_output", "logs")).mkdir(
            parents=True, exist_ok=True
        )

        Path(os.path.join(f"{self.prefix}_output", "reports")).mkdir(
            parents=True, exist_ok=True
        )

        Path(os.path.join(f"{self.prefix}_output", "alignments")).mkdir(
            parents=True, exist_ok=True
        )

    @timer
    def fit_predict(
        self, X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit and predict imputations with neural network models.

        Fits and predicts imputed 012-encoded genotypes using deep learning with any of the models. If ``gridsearch_method=None``\, then a grid search is not performed. If ``gridsearch_method!=None``\, then one of three possible types of grid searches is performed and a final imputation is done on the whole dataset using the best found parameters.

        Args:
            X (pandas.DataFrame): DataFrame with 012-encoded genotypes.

        Returns:
            GenotypeData: GenotypeData object with missing genotypes imputed.
            Dict[str, Any]: Best parameters found during grid search.
        """

        # Test if output file can be written to
        try:
            outfile = os.path.join(
                f"{self.prefix}_output", "alignments", "imputed_012.csv"
            )

            # Check if it can be opened.
            with open(outfile, "w") as fout:
                pass
        except IOError as e:
            print(f"Error: {e.errno}, {e.strerror}")
            if e.errno == errno.EACCES:
                sys.exit(f"Permission denied: Cannot write to {outfile}")
            elif e.errno == errno.EISDIR:
                sys.exit(f"Could not write to {outfile}; It is a directory")

        # Don't do a grid search
        if not self.do_gridsearch:
            imputed_df, df_scores, best_params = self._impute_single(X)

        # Do a grid search and get the transformed data with the best parameters
        else:
            imputed_df, df_scores, best_params = self._impute_gridsearch(X)

            if self.verbose > 0:
                print("\nBest Parameters:")
                pprint.pprint(best_params)

        imp_data = self._imputed2genotypedata(imputed_df, self.genotype_data)

        print("\nDone!\n")
        return imp_data, best_params

    def _impute_single(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, None]:
        """Train model with static parameters (i.e., no grid search).

        Args:
            df (pandas.DataFrame): DataFrame of 012-encoded genotypes.

        Returns:
            pandas.DataFrame: Imputed DataFrame of 012-encoded genotypes.
            NoneType: Only used with _impute_gridsearch. Set to None here for compatibility.
        """
        if self.verbose > 0:
            print(
                f"\nDoing {self.clf.__name__} imputation with static parameters..."
            )

        imputer = None

        if self.original_num_cols is None:
            self.original_num_cols = len(df.columns)

        if self.disable_progressbar:
            if self.verbose > 0:
                with open(self.logfilepath, "a") as fout:
                    # Redirect to progress logfile
                    with redirect_stdout(fout):
                        print(f"Doing {self.clf.__name__} imputation...\n")

        imputed_df = self._impute_df(df, imputer)

        if self.disable_progressbar:
            if self.verbose > 0:
                with open(self.logfilepath, "a") as fout:
                    # Redirect to progress logfile
                    with redirect_stdout(fout):
                        print(f"\nDone with {self.clf.__name__} imputation!\n")

        gc.collect()

        self._validate_imputed(imputed_df)

        if self.verbose > 0:
            print(f"\nDone with {self.clf.__name__} imputation!\n")

        return imputed_df, None, None

    def _impute_gridsearch(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Do parameter search with GridSearchCV, RandomizedSearchCV, or GASearchCV.

        Args:
            df (pandas.DataFrame): DataFrame with 012-encoded genotypes.

        Returns:
            pandas.DataFrame: DataFrame with 012-encoded genotypes imputed using the best parameters found with the grid search.
            float: Absolute value of best score found during the grid search.
            dict: Best parameters found during the grid search.
        """
        original_num_cols = len(df.columns)
        df_subset, cols_to_keep = self._subset_data_for_gridsearch(
            df, self.column_subset, original_num_cols
        )

        print(f"Doing {self.clf.__name__} grid search...")

        if self.verbose > 0:
            print(f"Validation dataset size: {len(df_subset.columns)}\n")

        if self.disable_progressbar:
            with open(self.logfilepath, "a") as fout:
                # Redirect to progress logfile
                with redirect_stdout(fout):
                    print(f"Doing {self.clf.__name__} grid search...\n")

        self.imp_kwargs.pop("str_encodings")
        imputer = self.clf(
            **self.imp_kwargs,
            ga_kwargs=self.ga_kwargs,
        )

        df_imp = pd.DataFrame(
            imputer.fit_transform(df_subset), columns=cols_to_keep
        )

        df_imp = df_imp.astype("float")
        df_imp = df_imp.astype("int64")

        if self.verbose > 0:
            print(f"\nDone with {self.clf.__name__} grid search!")

            if self.disable_progressbar:
                if self.verbose > 0:
                    with open(self.logfilepath, "a") as fout:
                        # Redirect to progress logfile
                        with redirect_stdout(fout):
                            print(
                                f"\nDone with {self.clf.__name__} grid search!"
                            )

        best_params = imputer.best_params_
        df_scores = imputer.best_score_
        df_scores = round(df_scores, 2) * 100
        best_imputer = None

        self._write_imputed_params_score(df_scores, best_params)

        # Change values to the ones in best_params
        self.imp_kwargs.update(best_params)

        gc.collect()

        if self.verbose > 0:
            print(
                f"\nDoing {self.clf.__name__} imputation "
                f"with best found parameters...\n"
            )

            if self.disable_progressbar:
                with open(self.logfilepath, "a") as fout:
                    # Redirect to progress logfile
                    with redirect_stdout(fout):
                        print(
                            f"\nDoing {self.clf.__name__} imputation "
                            f"with best found parameters...\n"
                        )

        if self.column_subset == 1.0:
            imputed_df = df_imp.copy()
        else:
            imputed_df = self._impute_df(df, best_imputer)

        gc.collect()

        self._validate_imputed(imputed_df)

        if self.verbose > 0:
            print(f"Done with {self.clf.__name__} imputation!\n")

            if self.disable_progressbar:
                with open(self.logfilepath, "a") as fout:
                    # Redirect to progress logfile
                    with redirect_stdout(fout):
                        print(f"Done with {self.clf.__name__} imputation!\n")

        return imputed_df, df_scores, best_params

    def _impute_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Impute list of pandas.DataFrame objects.

        The DataFrames are chunks of the whole input data, with each chunk correspoding to ``chunk_size`` features from ``_df2chunks()``\.

        Args:
            df_chunks (pandas.DataFrame): Dataframe with shape (n_samples, n_features).

        Returns:
            pandas.DataFrame: Single DataFrame object, with all the imputed chunks concatenated together.
        """
        imputer = self.clf(
            self.imp_kwargs["genotype_data"],
            disable_progressbar=self.disable_progressbar,
            prefix=self.prefix,
        )
        df_imp = pd.DataFrame(
            imputer.fit_transform(df),
        )
        df_imp = df_imp.astype("float")
        df_imp = df_imp.astype("Int8")

        gc.collect()
        return df_imp

    def _imputed2genotypedata(self, imp012, genotype_data):
        """Create new instance of GenotypeData object from imputed DataFrame.

        The imputed, decoded DataFrame gets written to file and re-loaded to instantiate a new GenotypeData object.

        Args:
            imp012 (pandas.DataFrame): Imputed 012-encoded DataFrame.

            genotype_data (GenotypeData): Original GenotypeData object to load attributes from.

        Returns:
            GenotypeData: GenotypeData object with imputed data.
        """
        imputed_filename = genotype_data.decode_imputed(
            imp012,
            write_output=True,
            prefix=self.prefix,
        )

        ft = genotype_data.filetype

        if ft.lower().startswith("structure") and ft.lower().endswith("row"):
            ft += "PopID"

        return GenotypeData(
            filename=imputed_filename,
            filetype=ft,
            popmapfile=genotype_data.popmapfile,
            guidetree=genotype_data.guidetree,
            qmatrix_iqtree=genotype_data.qmatrix_iqtree,
            qmatrix=genotype_data.qmatrix,
            siterates=genotype_data.siterates,
            siterates_iqtree=genotype_data.siterates_iqtree,
            prefix=genotype_data.prefix,
            verbose=False,
        )

    def _subset_data_for_gridsearch(
        self,
        df: pd.DataFrame,
        columns_to_subset: Union[int, float],
        original_num_cols: int,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Randomly subsets pandas.DataFrame.

        Subset pandas DataFrame with ``column_percent`` fraction of the data. Allows for faster validation.

        Args:
            df (pandas.DataFrame): DataFrame with 012-encoded genotypes.

            columns_to_subset (int or float): If float, proportion of DataFrame to randomly subset should be between 0 and 1. if integer, subsets ``columns_to_subset`` random columns.

            original_num_cols (int): Number of columns in original DataFrame.

        Returns:
            pandas.DataFrame: New DataFrame with random subset of features.
            numpy.ndarray: Sorted numpy array of column indices to keep.

        Raises:
            TypeError: column_subset must be of type float or int.
        """

        # Get a random numpy arrray of column names to select
        if isinstance(columns_to_subset, float):
            n = int(original_num_cols * columns_to_subset)
        elif isinstance(columns_to_subset, int):
            n = columns_to_subset
        else:
            raise TypeError(
                f"column_subset must be of type float or int, "
                f"but got {type(columns_to_subset)}"
            )

        col_arr = np.array(df.columns)

        if n > len(df.columns):
            if self.verbose > 0:
                print(
                    "Warning: Column_subset is greater than remaining columns following filtering. Using all columns"
                )

            df_sub = df.copy()
            cols = col_arr.copy()
        else:
            cols = np.random.choice(col_arr, n, replace=False)
            df_sub = df.loc[:, np.sort(cols)]

        df_sub.columns = df_sub.columns.astype(str)

        return df_sub, np.sort(cols)

    def _write_imputed_params_score(
        self, df_scores: pd.DataFrame, best_params: Dict[str, Any]
    ) -> None:
        """Save best_score and best_params to files on disk.

        Args:
            best_score (float): Best RMSE or accuracy score for the regressor or classifier, respectively.

            best_params (dict): Best parameters found in grid search.
        """

        best_score_outfile = os.path.join(
            f"{self.prefix}_output", "reports", "imputed_best_score.csv"
        )
        best_params_outfile = os.path.join(
            f"{self.prefix}_output", "reports", "imputed_best_params.csv"
        )

        if isinstance(df_scores, pd.DataFrame):
            df_scores.to_csv(
                best_score_outfile,
                header=True,
                index=False,
                float_format="%.2f",
            )

        else:
            with open(best_score_outfile, "w") as fout:
                fout.write(f"accuracy,{df_scores}\n")

        with open(best_params_outfile, "w") as fout:
            fout.write("parameter,best_value\n")
            for k, v in best_params.items():
                fout.write(f"{k},{v}\n")

    def _validate_imputed(self, df: pd.DataFrame) -> None:
        """Asserts that there is no missing data left in the imputed DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame with imputed 012-encoded genotypes.

        Raises:
            AssertionError: Error if missing values are still found in the dataset after imputation.
        """
        assert (
            not df.isnull().values.any()
        ), "Imputation failed...Missing values found in the imputed dataset"

    def _gather_impute_settings(
        self, kwargs: Dict[str, Any]
    ) -> Tuple[
        Optional[Dict[str, Any]],
        Optional[int],
        Optional[int],
        Optional[str],
        Optional[Union[int, float]],
        Optional[bool],
        Optional[bool],
        Optional[bool],
    ]:
        """Gather impute settings from kwargs object.

        Gather impute settings from the imputation class. Gathers them for use with the ``Impute`` class. Returns dictionary with keys as keyword arguments and the values as the settings.

        Args:
            kwargs (Dict[str, Any]): Dictionary with keys as the keyword arguments and their corresponding values.

        Returns:
            Dict[str, Any]: Imputer keyword arguments.
            Dict[str, Any]: Genetic algorithm keyword arguments.
            int: Verbosity setting. 0 is silent, 2 is most verbose.
            int: Number of processors to use with grid search.
            str or None: Prefix for output files.
            int or float: Proportion of dataset (if float) or number of columns (if int) to use for grid search.
            bool: If True, disables the tqdm progress bar and just prints status updates to a file. If False, uses tqdm progress bar.
            bool: True if doing grid search, False otherwise.
            bool: Whether to make test prints when training model.
        """
        n_jobs = kwargs.pop("n_jobs", 1)
        column_subset = kwargs.pop("column_subset", None)
        verbose = kwargs.get("verbose", 0)
        disable_progressbar = kwargs.get("disable_progressbar", False)
        prefix = kwargs.get("prefix")
        testing = kwargs.get("testing", False)
        do_gridsearch = False if kwargs["gridsearch_method"] is None else True

        imp_kwargs = kwargs.copy()
        ga_kwargs = kwargs.copy()

        to_remove = ["self", "__class__"]

        ga_keys = [
            "population_size",
            "tournament_size",
            "elitism",
            "crossover_probability",
            "mutation_probability",
            "ga_algorithm",
        ]

        for k in imp_kwargs.copy().keys():
            if k in to_remove:
                imp_kwargs.pop(k)

        for k in ga_kwargs.copy().keys():
            if k not in ga_keys:
                ga_kwargs.pop(k)

        for k in imp_kwargs.copy().keys():
            if k in ga_keys:
                imp_kwargs.pop(k)

        if "ga_algorithm" in ga_kwargs:
            ga_kwargs["algorithm"] = ga_kwargs.pop("ga_algorithm")

        return (
            imp_kwargs,
            ga_kwargs,
            verbose,
            n_jobs,
            prefix,
            column_subset,
            disable_progressbar,
            do_gridsearch,
            testing,
        )
