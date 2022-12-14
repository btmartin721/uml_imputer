# Standard library imports
import sys
from typing import Optional, Union, Dict, Any

# Third-party imports
import numpy as np
import pandas as pd

# Scikit-learn imports
import lightgbm as lgbm
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsClassifier

# Custom imports
try:
    from .impute import Impute
    from .unsupervised.neural_network_imputers import VAE, UBP, SAE
    from ..utils.misc import get_processor_name
except (ModuleNotFoundError, ValueError):
    from impute.impute import Impute
    from impute.unsupervised.neural_network_imputers import VAE, UBP, SAE
    from utils.misc import get_processor_name

# Requires scikit-learn-intellex package
if get_processor_name().strip().startswith("Intel"):
    try:
        from sklearnex import patch_sklearn

        patch_sklearn()
        intelex = True
    except ImportError:
        print(
            "Warning: Intel CPU detected but scikit-learn-intelex is not installed. We recommend installing it to speed up computation."
        )
        intelex = False
else:
    intelex = False


class ImputeKNN(Impute):
    """Does K-Nearest Neighbors Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If ``gridparams=None``\, a grid search is not performed, otherwise ``gridparams`` will be used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``, then the ``gridparams`` values can be lists of or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. Defaults to None.

        do_validation (bool, optional): Whether to validate the imputation if not doing a grid search. This validation method randomly replaces between 15% and 50% of the known, non-missing genotypes in ``n_features * column_subset`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None for ``do_validation`` to work. Calculating a validation score can be turned off altogether by setting ``do_validation`` to False. Defaults to False.

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search or validation. Should be between 0 and 1, and should also be small, because the grid search or validation takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        cv (int, optional): Number of folds for cross-validation during grid search. Defaults to 5.

        n_neighbors (int, optional): Number of neighbors to use by default for K-Nearest Neighbors queries. Defaults to 5.

        weights (str, optional): Weight function used in prediction. Possible values: 'Uniform': Uniform weights with all points in each neighborhood weighted equally; 'distance': Weight points by the inverse of their distance, in this case closer neighbors of a query point will have  a greater influence than neighbors that are further away; 'callable': A user-defined function that accepts an array of distances and returns an array of the same shape containing the weights. Defaults to "distance".

        algorithm (str, optional): Algorithm used to compute the nearest neighbors. Possible values: 'ball_tree', 'kd_tree', 'brute', 'auto'. Defaults to "auto".

        leaf_size (int, optional): Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem. Defaults to 30.

        p (int, optional): Power parameter for the Minkowski metric. When p=1, this is equivalent to using manhattan_distance (l1), and if p=2 it is equivalent to using euclidean distance (l2). For arbitrary p, minkowski_distance (l_p) is used. Defaults to 2.

        metric (str, optional): The distance metric to use for the tree. The default metric is minkowski, and with p=2 this is equivalent to the standard Euclidean metric. See the documentation of sklearn.DistanceMetric for a list of available metrics. If metric is 'precomputed', X is assumed to be a distance matrix and must be square during fit. Defaults to "minkowski".

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: ???most_frequent???, "populations", "phylogeny", or "nmf". "most_frequent" uses the overall mode of each column. "populations" uses the mode per population/ per column via a population map file and the ``ImputeAlleleFreq`` class. "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. "nmf" performs the imputaton via matrix factorization via the ``ImputeNMF`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (dict(str: int), optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        gridsearch_method (str, optional): Grid search method to use. Supported options include: {"gridsearch", "randomized_gridsearch", and "genetic_algorithm"}. Whether to use a genetic algorithm for the grid search. "gridsearch" uses GridSearchCV to test every possible parameter combination. "randomized_gridsearch" picks ``grid_iter`` random combinations of parameters to test. "genetic_algorithm" uses a genetic algorithm via sklearn-genetic-opt GASearchCV to do the grid search. If set to None, then does not do a grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to "gridsearch".

        grid_iter (int, optional): Number of iterations for randomized and genetic algorithm grid searches. Defaults to 80.

        population_size (int or str, optional): For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io). Defaults to "auto".

        tournament_size (int, optional): For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation. Defaults to 0.2.

        ga_algorithm (str, optional): For genetic algorithm grid search: Evolutionary algorithm to use. Supported options include: {"eaMuPlusLambda", "eaMuCommaLambda", "eaSimple"}. If you need to speed up the genetic algorithm grid search, try setting ``algorithm`` to "euSimple", at the expense of evolutionary model robustness. See more details in the DEAP algorithms documentation (https://deap.readthedocs.io). Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): If the genetic algorithm sees ``early_stop_gen`` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform. Defaults to 5.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``\%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the regressor. -1 means using all available processors. Defaults to 1.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>from sklearn_genetic.space import Categorical, Integer, Continuous
        >>>
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>># Genetic Algorithm grid_params
        >>>grid_params = {
        >>>    "n_neighbors": Integer(3, 10),
        >>>    "leaf_size": Integer(10, 50),
        >>>}
        >>>
        >>>knn = ImputeKNN(
        >>>     genotype_data=data,
        >>>     gridparams=grid_params,
        >>>     cv=5,
        >>>     gridsearch_method="genetic_algorithm",
        >>>     n_nearest_features=10,
        >>>     n_estimators=100,
        >>>     initial_strategy="phylogeny",
        >>>)
        >>>
        >>>knn_gtdata = knn.imputed
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prefix: str = "output",
        gridparams: Optional[Dict[str, Any]] = None,
        do_validation: bool = False,
        column_subset: Union[int, float] = 0.1,
        cv: int = 5,
        n_neighbors: int = 5,
        weights: str = "distance",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        max_iter: int = 10,
        tol: float = 1e-3,
        n_nearest_features: Optional[int] = 10,
        initial_strategy: str = "most_frequent",
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        random_state: Optional[int] = None,
        gridsearch_method: str = "gridsearch",
        grid_iter: int = 80,
        population_size: Union[int, str] = "auto",
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.2,
        ga_algorithm: str = "eaMuPlusLambda",
        early_stop_gen: int = 5,
        scoring_metric: str = "accuracy",
        chunk_size: Union[int, float] = 1.0,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        # Get local variables into dictionary object
        kwargs = locals()

        self.clf_type = "classifier"
        self.clf = KNeighborsClassifier

        super().__init__(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = self.fit_predict(
            genotype_data.genotypes012_df
        )


class ImputeRandomForest(Impute):
    """Does Random Forest or Extra Trees Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        write_output (bool, optional): If True, writes imputed data to file on disk. Otherwise just stores it as a class attribute.

        gridparams (Dict[str, Any] or None or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If ``gridparams=None``\, a grid search is not performed, otherwise ``gridparams`` will be used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``, then the ``gridparams`` values can be lists of or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. Defaults to None.

        do_validation (bool, optional): Whether to validate the imputation if not doing a grid search. This validation method randomly replaces between 15% and 50% of the known, non-missing genotypes in ``n_features * column_subset`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None for ``do_validation`` to work. Calculating a validation score can be turned off altogether by setting ``do_validation`` to False. Defaults to False.

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search or validation. Should be between 0 and 1, and should also be small, because the grid search or validation takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        cv (int, optional): Number of folds for cross-validation during grid search. Defaults to 5.

        extra_trees (bool, optional): Whether to use ExtraTreesClassifier (If True) instead of RandomForestClassifier (If False). ExtraTreesClassifier is faster, but is not supported by the scikit-learn-intelex patch, whereas RandomForestClassifier is. If using an Intel CPU, the optimizations provided by the scikit-learn-intelex patch might make setting ``extratrees=False`` worthwhile. If you are not using an Intel CPU, the scikit-learn-intelex library is not supported and ExtraTreesClassifier will be faster with similar performance. NOTE: If using scikit-learn-intelex, ``criterion`` must be set to "gini" and ``oob_score`` to False, as those parameters are not currently supported herein. Defaults to True.

        n_estimators (int, optional): The number of trees in the forest. Increasing this value can improve the fit, but at the cost of compute time and resources. Defaults to 100.

        criterion (str, optional): The function to measure the quality of a split. Supported values are "gini" for the Gini impurity and "entropy" for the information gain. Defaults to "gini".

        max_depth (int, optional): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. Defaults to None.

        min_samples_split (int or float, optional): The minimum number of samples required to split an internal node. If value is an integer, then considers min_samples_split as the minimum number. If value is a floating point, then min_samples_split is a fraction and (min_samples_split * n_samples), rounded up to the nearest integer, are the minimum number of samples for each split. Defaults to 2.

        min_samples_leaf (int or float, optional): The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If value is an integer, then ``min_samples_leaf`` is the minimum number. If value is floating point, then ``min_samples_leaf`` is a fraction and ``int(min_samples_leaf * n_samples)`` is the minimum number of samples for each node. Defaults to 1.

        min_weight_fraction_leaf (float, optional): The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided. Defaults to 0.0.

        max_features (str, int, float, or None, optional): The number of features to consider when looking for the best split. If int, then consider "max_features" features at each split. If float, then "max_features" is a fraction and ``int(max_features * n_samples)`` features are considered at each split. If "auto", then ``max_features=sqrt(n_features)``\. If "sqrt", then ``max_features=sqrt(n_features)``\. If "log2", then ``max_features=log2(n_features)``\. If None, then ``max_features=n_features``\. Defaults to "auto".

        max_leaf_nodes (int or None, optional): Grow trees with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes. Defaults to None.

        min_impurity_decrease (float, optional): A node will be split if this split induces a decrease of the impurity greater than or equal to this value. See ``sklearn.ensemble.ExtraTreesClassifier`` documentation for more information. Defaults to 0.0.

        bootstrap (bool, optional): Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree. Defaults to False.

        oob_score (bool, optional): Whether to use out-of-bag samples to estimate the generalization score. Only available if ``bootstrap=True``\. Defaults to False.

        max_samples (int or float, optional): If bootstrap is True, the number of samples to draw from X to train each base estimator. If None (default), then draws ``X.shape[0] samples``\. if int, then draws ``max_samples`` samples. If float, then draws ``int(max_samples * X.shape[0] samples)`` with ``max_samples`` in the interval (0, 1). Defaults to None.

        clf_random_state (int or None, optional): Controls three sources of randomness for ``sklearn.ensemble.ExtraTreesClassifier``: The bootstrapping of the samples used when building trees (if ``bootstrap=True``), the sampling of the features to consider when looking for the best split at each node (if ``max_features < n_features``), and the draw of the splits for each of the ``max_features``\. If None, then uses a different random seed each time the imputation is run. Defaults to None.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: ???most_frequent???, "populations", "phylogeny", or "nmf". "most_frequent" uses the overall mode of each column. "populations" uses the mode per population/ per column via a population map file and the ``ImputeAlleleFreq`` class. "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. "nmf" performs the imputaton via matrix factorization via the ``ImputeNMF`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        gridsearch_method (str, optional): Grid search method to use. Supported options include: {"gridsearch", "randomized_gridsearch", and "genetic_algorithm"}. Whether to use a genetic algorithm for the grid search. "gridsearch" uses GridSearchCV to test every possible parameter combination. "randomized_gridsearch" picks ``grid_iter`` random combinations of parameters to test. "genetic_algorithm" uses a genetic algorithm via sklearn-genetic-opt GASearchCV to do the grid search. If set to None, then does not do a grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to "gridsearch".

        grid_iter (int, optional): Number of iterations for randomized and genetic algorithm grid searches. Defaults to 80.

        population_size (int or str, optional): For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io). Defaults to "auto".

        tournament_size (int, optional): For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation. Defaults to 0.2.

        ga_algorithm (str, optional): For genetic algorithm grid search: Evolutionary algorithm to use. Supported options include: {"eaMuPlusLambda", "eaMuCommaLambda", "eaSimple"}. If you need to speed up the genetic algorithm grid search, try setting ``algorithm`` to "euSimple", at the expense of evolutionary model robustness. See more details in the DEAP algorithms documentation (https://deap.readthedocs.io). Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): If the genetic algorithm sees ``early_stop_gen`` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform. Defaults to 5.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``\%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the regressor. -1 means using all available processors. Defaults to 1.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>from sklearn_genetic.space import Categorical, Integer, Continuous
        >>>
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>># Genetic Algorithm grid_params
        >>>grid_params = {
        >>>    "min_samples_leaf": Integer(1, 10),
        >>>    "max_depth": Integer(2, 110),
        >>>}
        >>>
        >>>rf = ImputeRandomForest(
        >>>     genotype_data=data,
        >>>     gridparams=grid_params,
        >>>     cv=5,
        >>>     gridsearch_method="genetic_algorithm",
        >>>     n_nearest_features=10,
        >>>     n_estimators=100,
        >>>     initial_strategy="phylogeny",
        >>>)
        >>>
        >>>rf_gtdata = rf.imputed
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prefix: str = "output",
        gridparams: Optional[Dict[str, Any]] = None,
        do_validation: bool = False,
        column_subset: Union[int, float] = 0.1,
        cv: int = 5,
        extratrees: bool = True,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[str, int, float]] = "auto",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = False,
        oob_score: bool = False,
        max_samples: Optional[Union[int, float]] = None,
        clf_random_state: Optional[Union[int, np.random.RandomState]] = None,
        max_iter: int = 10,
        tol: float = 1e-3,
        n_nearest_features: Optional[int] = 10,
        initial_strategy: str = "populations",
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        random_state: Optional[int] = None,
        gridsearch_method: str = "gridsearch",
        grid_iter: int = 80,
        population_size: Union[int, str] = "auto",
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.2,
        ga_algorithm: str = "eaMuPlusLambda",
        early_stop_gen: int = 5,
        scoring_metric: str = "accuracy",
        chunk_size: Union[int, float] = 1.0,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        # Get local variables into dictionary object
        kwargs = locals()

        self.extratrees = kwargs.pop("extratrees")

        if self.extratrees:
            self.clf = ExtraTreesClassifier

        elif intelex and not self.extratrees:
            self.clf = RandomForestClassifier

            if kwargs["criterion"] != "gini":
                raise ValueError(
                    "criterion must be set to 'gini' if using the RandomForestClassifier with scikit-learn-intelex"
                )
            if kwargs["oob_score"]:
                raise ValueError(
                    "oob_score must be set to False if using the RandomForestClassifier with scikit-learn-intelex"
                )
        else:
            self.clf = RandomForestClassifier

        self.clf_type = "classifier"

        super().__init__(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = self.fit_predict(
            genotype_data.genotypes012_df
        )


class ImputeGradientBoosting(Impute):
    """Does Gradient Boosting Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If ``gridparams=None``\, a grid search is not performed, otherwise ``gridparams`` will be used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``, then the ``gridparams`` values can be lists of or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. Defaults to None.

        do_validation (bool, optional): Whether to validate the imputation if not doing a grid search. This validation method randomly replaces between 15% and 50% of the known, non-missing genotypes in ``n_features * column_subset`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None for ``do_validation`` to work. Calculating a validation score can be turned off altogether by setting ``do_validation`` to False. Defaults to False.

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search or validation. Should be between 0 and 1, and should also be small, because the grid search or validation takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        cv (int, optional): Number of folds for cross-validation during grid search. Defaults to 5.

        n_estimators (int, optional): The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance but also increases compute time and required resources. Defaults to 100.

        loss (str, optional): The loss function to be optimized. "deviance" refers to deviance (=logistic regression) for classification with probabilistic outputs. For loss "exponential" gradient boosting recovers the AdaBoost algorithm. Defaults to "deviance".

        learning_rate (float, optional): Learning rate shrinks the contribution of each tree by ``learning_rate``\. There is a trade-off between ``learning_rate`` and ``n_estimators``\. Defaults to 0.1.

        subsample (float, optional): The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. ``subsample`` interacts with the parameter ``n_estimators``\. Choosing ``subsample < 1.0`` leads to a reduction of variance and an increase in bias. Defaults to 1.0.

        criterion (str, optional): The function to measure the quality of a split. Supported criteria are "friedman_mse" for the mean squared error with improvement score by Friedman and "mse" for mean squared error. The default value of "friedman_mse" is generally the best as it can provide a better approximation in some cases. Defaults to "friedman_mse".

        min_samples_split (int or float, optional): The minimum number of samples required to split an internal node. If value is an integer, then consider ``min_samples_split`` as the minimum number. If value is a floating point, then min_samples_split is a fraction and ``(min_samples_split * n_samples)`` is rounded up to the nearest integer and used as the number of samples per split. Defaults to 2.

        min_samples_leaf (int or float, optional): The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If value is an integer, consider ``min_samples_leaf`` as the minimum number. If value is a floating point, then ``min_samples_leaf`` is a fraction and ``(min_samples_leaf * n_samples)`` rounded up to the nearest integer is the minimum number of samples per node. Defaults to 1.

        min_weight_fraction_leaf (float, optional): The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when ``sample_weight`` is not provided. Defaults to 0.0.

        max_depth (int, optional): The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables. Defaults to 3.

        min_impurity_decrease (float, optional): A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Defaults to 0.0. See ``sklearn.ensemble.GradientBoostingClassifier`` documentation for more information. Defaults to 0.0.

        max_features (int, float, str, or None, optional): The number of features to consider when looking for the best split. If value is an integer, then consider ``max_features`` features at each split. If value is a floating point, then ``max_features`` is a fraction and ``(max_features * n_features)`` is rounded to the nearest integer and considered as the number of features per split. If "auto", then ``max_features=sqrt(n_features)``\. If "sqrt", then ``max_features=sqrt(n_features)``\. If "log2", then ``max_features=log2(n_features)``\. If None, then ``max_features=n_features``\. Defaults to None.

        max_leaf_nodes (int or None, optional): Grow trees with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then uses an unlimited number of leaf nodes. Defaults to None.

        clf_random_state (int, numpy.random.RandomState object, or None, optional): Controls the random seed given to each Tree estimator at each boosting iteration. In addition, it controls the random permutation of the features at each split. Pass an int for reproducible output across multiple function calls. If None, then uses a different random seed for each function call. Defaults to None.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int or None, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: ???most_frequent???, "populations", "phylogeny", or "nmf". "most_frequent" uses the overall mode of each column. "populations" uses the mode per population/ per column via a population map file and the ``ImputeAlleleFreq`` class. "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. "nmf" performs the imputaton via matrix factorization via the ``ImputeNMF`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if ``n_nearest_features`` is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        gridsearch_method (str, optional): Grid search method to use. Supported options include: {"gridsearch", "randomized_gridsearch", and "genetic_algorithm"}. Whether to use a genetic algorithm for the grid search. "gridsearch" uses GridSearchCV to test every possible parameter combination. "randomized_gridsearch" picks ``grid_iter`` random combinations of parameters to test. "genetic_algorithm" uses a genetic algorithm via sklearn-genetic-opt GASearchCV to do the grid search. If set to None, then does not do a grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to "gridsearch".

        grid_iter (int, optional): Number of iterations for randomized and genetic algorithm grid searches. Defaults to 80.

        population_size (int or str, optional): For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io). Defaults to "auto".

        tournament_size (int, optional): For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search: If True takes the tournament_size best solution to the next generation. See GASearchCV documentation. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation. Defaults to 0.2.

        ga_algorithm (str, optional): For genetic algorithm grid search: Evolutionary algorithm to use. Supported options include: {"eaMuPlusLambda", "eaMuCommaLambda", "eaSimple"}. If you need to speed up the genetic algorithm grid search, try setting ``algorithm`` to "euSimple", at the expense of evolutionary model robustness. See more details in the DEAP algorithms documentation (https://deap.readthedocs.io). Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): If the genetic algorithm sees ``early_stop_gen`` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform. Defaults to 5.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``\%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the regressor. -1 means using all available processors. Defaults to 1.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>from sklearn_genetic.space import Categorical, Integer, Continuous
        >>>
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>># Genetic Algorithm grid_params
        >>>grid_params = {
        >>>    "learning_rate": Continuous(lower=0.01, upper=0.1),
        >>>    "max_depth": Integer(2, 110),
        >>>}
        >>>
        >>>gb = ImputeGradientBoosting(
        >>>     genotype_data=data,
        >>>     gridparams=grid_params,
        >>>     cv=5,
        >>>     gridsearch_method="genetic_algorithm",
        >>>     n_nearest_features=10,
        >>>     n_estimators=100,
        >>>     initial_strategy="phylogeny",
        >>>)
        >>>
        >>>gb_gtdata = gb.imputed
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prefix: str = "output",
        gridparams: Optional[Dict[str, Any]] = None,
        do_validation: bool = False,
        column_subset: Union[int, float] = 0.1,
        cv: int = 5,
        n_estimators: int = 100,
        loss: str = "deviance",
        learning_rate: float = 0.1,
        subsample: Union[int, float] = 1.0,
        criterion: str = "friedman_mse",
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_depth: Optional[int] = 3,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Union[str, int, float]] = None,
        max_leaf_nodes: Optional[int] = None,
        clf_random_state: Optional[Union[int, np.random.RandomState]] = None,
        max_iter: int = 10,
        tol: float = 1e-3,
        n_nearest_features: Optional[int] = 10,
        initial_strategy: str = "populations",
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        random_state: Optional[int] = None,
        gridsearch_method: str = "gridsearch",
        grid_iter: int = 80,
        population_size: Union[int, str] = "auto",
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.2,
        ga_algorithm: str = "eaMuPlusLambda",
        early_stop_gen: int = 5,
        scoring_metric: str = "accuracy",
        chunk_size: Union[int, float] = 1.0,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        # Get local variables into dictionary object
        kwargs = locals()

        self.clf_type = "classifier"
        self.clf = GradientBoostingClassifier

        super().__init__(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = self.fit_predict(
            genotype_data.genotypes012_df
        )


class ImputeBayesianRidge(Impute):
    """NOTE: This is a regressor estimator and is only intended for testing purposes, as it is faster than the classifiers. Does Bayesian Ridge Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If ``gridparams=None``\, a grid search is not performed, otherwise ``gridparams`` will be used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``, then the ``gridparams`` values can be lists of or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. Defaults to None.

        do_validation (bool, optional): Whether to validate the imputation if not doing a grid search. This validation method randomly replaces between 15% and 50% of the known, non-missing genotypes in ``n_features * column_subset`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None for ``do_validation`` to work. Calculating a validation score can be turned off altogether by setting ``do_validation`` to False. Defaults to False.

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search or validation. Should be between 0 and 1, and should also be small, because the grid search or validation takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        cv (int, optional): Number of folds for cross-validation during grid search. Defaults to 5.

        n_iter (int, optional): Maximum number of iterations. Should be greater than or equal to 1. Defaults to 300.

        clf_tol (float, optional): Stop the algorithm if w has converged. Defaults to 1e-3.

        alpha_1 (float, optional): Hyper-parameter: shape parameter for the Gamma distribution prior over the alpha parameter. Defaults to 1e-6.

        alpha_2 (float, optional): Hyper-parameter: inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter. Defaults to 1e-6.

        lambda_1 (float, optional): Hyper-parameter: shape parameter for the Gamma distribution prior over the lambda parameter. Defaults to 1e-6.

        lambda_2 (float, optional): Hyper-parameter: inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter. Defaults to 1e-6.

        alpha_init (float or None, optional): Initial value for alpha (precision of the noise). If None, ``alpha_init`` is ``1/Var(y)``\. Defaults to None.

        lambda_init (float or None, optional): Initial value for lambda (precision of the weights). If None, ``lambda_init`` is 1. Defaults to None.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int or None, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: ???most_frequent???, "populations", "phylogeny", or "nmf". "most_frequent" uses the overall mode of each column. "populations" uses the mode per population/ per column via a population map file and the ``ImputeAlleleFreq`` class. "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. "nmf" performs the imputaton via matrix factorization via the ``ImputeNMF`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if ``n_nearest_features`` is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        gridsearch_method (str, optional): Grid search method to use. Supported options include: {"gridsearch", "randomized_gridsearch", and "genetic_algorithm"}. Whether to use a genetic algorithm for the grid search. "gridsearch" uses GridSearchCV to test every possible parameter combination. "randomized_gridsearch" picks ``grid_iter`` random combinations of parameters to test. "genetic_algorithm" uses a genetic algorithm via sklearn-genetic-opt GASearchCV to do the grid search. If set to None, then does not do a grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to "gridsearch".

        grid_iter (int, optional): Number of iterations for randomized and genetic algorithm grid searches. Defaults to 80.

        population_size (int or str, optional): For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io). Defaults to "auto".

        tournament_size (int, optional): For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation. Defaults to 0.2.

        ga_algorithm (str, optional): For genetic algorithm grid search: Evolutionary algorithm to use. Supported options include: {"eaMuPlusLambda", "eaMuCommaLambda", "eaSimple"}. If you need to speed up the genetic algorithm grid search, try setting ``algorithm`` to "euSimple", at the expense of evolutionary model robustness. See more details in the DEAP algorithms documentation (https://deap.readthedocs.io). Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): If the genetic algorithm sees ``early_stop_gen`` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform. Defaults to 5.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``\%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the regressor. -1 means using all available processors. Defaults to 1.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>from sklearn_genetic.space import Categorical, Integer, Continuous
        >>>
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>># Genetic Algorithm grid_params
        >>>grid_params = {
        >>>    "alpha_1": Continuous(
        >>>         lower=1e-6, upper=0.1, distribution="log-uniform"
        >>>     ),
        >>>     "lambda_1": Continuous(
        >>>         lower=1e-6, upper=0.1, distribution="log-uniform"
        >>>     ),
        >>>}
        >>>
        >>>br = ImputeBayesianRidge(
        >>>     genotype_data=data,
        >>>     gridparams=grid_params,
        >>>     cv=5,
        >>>     gridsearch_method="genetic_algorithm",
        >>>     n_nearest_features=10,
        >>>     n_estimators=100,
        >>>     initial_strategy="phylogeny",
        >>>)
        >>>
        >>>br_gtdata = br.imputed
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prefix: str = "output",
        gridparams: Optional[Dict[str, Any]] = None,
        do_validation: bool = False,
        column_subset: Union[int, float] = 0.1,
        cv: int = 5,
        n_iter: int = 300,
        clf_tol: float = 1e-3,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        alpha_init: Optional[float] = None,
        lambda_init: Optional[float] = None,
        max_iter: int = 10,
        tol: float = 1e-3,
        n_nearest_features: Optional[int] = 10,
        initial_strategy: str = "populations",
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        random_state: Optional[int] = None,
        gridsearch_method: str = "gridsearch",
        grid_iter: int = 80,
        population_size: Union[int, str] = "auto",
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.2,
        ga_algorithm: str = "eaMuPlusLambda",
        early_stop_gen: int = 5,
        scoring_metric: str = "neg_mean_squared_error",
        chunk_size: Union[int, float] = 1.0,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        # Get local variables into dictionary object
        kwargs = locals()
        kwargs["normalize"] = True
        kwargs["sample_posterior"] = False

        self.clf_type = "regressor"
        self.clf = BayesianRidge

        super().__init__(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = self.fit_predict(
            genotype_data.genotypes012_df
        )


class ImputeXGBoost(Impute):
    """Does XGBoost (Extreme Gradient Boosting) Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process. The grid searches are not compatible with XGBoost, but validation scores can still be calculated without a grid search. In addition, ImputeLightGBM is a similar algorithm and is compatible with grid searches, so use ImputeLightGBM if you want a grid search.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If ``gridparams=None``\, a grid search is not performed, otherwise ``gridparams`` will be used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``, then the ``gridparams`` values can be lists of or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. Defaults to None.

        do_validation (bool, optional): Whether to validate the imputation if not doing a grid search. This validation method randomly replaces between 15% and 50% of the known, non-missing genotypes in ``n_features * column_subset`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None for ``do_validation`` to work. Calculating a validation score can be turned off altogether by setting ``do_validation`` to False. Defaults to False.

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search or validation. Should be between 0 and 1, and should also be small, because the grid search or validation takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        cv (int, optional): Number of folds for cross-validation during grid search. Defaults to 5.

        n_estimators (int, optional): The number of boosting rounds. Increasing this value can improve the fit, but at the cost of compute time and RAM usage. Defaults to 100.

        max_depth (int, optional): Maximum tree depth for base learners. Defaults to 3.

        learning_rate (float, optional): Boosting learning rate (eta). Basically, it serves as a weighting factor for correcting new trees when they are added to the model. Typical values are between 0.1 and 0.3. Lower learning rates generally find the best optimum at the cost of requiring far more compute time and resources. Defaults to 0.1.

        booster (str, optional): Specify which booster to use. Possible values include "gbtree", "gblinear", and "dart". Defaults to "gbtree".

        gamma (float, optional): Minimum loss reduction required to make a further partition on a leaf node of the tree. Defaults to 0.0.

        min_child_weight (float, optional): Minimum sum of instance weight(hessian) needed in a child. Defaults to 1.0.

        max_delta_step (float, optional): Maximum delta step we allow each tree's weight estimation to be. Defaults to 0.0.

        subsample (float, optional): Subsample ratio of the training instance. Defaults to 1.0.

        colsample_bytree (float, optional): Subsample ratio of columns when constructing each tree. Defaults to 1.0.

        reg_lambda (float, optional): L2 regularization term on weights (xgb's lambda parameter). Defaults to 1.0.

        reg_alpha (float, optional): L1 regularization term on weights (xgb's alpha parameter). Defaults to 1.0.

        clf_random_state (int, numpy.random.RandomState object, or None, optional): Controls three sources of randomness for ``sklearn.ensemble.ExtraTreesClassifier``: The bootstrapping of the samples used when building trees (if ``bootstrap=True``), the sampling of the features to consider when looking for the best split at each node (if ``max_features < n_features``), and the draw of the splits for each of the ``max_features``\. If None, then uses a different random seed each time the imputation is run. Defaults to None.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int or None, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: ???most_frequent???, "populations", "phylogeny", or "nmf". "most_frequent" uses the overall mode of each column. "populations" uses the mode per population/ per column via a population map file and the ``ImputeAlleleFreq`` class. "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. "nmf" performs the imputaton via matrix factorization via the ``ImputeNMF`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if ``n_nearest_features`` is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        gridsearch_method (str, optional): Grid search method to use. Supported options include: {"gridsearch", "randomized_gridsearch", and "genetic_algorithm"}. Whether to use a genetic algorithm for the grid search. "gridsearch" uses GridSearchCV to test every possible parameter combination. "randomized_gridsearch" picks ``grid_iter`` random combinations of parameters to test. "genetic_algorithm" uses a genetic algorithm via sklearn-genetic-opt GASearchCV to do the grid search. If set to None, then does not do a grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to "gridsearch".

        grid_iter (int, optional): Number of iterations for randomized and genetic algorithm grid searches. Defaults to 80.

        population_size (int or str, optional): For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io). Defaults to "auto".

        tournament_size (int, optional): For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search: If True takes the tournament_size best solution to the next generation. See GASearchCV documentation. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation. Defaults to 0.2.

        ga_algorithm (str, optional): For genetic algorithm grid search: Evolutionary algorithm to use. Supported options include: {"eaMuPlusLambda", "eaMuCommaLambda", "eaSimple"}. If you need to speed up the genetic algorithm grid search, try setting ``algorithm`` to "euSimple", at the expense of evolutionary model robustness. See more details in the DEAP algorithms documentation (https://deap.readthedocs.io). Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): If the genetic algorithm sees ``early_stop_gen`` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform. Defaults to 5.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``\%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors. Defaults to 1.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>from sklearn_genetic.space import Categorical, Integer, Continuous
        >>>
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>># Genetic Algorithm grid_params
        >>>grid_params = {
        >>>    "learning_rate": Continuous(lower=0.01, upper=0.1),
        >>>    "max_depth": Integer(2, 110),
        >>>}
        >>>
        >>>xgb = ImputeXGBoost(
        >>>     genotype_data=data,
        >>>     gridparams=grid_params,
        >>>     cv=5,
        >>>     gridsearch_method="genetic_algorithm",
        >>>     n_nearest_features=10,
        >>>     n_estimators=100,
        >>>     initial_strategy="phylogeny",
        >>>)
        >>>
        >>>xgb_gtdata = xgb.imputed
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prefix: str = "output",
        gridparams: Optional[Dict[str, Any]] = None,
        do_validation: bool = False,
        column_subset: Union[int, float] = 0.1,
        cv: int = 5,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        booster: str = "gbtree",
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        max_delta_step: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        clf_random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_nearest_features: Optional[int] = 10,
        max_iter: int = 10,
        tol: float = 1e-3,
        initial_strategy: str = "populations",
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        random_state: Optional[int] = None,
        gridsearch_method: str = "gridsearch",
        grid_iter: int = 80,
        population_size: Union[int, str] = "auto",
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.2,
        ga_algorithm: str = "eaMuPlusLambda",
        early_stop_gen: int = 5,
        scoring_metric: str = "accuracy",
        chunk_size: Union[int, float] = 1.0,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        # Get local variables into dictionary object
        kwargs = locals()
        kwargs["gridparams"] = None
        # kwargs["num_class"] = 3
        # kwargs["use_label_encoder"] = False

        self.clf_type = "classifier"
        self.clf = xgb.XGBClassifier
        kwargs["verbosity"] = verbose

        super().__init__(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = self.fit_predict(
            genotype_data.genotypes012_df
        )


class ImputeLightGBM(Impute):
    """Does LightGBM (Light Gradient Boosting) Iterative imputation of missing data. LightGBM is an alternative to XGBoost that is around 7X faster and uses less memory, while still maintaining high accuracy. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If ``gridparams=None``\, a grid search is not performed, otherwise ``gridparams`` will be used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``, then the ``gridparams`` values can be lists of or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. Defaults to None.

        do_validation (bool, optional): Whether to validate the imputation if not doing a grid search. This validation method randomly replaces between 15% and 50% of the known, non-missing genotypes in ``n_features * column_subset`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None for ``do_validation`` to work. Calculating a validation score can be turned off altogether by setting ``do_validation`` to False. Defaults to False.

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search or validation. Should be between 0 and 1, and should also be small, because the grid search or validation takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        cv (int, optional): Number of folds for cross-validation during grid search. Defaults to 5.

        n_estimators (int, optional): The number of boosting rounds. Increasing this value can improve the fit, but at the cost of compute time and RAM usage. Defaults to 100.

        max_depth (int, optional): Maximum tree depth for base learners. Defaults to 3.

        learning_rate (float, optional): Boosting learning rate (eta). Basically, it serves as a weighting factor for correcting new trees when they are added to the model. Typical values are between 0.1 and 0.3. Lower learning rates generally find the best optimum at the cost of requiring far more compute time and resources. Defaults to 0.1.

        boosting_type (str, optional): Possible values: "gbdt", traditional Gradient Boosting Decision Tree. "dart", Dropouts meet Multiple Additive Regression Trees. "goss", Gradient-based One-Side Sampling. The "rf" option is not currently supported. Defaults to "gbdt".

        num_leaves (int, optional): Maximum tree leaves for base learners. Defaults to 31.

        subsample_for_bin (int, optional): Number of samples for constructing bins. Defaults to 200000.

        min_split_gain (float, optional): Minimum loss reduction required to make a further partition on a leaf node of the tree. Defaults to 0.0.

        min_child_weight (float, optional): Minimum sum of instance weight (hessian) needed in a child (leaf). Defaults to 1e-3.

        min_child_samples (int, optional): Minimum number of data needed in a child (leaf). Defaults to 20.

        subsample (float, optional): Subsample ratio of the training instance. Defaults to 1.0.

        subsample_freq (int, optional): Frequency of subsample, <=0 means no enable. Defaults to 0.

        colsample_bytree (float, optional): Subsample ratio of columns when constructing each tree. Defaults to 1.0.

        reg_lambda (float, optional): L2 regularization term on weights. Defaults to 0.

        reg_alpha (float, optional): L1 regularization term on weights. Defaults to 0.

        clf_random_state (int, numpy.random.RandomState object, or None, optional): Random number seed. If int, this number is used to seed the C++ code. If RandomState object (numpy), a random integer is picked based on its state to seed the C++ code. If None, default seeds in C++ code are used. Defaults to None.

        silent (bool, optional): Whether to print messages while running boosting. Defaults to True.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int or None, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: ???most_frequent???, "populations", "phylogeny", or "nmf". "most_frequent" uses the overall mode of each column. "populations" uses the mode per population/ per column via a population map file and the ``ImputeAlleleFreq`` class. "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. "nmf" performs the imputaton via matrix factorization via the ``ImputeNMF`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        gridsearch_method (str, optional): Grid search method to use. Supported options include: {"gridsearch", "randomized_gridsearch", and "genetic_algorithm"}. Whether to use a genetic algorithm for the grid search. "gridsearch" uses GridSearchCV to test every possible parameter combination. "randomized_gridsearch" picks ``grid_iter`` random combinations of parameters to test. "genetic_algorithm" uses a genetic algorithm via sklearn-genetic-opt GASearchCV to do the grid search. If set to None, then does not do a grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to "gridsearch".

        grid_iter (int, optional): Number of iterations for randomized and genetic algorithm grid searches. Defaults to 80.

        population_size (int or str, optional): For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io). Defaults to "auto".

        tournament_size (int, optional): For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search: If True takes the tournament_size best solution to the next generation. See GASearchCV documentation. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation. Defaults to 0.2.

        ga_algorithm (str, optional): For genetic algorithm grid search: Evolutionary algorithm to use. Supported options include: {"eaMuPlusLambda", "eaMuCommaLambda", "eaSimple"}. If you need to speed up the genetic algorithm grid search, try setting ``algorithm`` to "euSimple", at the expense of evolutionary model robustness. See more details in the DEAP algorithms documentation (https://deap.readthedocs.io). Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): If the genetic algorithm sees ``early_stop_gen`` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform. Defaults to 5.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``\%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors. Defaults to 1.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>from sklearn_genetic.space import Categorical, Integer, Continuous
        >>>
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>># Genetic Algorithm grid_params
        >>>grid_params = {
        >>>    "learning_rate": Continuous(lower=0.01, upper=0.1),
        >>>    "max_depth": Integer(2, 110),
        >>>}
        >>>
        >>>lgbm = ImputeLightGBM(
        >>>     genotype_data=data,
        >>>     gridparams=grid_params,
        >>>     cv=5,
        >>>     gridsearch_method="genetic_algorithm",
        >>>     n_nearest_features=10,
        >>>     n_estimators=100,
        >>>     initial_strategy="phylogeny",
        >>>)
        >>>
        >>>lgbm_gtdata = lgbm.imputed
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prefix: str = "output",
        gridparams: Optional[Dict[str, Any]] = None,
        do_validation: bool = False,
        column_subset: Union[int, float] = 0.1,
        cv: int = 5,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        subsample_for_bin: int = 200000,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 0.0,
        reg_alpha: float = 0.0,
        clf_random_state: Optional[Union[int, np.random.RandomState]] = None,
        silent: bool = True,
        n_nearest_features: Optional[int] = 10,
        max_iter: int = 10,
        tol: float = 1e-3,
        initial_strategy: str = "populations",
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        random_state: Optional[int] = None,
        gridsearch_method: str = "gridsearch",
        grid_iter: int = 80,
        population_size: Union[int, str] = "auto",
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.2,
        ga_algorithm: str = "eaMuPlusLambda",
        early_stop_gen: int = 5,
        scoring_metric: str = "accuracy",
        chunk_size: Union[int, float] = 1.0,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:

        # Get local variables into dictionary object
        kwargs = locals()

        if kwargs["boosting_type"] == "rf":
            raise ValueError("boosting_type 'rf' is not supported!")

        self.clf_type = "classifier"
        self.clf = lgbm.LGBMClassifier

        super().__init__(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = self.fit_predict(
            genotype_data.genotypes012_df
        )


class ImputeVAE(Impute):
    """Class to impute missing data using a Variational Autoencoder neural network model.

    Args:
        genotype_data (GenotypeData object): Input data initialized as GenotypeData object. Required positional argument.

        prefix (str): Prefix for output files. Defaults to "output".

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If ``gridparams=None``\, a grid search is not performed, otherwise ``gridparams`` will be used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``, then the ``gridparams`` values can be lists of or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). If it takes a long time, run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. Defaults to None (no gridsearch performed).

        validation_split (float, optional): Proportion of training dataset to set aside for loss validation during model training. Defaults to 0.2.

        column_subset (int or float, optional): If float is provided, gets the proportion of the dataset to randomly subset for the grid search or validation. Subsets ``int(n_features * column_subset)`` columns and Should be in the range [0, 1]. It can be small if the grid search or validation takes a long time. If int is provided, subset ``column_subset`` columns. Defaults to 1.0.

        epochs (int, optional): Number of epochs (cycles through the data) to run during training.Defaults to 100.

        batch_size (int, optional): Batch size to train the model with. Model training per epoch is performed over multiple subsets of samples (rows) of size ``batch_size``\. Defaults to 32.

        n_components (int, optional): Number of components (latent dimensions) to compress the input features to. Defaults to 3.

        early_stop_gen (int, optional): Only used with the genetic algorithm grid search option. Stop training early if the model sees ``early_stop_gen`` consecutive generations without improvement to the scoring metric. This can save training time by reducing the number of epochs and generations that are performed. Defaults to 25.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model. Adjust if overfitting or underfitting occurs. Defaults to 1.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in the hidden layers. If string or a list of strings is passed, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``\. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``\. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If multiple hidden layers are supplied, each subsequent layer's dimensions are further reduced by the "midpoint", "sqrt", or "log2". E.g., if using ``num_hidden_layers=3`` and ``n_components=2``\, and there are 100 features (columns), the hidden layer sizes for ``midpoint`` will be: [51, 27, 14]. If a single string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list. The list length must be equal to the ``num_hidden_layers`` and all hidden layer sizes must be > n_components. Defaults to "midpoint".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Supported activation functions include: ["elu", "selu", "leaky_relu", "prelu", "relu"]. Each activation function has some advantages and disadvantages and determines the curve and non-linearity of gradient descent. Some are also faster than others. See https://towardsdatascience.com/7-popular-activation-functions-you-should-know-in-deep-learning-and-how-to-use-them-with-keras-and-27b4d838dfe6 for more information. Note that using ``hidden_activation="selu"`` will force ``weights_initializer`` to be "lecun_normal". Defaults to "elu".

        optimizer (str, optional): The optimizer to use with gradient descent. Supported options are: "adam", "sgd", and "adagrad". See tf.keras.optimizers for more info. Defaults to "adam".

        learning_rate (float, optional): The learning rate for the optimizer. Adjust if the loss is learning too slowly or quickly. If you are getting overfitting, it is likely too high, and likewise underfitting can occur when the learning rate is too low. Defaults to 0.01.

        lr_patience (int, optional): Number of epochs without loss improvement to wait before reducing the learning rate. Defaults to 1.0.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply. Adjust if the model is over or underfitting. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        l2_penalty (float, optional) L2 regularization penalty to apply. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        dropout_rate (float, optional): Neuron dropout rate during training. Dropout randomly disables ``dropout_rate`` proportion of neurons during training, which can reduce overfitting. E.g., if dropout_rate is set to 0.2, then 20% of the neurons are randomly dropped out per epoch. Adjust if the model is over or underfitting. Must be a float in the range [0, 1]. . Defaults to 0.2.

        kl_beta (float, optional): Weight to apply to Kullback-Liebler divergence loss. If the latent distribution is not learned well, this weight can be adjusted to adjust how much KL divergence affects the total loss. Should be in the range [0, 1]. If set to 1.0, the KL loss is unweighted. If set to 0.0, the KL loss is negated entirely and does not affect the total loss. Defaults to 1.0.

        sample_weights (str, Dict[int, float], or None, optional): Weights for the 012-encoded classes during training. If None, then does not weight classes. If set to "auto", then class weights are automatically calculated for each column. If a dictionary is passed, it must contain 0, 1, and 2 as the keys and the class weights as the values. E.g., {0: 1.0, 1: 1.0, 2: 1.0}. The dictionary is then used as the overall class weights. If you wanted to prevent the model from learning to predict heterozygotes, for example, you could set the class weights to {0: 1.0, 1: 0.0, 2: 1.0}. Defaults to None (equal weighting).

        gridsearch_method (str, optional): Grid search method to use. Supported options include: {"gridsearch", "randomized_gridsearch", "genetic_algorithm"}. "gridsearch" uses GridSearchCV to test every possible parameter combination. "randomized_gridsearch" picks ``grid_iter`` random combinations of parameters to test. "genetic_algorithm" uses a genetic algorithm via the sklearn-genetic-opt GASearchCV module to do the grid search. If set to None, then does not do a grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to "gridsearch".

        grid_iter (int, optional): Number of iterations to use for randomized and genetic algorithm grid searches. For randomized grid search, ``grid_iter`` parameter combinations will be randomly sampled. For the genetic algorithm, this determines how many generations the genetic algorithm will run. Defaults to 80.

        scoring_metric (str, optional): Scoring metric to use for the grid search. Supported options include: {"auc_macro", "auc_micro", "precision_recall_macro", "precision_recall_micro", "accuracy", "hamming"}. Note that all metrics are automatically calculated when doing a grid search, the results of which are logged to a CSV file. However, when refitting following the grid search, the value passed to ``scoring_metric`` is used to select the best parameters. If you wish to choose the best parameters from a different metric, that information will also be in the CSV file. "auc_macro" and "auc_micro" get the AUC (area under curve) score for the ROC (Receiver Operating Characteristic) curve. The ROC curves plot the false positive rate (X-axis) versus the true positive rate (Y-axis) for each 012-encoded class and for the macro and micro averages among classes. The false positive rate is defined as: ``False Positive Rate = False Positives / (False Positives + True Negatives)`` and the true positive rate is defined as ``True Positive Rate = True Positives / (True Positives + False Negatives)``\. Macro averaging places equal importance on each class, whereas the micro average is the global average across all classes. AUC scores allow the ROC curve, and thus the model's classification skill, to be summarized as a single number. "precision_recall_macro" and "precision_recall_micro" create Precision-Recall (PR) curves for each class plus the macro and micro averages among classes. Precision is defined as ``True Positives / (True Positives + False Positives)`` and recall is defined as ``Recall = True Positives / (True Positives + False Negatives)``\. Reviewing both precision and recall is useful in cases where there is an imbalance in the observations between the two classes. For example, if there are many examples of major alleles (class 0) and only a few examples of minor alleles (class 2). PR curves take into account the use the Average Precision (AP) instead of AUC. AUC and AP are similar metrics, but AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each probability threshold, with the increase in recall from the previous threshold used as the weight. On the contrary, AUC uses linear interpolation with the trapezoidal rule to calculate the area under the curve. "accuracy" calculates ``number of correct predictions / total predictions``\, but can often be misleading when used without considering the model's classification skill for each class. Defaults to "auc_macro".

        population_size (int or str, optional): Only used for the genetic algorithm grid search. Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to "auto".

        tournament_size (int, optional): For genetic algorithm grid search only. Number of individuals to perform tournament selection. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search only. If set to True, takes the ``tournament_size`` best solution to the next generation. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search only. Probability of crossover operation between two individuals. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search only. Probability of child mutation. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to 0.2.

        ga_algorithm (str, optional): For genetic algorithm grid search only. Evolutionary algorithm to use. Supported options include: {"eaMuPlusLambda", "eaMuCommaLambda", "eaSimple"}. If you need to speed up the genetic algorithm grid search, try setting ``algorithm`` to "euSimple", at the expense of evolutionary model robustness. See more details in the DEAP algorithms documentation (https://deap.readthedocs.io). Defaults to "eaMuPlusLambda".

        sim_strategy (str, optional): Strategy to use for simulating missing data. Only used to validate the accuracy of the imputation. The final model will be trained with the non-simulated dataset. Supported options include: {"random", "nonrandom", "nonrandom_weighted"}. "random" randomly simulates missing data. When set to "nonrandom", branches from ``GenotypeData.guidetree`` will be randomly sampled to generate missing data on descendant nodes. For "nonrandom_weighted", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). If using the "nonrandom" or "nonrandom_weighted" options, a guide tree is required to have been initialized in the passed ``genotype_data`` object. Defaults to "random".

        sim_prop_missing (float, optional): Proportion of missing data to use with missing data simulation. Defaults to 0.1.

        disable_progressbar (bool, optional): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly when being written to a file. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

        n_jobs (int, optional): Number of parallel jobs to use in the grid search if ``gridparams`` is not None. -1 means use all available processors. Defaults to 1.

        verbose (int, optional): Verbosity flag. The higher, the more verbose. Possible values are 0, 1, or 2. 0 = silent, 1 = progress bar, 2 = one line per epoch. Note that the progress bar is not particularly useful when logged to a file, so verbose=0 or verbose=2 is recommended when not running interactively. Setting verbose higher than 0 is useful for initial runs and debugging, but can slow down training. Defaults to 0.

        kwargs (Dict[str, Any], optional): Possible options include: {"testing": True/False}. If testing is True, a confusion matrix plot will be created showing model performance. Arrays of the true and predicted values will also be printed to STDOUT. testing defaults to False.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.
        best_params (Dict[str, Any]): Best found parameters from grid search.

    Example:
        >>> data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>> )
        >>>
        >>> vae = ImputeVAE(
        >>>     genotype_data=data,
        >>>     learning_rate=0.001,
        >>>     epochs=200,
        >>> )
        >>>
        >>> vae_gtdata = vae.imputed
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix="imputer",
        gridparams=None,
        validation_split=0.2,
        column_subset=1.0,
        epochs=100,
        batch_size=32,
        n_components=3,
        early_stop_gen=25,
        num_hidden_layers=1,
        hidden_layer_sizes="midpoint",
        optimizer="adam",
        hidden_activation="elu",
        learning_rate=0.01,
        weights_initializer="glorot_normal",
        l1_penalty=1e-6,
        l2_penalty=1e-6,
        dropout_rate=0.2,
        kl_beta=1.0,
        sample_weights=None,
        gridsearch_method="gridsearch",
        grid_iter=80,
        scoring_metric="auc_macro",
        population_size="auto",
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.2,
        ga_algorithm="eaMuPlusLambda",
        sim_strategy="random",
        sim_prop_missing=0.1,
        disable_progressbar=False,
        n_jobs=1,
        verbose=0,
        **kwargs,
    ):

        # Get local variables into dictionary object
        all_kwargs = locals()

        self.clf = VAE
        self.clf_type = "classifier"

        imp_kwargs = {
            "str_encodings": {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        }

        all_kwargs.update(imp_kwargs)
        all_kwargs["testing"] = kwargs.get("testing", False)
        all_kwargs.pop("kwargs")

        super().__init__(self.clf, self.clf_type, all_kwargs)

        if genotype_data is None:
            raise TypeError("genotype_data cannot be NoneType")

        X = genotype_data.int_iupac

        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        self.imputed, self.best_params = self.fit_predict(df)


class ImputeStandardAutoEncoder(Impute):
    """Class to impute missing data using a standard Autoencoder (SAE) neural network model.

    Args:
        genotype_data (GenotypeData object): Input data initialized as GenotypeData object. Required positional argument.

        prefix (str): Prefix for output files. Defaults to "output".

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If ``gridparams=None``\, a grid search is not performed, otherwise ``gridparams`` will be used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``, then the ``gridparams`` values can be lists of or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). If it takes a long time, run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. Defaults to None (no gridsearch performed).

        validation_split (float, optional): Proportion of training dataset to set aside for loss validation during model training. Defaults to 0.2.

        column_subset (int or float, optional): If float is provided, gets the proportion of the dataset to randomly subset for the grid search or validation. Subsets ``int(n_features * column_subset)`` columns and Should be in the range [0, 1]. It can be small if the grid search or validation takes a long time. If int is provided, subset ``column_subset`` columns. Defaults to 1.0.

        epochs (int, optional): Number of epochs (cycles through the data) to run during training.Defaults to 100.

        batch_size (int, optional): Batch size to train the model with. Model training per epoch is performed over multiple subsets of samples (rows) of size ``batch_size``\. Defaults to 32.

        n_components (int, optional): Number of components (latent dimensions) to compress the input features to. Defaults to 3.

        early_stop_gen (int, optional): Only used with the genetic algorithm grid search option. Stop training early if the model sees ``early_stop_gen`` consecutive generations without improvement to the scoring metric. This can save training time by reducing the number of epochs and generations that are performed. Defaults to 25.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model. Adjust if overfitting or underfitting occurs. Defaults to 1.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in the hidden layers. If string or a list of strings is passed, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``\. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``\. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If multiple hidden layers are supplied, each subsequent layer's dimensions are further reduced by the "midpoint", "sqrt", or "log2". E.g., if using ``num_hidden_layers=3`` and ``n_components=2``\, and there are 100 features (columns), the hidden layer sizes for ``midpoint`` will be: [51, 27, 14]. If a single string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list. The list length must be equal to the ``num_hidden_layers`` and all hidden layer sizes must be > n_components. Defaults to "midpoint".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Supported activation functions include: ["elu", "selu", "leaky_relu", "prelu", "relu"]. Each activation function has some advantages and disadvantages and determines the curve and non-linearity of gradient descent. Some are also faster than others. See https://towardsdatascience.com/7-popular-activation-functions-you-should-know-in-deep-learning-and-how-to-use-them-with-keras-and-27b4d838dfe6 for more information. Note that using ``hidden_activation="selu"`` will force ``weights_initializer`` to be "lecun_normal". Defaults to "elu".

        optimizer (str, optional): The optimizer to use with gradient descent. Supported options are: "adam", "sgd", and "adagrad". See tf.keras.optimizers for more info. Defaults to "adam".

        learning_rate (float, optional): The learning rate for the optimizer. Adjust if the loss is learning too slowly or quickly. If you are getting overfitting, it is likely too high, and likewise underfitting can occur when the learning rate is too low. Defaults to 0.01.

        lr_patience (int, optional): Number of epochs without loss improvement to wait before reducing the learning rate. Defaults to 1.0.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply. Adjust if the model is over or underfitting. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        l2_penalty (float, optional) L2 regularization penalty to apply. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        dropout_rate (float, optional): Neuron dropout rate during training. Dropout randomly disables ``dropout_rate`` proportion of neurons during training, which can reduce overfitting. E.g., if dropout_rate is set to 0.2, then 20% of the neurons are randomly dropped out per epoch. Adjust if the model is over or underfitting. Must be a float in the range [0, 1]. . Defaults to 0.2.

        sample_weights (str, Dict[int, float], or None, optional): Weights for the 012-encoded classes during training. If None, then does not weight classes. If set to "auto", then class weights are automatically calculated for each column. If a dictionary is passed, it must contain 0, 1, and 2 as the keys and the class weights as the values. E.g., {0: 1.0, 1: 1.0, 2: 1.0}. The dictionary is then used as the overall class weights. If you wanted to prevent the model from learning to predict heterozygotes, for example, you could set the class weights to {0: 1.0, 1: 0.0, 2: 1.0}. Defaults to None (equal weighting).

        gridsearch_method (str, optional): Grid search method to use. Supported options include: {"gridsearch", "randomized_gridsearch", "genetic_algorithm"}. "gridsearch" uses GridSearchCV to test every possible parameter combination. "randomized_gridsearch" picks ``grid_iter`` random combinations of parameters to test. "genetic_algorithm" uses a genetic algorithm via the sklearn-genetic-opt GASearchCV module to do the grid search. If set to None, then does not do a grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to "gridsearch".

        grid_iter (int, optional): Number of iterations to use for randomized and genetic algorithm grid searches. For randomized grid search, ``grid_iter`` parameter combinations will be randomly sampled. For the genetic algorithm, this determines how many generations the genetic algorithm will run. Defaults to 80.

        scoring_metric (str, optional): Scoring metric to use for the grid search. Supported options include: {"auc_macro", "auc_micro", "precision_recall_macro", "precision_recall_micro", "accuracy"}. Note that all metrics are automatically calculated when doing a grid search, the results of which are logged to a CSV file. However, when refitting following the grid search, the value passed to ``scoring_metric`` is used to select the best parameters. If you wish to choose the best parameters from a different metric, that information will also be in the CSV file. "auc_macro" and "auc_micro" get the AUC (area under curve) score for the ROC (Receiver Operating Characteristic) curve. The ROC curves plot the false positive rate (X-axis) versus the true positive rate (Y-axis) for each 012-encoded class and for the macro and micro averages among classes. The false positive rate is defined as: ``False Positive Rate = False Positives / (False Positives + True Negatives)`` and the true positive rate is defined as ``True Positive Rate = True Positives / (True Positives + False Negatives)``\. Macro averaging places equal importance on each class, whereas the micro average is the global average across all classes. AUC scores allow the ROC curve, and thus the model's classification skill, to be summarized as a single number. "precision_recall_macro" and "precision_recall_micro" create Precision-Recall (PR) curves for each class plus the macro and micro averages among classes. Precision is defined as ``True Positives / (True Positives + False Positives)`` and recall is defined as ``Recall = True Positives / (True Positives + False Negatives)``\. Reviewing both precision and recall is useful in cases where there is an imbalance in the observations between the two classes. For example, if there are many examples of major alleles (class 0) and only a few examples of minor alleles (class 2). PR curves take into account the use the Average Precision (AP) instead of AUC. AUC and AP are similar metrics, but AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each probability threshold, with the increase in recall from the previous threshold used as the weight. On the contrary, AUC uses linear interpolation with the trapezoidal rule to calculate the area under the curve. "accuracy" calculates ``number of correct predictions / total predictions``\, but can often be misleading when used without considering the model's classification skill for each class. Defaults to "auc_macro".

        population_size (int or str, optional): Only used for the genetic algorithm grid search. Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to "auto".

        tournament_size (int, optional): For genetic algorithm grid search only. Number of individuals to perform tournament selection. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search only. If set to True, takes the ``tournament_size`` best solution to the next generation. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search only. Probability of crossover operation between two individuals. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search only. Probability of child mutation. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to 0.2.

        ga_algorithm (str, optional): For genetic algorithm grid search only. Evolutionary algorithm to use. Supported options include: {"eaMuPlusLambda", "eaMuCommaLambda", "eaSimple"}. If you need to speed up the genetic algorithm grid search, try setting ``algorithm`` to "euSimple", at the expense of evolutionary model robustness. See more details in the DEAP algorithms documentation (https://deap.readthedocs.io). Defaults to "eaMuPlusLambda".

        sim_strategy (str, optional): Strategy to use for simulating missing data. Only used to validate the accuracy of the imputation. The final model will be trained with the non-simulated dataset. Supported options include: {"random", "nonrandom", "nonrandom_weighted"}. "random" randomly simulates missing data. When set to "nonrandom", branches from ``GenotypeData.guidetree`` will be randomly sampled to generate missing data on descendant nodes. For "nonrandom_weighted", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). If using the "nonrandom" or "nonrandom_weighted" options, a guide tree is required to have been initialized in the passed ``genotype_data`` object. Defaults to "random".

        sim_prop_missing (float, optional): Proportion of missing data to use with missing data simulation. Defaults to 0.1.

        disable_progressbar (bool, optional): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly when being written to a file. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

        n_jobs (int, optional): Number of parallel jobs to use in the grid search if ``gridparams`` is not None. -1 means use all available processors. Defaults to 1.

        verbose (int, optional): Verbosity flag. The higher, the more verbose. Possible values are 0, 1, or 2. 0 = silent, 1 = progress bar, 2 = one line per epoch. Note that the progress bar is not particularly useful when logged to a file, so verbose=0 or verbose=2 is recommended when not running interactively. Setting verbose higher than 0 is useful for initial runs and debugging, but can slow down training. Defaults to 0.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.
        best_params (Dict[str, Any]): Best found parameters from grid search.

    Example:
        >>> data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>> )
        >>>
        >>> sae = ImputeStandardAutoEncoder(
        >>>     genotype_data=data,
        >>>     learning_rate=0.001,
        >>>     n_components=5,
        >>>     epochs=200,
        >>> )
        >>>
        >>> # Get the imputed data.
        >>> sae_gtdata = sae.imputed
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix="imputer",
        gridparams=None,
        validation_split=0.2,
        column_subset=1.0,
        epochs=100,
        batch_size=32,
        n_components=3,
        early_stop_gen=25,
        num_hidden_layers=1,
        hidden_layer_sizes="midpoint",
        hidden_activation="elu",
        optimizer="adam",
        learning_rate=0.01,
        lr_patience=1,
        weights_initializer="glorot_normal",
        l1_penalty=1e-6,
        l2_penalty=1e-6,
        dropout_rate=0.2,
        sample_weights=None,
        gridsearch_method="gridsearch",
        grid_iter=80,
        scoring_metric="auc_macro",
        population_size="auto",
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.2,
        ga_algorithm="eaMuPlusLambda",
        sim_strategy="random",
        sim_prop_missing=0.1,
        disable_progressbar=False,
        n_jobs=1,
        verbose=0,
    ):

        # Get local variables into dictionary object
        all_kwargs = locals()

        self.clf = SAE
        self.clf_type = "classifier"

        imp_kwargs = {
            "str_encodings": {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        }

        all_kwargs.update(imp_kwargs)

        super().__init__(self.clf, self.clf_type, all_kwargs)

        if genotype_data is None:
            raise TypeError("genotype_data cannot be NoneType")

        X = genotype_data.genotypes012_array

        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        self.imputed, self.best_params = self.fit_predict(df)


class ImputeUBP(Impute):
    """Class to impute missing data using an unsupervised backpropagation (UBP) neural network model.

    UBP [1]_ is an extension of NLPCA [2]_ with the input being randomly generated and of reduced dimensionality that gets trained to predict the supplied output based on only known values. It then uses the trained model to predict missing values. However, in contrast to NLPCA, UBP trains the model over three phases. The first is a single layer perceptron used to refine the randomly generated input. The second phase is a multi-layer perceptron that uses the refined reduced-dimension data from the first phase as input. In the second phase, the model weights are refined but not the input. In the third phase, the model weights and the inputs are then refined.

    Args:
        genotype_data (GenotypeData object): Input data initialized as GenotypeData object. Required positional argument.

        prefix (str): Prefix for output files. Defaults to "output".

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If ``gridparams=None``\, a grid search is not performed, otherwise ``gridparams`` will be used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``, then the ``gridparams`` values can be lists of or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). If it takes a long time, run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. Defaults to None (no gridsearch performed).

        validation_split (float, optional): Proportion of training dataset to set aside for loss validation during model training. Defaults to 0.2.

        column_subset (int or float, optional): If float is provided, gets the proportion of the dataset to randomly subset for the grid search or validation. Subsets ``int(n_features * column_subset)`` columns and Should be in the range [0, 1]. It can be small if the grid search or validation takes a long time. If int is provided, subset ``column_subset`` columns. Defaults to 1.0.

        epochs (int, optional): Number of epochs (cycles through the data) to run during training.Defaults to 100.

        batch_size (int, optional): Batch size to train the model with. Model training per epoch is performed over multiple subsets of samples (rows) of size ``batch_size``\. Defaults to 32.

        n_components (int, optional): Number of components (latent dimensions) to compress the input features to. Defaults to 3.

        early_stop_gen (int, optional): Only used with the genetic algorithm grid search option. Stop training early if the model sees ``early_stop_gen`` consecutive generations without improvement to the scoring metric. This can save training time by reducing the number of epochs and generations that are performed. Defaults to 25.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model. Adjust if overfitting or underfitting occurs. Defaults to 1.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in the hidden layers. If string or a list of strings is passed, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``\. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``\. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If multiple hidden layers are supplied, each subsequent layer's dimensions are further reduced by the "midpoint", "sqrt", or "log2". E.g., if using ``num_hidden_layers=3`` and ``n_components=2``\, and there are 100 features (columns), the hidden layer sizes for ``midpoint`` will be: [51, 27, 14]. If a single string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list. The list length must be equal to the ``num_hidden_layers`` and all hidden layer sizes must be > n_components. Defaults to "midpoint".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Supported activation functions include: ["elu", "selu", "leaky_relu", "prelu", "relu"]. Each activation function has some advantages and disadvantages and determines the curve and non-linearity of gradient descent. Some are also faster than others. See https://towardsdatascience.com/7-popular-activation-functions-you-should-know-in-deep-learning-and-how-to-use-them-with-keras-and-27b4d838dfe6 for more information. Note that using ``hidden_activation="selu"`` will force ``weights_initializer`` to be "lecun_normal". Defaults to "elu".

        optimizer (str, optional): The optimizer to use with gradient descent. Supported options are: "adam", "sgd", and "adagrad". See tf.keras.optimizers for more info. Defaults to "adam".

        learning_rate (float, optional): The learning rate for the optimizer. Adjust if the loss is learning too slowly or quickly. If you are getting overfitting, it is likely too high, and likewise underfitting can occur when the learning rate is too low. Defaults to 0.01.

        lr_patience (int, optional): Number of epochs without loss improvement to wait before reducing the learning rate. Defaults to 1.0.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply. Adjust if the model is over or underfitting. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        l2_penalty (float, optional) L2 regularization penalty to apply. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        dropout_rate (float, optional): Neuron dropout rate during training. Dropout randomly disables ``dropout_rate`` proportion of neurons during training, which can reduce overfitting. E.g., if dropout_rate is set to 0.2, then 20% of the neurons are randomly dropped out per epoch. Adjust if the model is over or underfitting. Must be a float in the range [0, 1]. . Defaults to 0.2.

        sample_weights (str, Dict[int, float], or None, optional): Weights for the 012-encoded classes during training. If None, then does not weight classes. If set to "auto", then class weights are automatically calculated for each column. If a dictionary is passed, it must contain 0, 1, and 2 as the keys and the class weights as the values. E.g., {0: 1.0, 1: 1.0, 2: 1.0}. The dictionary is then used as the overall class weights. If you wanted to prevent the model from learning to predict heterozygotes, for example, you could set the class weights to {0: 1.0, 1: 0.0, 2: 1.0}. Defaults to None (equal weighting).

        gridsearch_method (str, optional): Grid search method to use. Supported options include: {"gridsearch", "randomized_gridsearch", "genetic_algorithm"}. "gridsearch" uses GridSearchCV to test every possible parameter combination. "randomized_gridsearch" picks ``grid_iter`` random combinations of parameters to test. "genetic_algorithm" uses a genetic algorithm via the sklearn-genetic-opt GASearchCV module to do the grid search. If set to None, then does not do a grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to "gridsearch".

        grid_iter (int, optional): Number of iterations to use for randomized and genetic algorithm grid searches. For randomized grid search, ``grid_iter`` parameter combinations will be randomly sampled. For the genetic algorithm, this determines how many generations the genetic algorithm will run. Defaults to 80.

        scoring_metric (str, optional): Scoring metric to use for the grid search. Supported options include: {"auc_macro", "auc_micro", "precision_recall_macro", "precision_recall_micro", "accuracy"}. Note that all metrics are automatically calculated when doing a grid search, the results of which are logged to a CSV file. However, when refitting following the grid search, the value passed to ``scoring_metric`` is used to select the best parameters. If you wish to choose the best parameters from a different metric, that information will also be in the CSV file. "auc_macro" and "auc_micro" get the AUC (area under curve) score for the ROC (Receiver Operating Characteristic) curve. The ROC curves plot the false positive rate (X-axis) versus the true positive rate (Y-axis) for each 012-encoded class and for the macro and micro averages among classes. The false positive rate is defined as: ``False Positive Rate = False Positives / (False Positives + True Negatives)`` and the true positive rate is defined as ``True Positive Rate = True Positives / (True Positives + False Negatives)``\. Macro averaging places equal importance on each class, whereas the micro average is the global average across all classes. AUC scores allow the ROC curve, and thus the model's classification skill, to be summarized as a single number. "precision_recall_macro" and "precision_recall_micro" create Precision-Recall (PR) curves for each class plus the macro and micro averages among classes. Precision is defined as ``True Positives / (True Positives + False Positives)`` and recall is defined as ``Recall = True Positives / (True Positives + False Negatives)``\. Reviewing both precision and recall is useful in cases where there is an imbalance in the observations between the two classes. For example, if there are many examples of major alleles (class 0) and only a few examples of minor alleles (class 2). PR curves take into account the use the Average Precision (AP) instead of AUC. AUC and AP are similar metrics, but AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each probability threshold, with the increase in recall from the previous threshold used as the weight. On the contrary, AUC uses linear interpolation with the trapezoidal rule to calculate the area under the curve. "accuracy" calculates ``number of correct predictions / total predictions``\, but can often be misleading when used without considering the model's classification skill for each class. Defaults to "auc_macro".

        population_size (int or str, optional): Only used for the genetic algorithm grid search. Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to "auto".

        tournament_size (int, optional): For genetic algorithm grid search only. Number of individuals to perform tournament selection. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search only. If set to True, takes the ``tournament_size`` best solution to the next generation. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search only. Probability of crossover operation between two individuals. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search only. Probability of child mutation. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to 0.2.

        ga_algorithm (str, optional): For genetic algorithm grid search only. Evolutionary algorithm to use. Supported options include: {"eaMuPlusLambda", "eaMuCommaLambda", "eaSimple"}. If you need to speed up the genetic algorithm grid search, try setting ``algorithm`` to "euSimple", at the expense of evolutionary model robustness. See more details in the DEAP algorithms documentation (https://deap.readthedocs.io). Defaults to "eaMuPlusLambda".

        sim_strategy (str, optional): Strategy to use for simulating missing data. Only used to validate the accuracy of the imputation. The final model will be trained with the non-simulated dataset. Supported options include: {"random", "nonrandom", "nonrandom_weighted"}. "random" randomly simulates missing data. When set to "nonrandom", branches from ``GenotypeData.guidetree`` will be randomly sampled to generate missing data on descendant nodes. For "nonrandom_weighted", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). If using the "nonrandom" or "nonrandom_weighted" options, a guide tree is required to have been initialized in the passed ``genotype_data`` object. Defaults to "random".

        sim_prop_missing (float, optional): Proportion of missing data to use with missing data simulation. Defaults to 0.1.

        disable_progressbar (bool, optional): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly when being written to a file. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

        n_jobs (int, optional): Number of parallel jobs to use in the grid search if ``gridparams`` is not None. -1 means use all available processors. Defaults to 1.

        verbose (int, optional): Verbosity flag. The higher, the more verbose. Possible values are 0, 1, or 2. 0 = silent, 1 = progress bar, 2 = one line per epoch. Note that the progress bar is not particularly useful when logged to a file, so verbose=0 or verbose=2 is recommended when not running interactively. Setting verbose higher than 0 is useful for initial runs and debugging, but can slow down training. Defaults to 0.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.
        best_params (Dict[str, Any]): Best found parameters from grid search.

    Example:
        >>> data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>> )
        >>>
        >>> ubp = ImputeUBP(
        >>>     genotype_data=data,
        >>>     learning_rate=0.001,
        >>>     n_components=5
        >>> )
        >>>
        >>> # Get the imputed data.
        >>> ubp_gtdata = ubp.imputed


    References:
        .. [1] Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. Computational Intelligence, 32(2), 196-215.

        .. [2] Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. Bioinformatics, 21(20), 3887-3895.
    """

    nlpca = False

    def __init__(
        self,
        genotype_data,
        *,
        prefix="imputer",
        gridparams=None,
        column_subset=1.0,
        epochs=100,
        batch_size=32,
        n_components=3,
        early_stop_gen=25,
        num_hidden_layers=1,
        hidden_layer_sizes="midpoint",
        hidden_activation="elu",
        optimizer="adam",
        learning_rate=0.01,
        weights_initializer="glorot_normal",
        l1_penalty=1e-6,
        l2_penalty=1e-6,
        dropout_rate=0.2,
        sample_weights=None,
        gridsearch_method="gridsearch",
        grid_iter=80,
        scoring_metric="auc_macro",
        population_size="auto",
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.2,
        ga_algorithm="eaMuPlusLambda",
        sim_strategy="random",
        sim_prop_missing=0.1,
        disable_progressbar=False,
        n_jobs=1,
        verbose=0,
    ):

        # Get local variables into dictionary object
        settings = locals()
        settings["nlpca"] = self.nlpca

        self.clf = UBP
        self.clf_type = "classifier"
        if self.nlpca:
            self.clf.__name__ = "NLPCA"

        imp_kwargs = {
            "str_encodings": {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        }

        settings.update(imp_kwargs)

        if genotype_data is None:
            raise TypeError("genotype_data cannot be NoneType")

        X = genotype_data.genotypes012_array

        super().__init__(self.clf, self.clf_type, settings)

        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        self.imputed, self.best_params = self.fit_predict(df)


class ImputeNLPCA(ImputeUBP):
    """Class to impute missing data using inverse non-linear principal component analysis (NLPCA) neural network models.

    NLPCA [1]_ trains randomly generated, reduced-dimensionality input to predict the correct output. In the case of imputation, the model is trained only on known values, and the trained model is then used to predict the missing values.

    Args:
        genotype_data (GenotypeData object): Input data initialized as GenotypeData object. Required positional argument.

        prefix (str): Prefix for output files. Defaults to "output".

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If ``gridparams=None``\, a grid search is not performed, otherwise ``gridparams`` will be used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``, then the ``gridparams`` values can be lists of or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). If it takes a long time, run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. Defaults to None (no gridsearch performed).

        validation_split (float, optional): Proportion of training dataset to set aside for loss validation during model training. Defaults to 0.2.

        column_subset (int or float, optional): If float is provided, gets the proportion of the dataset to randomly subset for the grid search or validation. Subsets ``int(n_features * column_subset)`` columns and Should be in the range [0, 1]. It can be small if the grid search or validation takes a long time. If int is provided, subset ``column_subset`` columns. Defaults to 1.0.

        epochs (int, optional): Number of epochs (cycles through the data) to run during training.Defaults to 100.

        batch_size (int, optional): Batch size to train the model with. Model training per epoch is performed over multiple subsets of samples (rows) of size ``batch_size``\. Defaults to 32.

        n_components (int, optional): Number of components (latent dimensions) to compress the input features to. Defaults to 3.

        early_stop_gen (int, optional): Only used with the genetic algorithm grid search option. Stop training early if the model sees ``early_stop_gen`` consecutive generations without improvement to the scoring metric. This can save training time by reducing the number of epochs and generations that are performed. Defaults to 25.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model. Adjust if overfitting or underfitting occurs. Defaults to 1.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in the hidden layers. If string or a list of strings is passed, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``\. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``\. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If multiple hidden layers are supplied, each subsequent layer's dimensions are further reduced by the "midpoint", "sqrt", or "log2". E.g., if using ``num_hidden_layers=3`` and ``n_components=2``\, and there are 100 features (columns), the hidden layer sizes for ``midpoint`` will be: [51, 27, 14]. If a single string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list. The list length must be equal to the ``num_hidden_layers`` and all hidden layer sizes must be > n_components. Defaults to "midpoint".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Supported activation functions include: ["elu", "selu", "leaky_relu", "prelu", "relu"]. Each activation function has some advantages and disadvantages and determines the curve and non-linearity of gradient descent. Some are also faster than others. See https://towardsdatascience.com/7-popular-activation-functions-you-should-know-in-deep-learning-and-how-to-use-them-with-keras-and-27b4d838dfe6 for more information. Note that using ``hidden_activation="selu"`` will force ``weights_initializer`` to be "lecun_normal". Defaults to "elu".

        optimizer (str, optional): The optimizer to use with gradient descent. Supported options are: "adam", "sgd", and "adagrad". See tf.keras.optimizers for more info. Defaults to "adam".

        learning_rate (float, optional): The learning rate for the optimizer. Adjust if the loss is learning too slowly or quickly. If you are getting overfitting, it is likely too high, and likewise underfitting can occur when the learning rate is too low. Defaults to 0.01.

        lr_patience (int, optional): Number of epochs without loss improvement to wait before reducing the learning rate. Defaults to 1.0.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply. Adjust if the model is over or underfitting. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        l2_penalty (float, optional) L2 regularization penalty to apply. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        dropout_rate (float, optional): Neuron dropout rate during training. Dropout randomly disables ``dropout_rate`` proportion of neurons during training, which can reduce overfitting. E.g., if dropout_rate is set to 0.2, then 20% of the neurons are randomly dropped out per epoch. Adjust if the model is over or underfitting. Must be a float in the range [0, 1]. . Defaults to 0.2.

        sample_weights (str, Dict[int, float], or None, optional): Weights for the 012-encoded classes during training. If None, then does not weight classes. If set to "auto", then class weights are automatically calculated for each column. If a dictionary is passed, it must contain 0, 1, and 2 as the keys and the class weights as the values. E.g., {0: 1.0, 1: 1.0, 2: 1.0}. The dictionary is then used as the overall class weights. If you wanted to prevent the model from learning to predict heterozygotes, for example, you could set the class weights to {0: 1.0, 1: 0.0, 2: 1.0}. Defaults to None (equal weighting).

        gridsearch_method (str, optional): Grid search method to use. Supported options include: {"gridsearch", "randomized_gridsearch", "genetic_algorithm"}. "gridsearch" uses GridSearchCV to test every possible parameter combination. "randomized_gridsearch" picks ``grid_iter`` random combinations of parameters to test. "genetic_algorithm" uses a genetic algorithm via the sklearn-genetic-opt GASearchCV module to do the grid search. If set to None, then does not do a grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to "gridsearch".

        grid_iter (int, optional): Number of iterations to use for randomized and genetic algorithm grid searches. For randomized grid search, ``grid_iter`` parameter combinations will be randomly sampled. For the genetic algorithm, this determines how many generations the genetic algorithm will run. Defaults to 80.

        scoring_metric (str, optional): Scoring metric to use for the grid search. Supported options include: {"auc_macro", "auc_micro", "precision_recall_macro", "precision_recall_micro", "accuracy"}. Note that all metrics are automatically calculated when doing a grid search, the results of which are logged to a CSV file. However, when refitting following the grid search, the value passed to ``scoring_metric`` is used to select the best parameters. If you wish to choose the best parameters from a different metric, that information will also be in the CSV file. "auc_macro" and "auc_micro" get the AUC (area under curve) score for the ROC (Receiver Operating Characteristic) curve. The ROC curves plot the false positive rate (X-axis) versus the true positive rate (Y-axis) for each 012-encoded class and for the macro and micro averages among classes. The false positive rate is defined as: ``False Positive Rate = False Positives / (False Positives + True Negatives)`` and the true positive rate is defined as ``True Positive Rate = True Positives / (True Positives + False Negatives)``\. Macro averaging places equal importance on each class, whereas the micro average is the global average across all classes. AUC scores allow the ROC curve, and thus the model's classification skill, to be summarized as a single number. "precision_recall_macro" and "precision_recall_micro" create Precision-Recall (PR) curves for each class plus the macro and micro averages among classes. Precision is defined as ``True Positives / (True Positives + False Positives)`` and recall is defined as ``Recall = True Positives / (True Positives + False Negatives)``\. Reviewing both precision and recall is useful in cases where there is an imbalance in the observations between the two classes. For example, if there are many examples of major alleles (class 0) and only a few examples of minor alleles (class 2). PR curves take into account the use the Average Precision (AP) instead of AUC. AUC and AP are similar metrics, but AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each probability threshold, with the increase in recall from the previous threshold used as the weight. On the contrary, AUC uses linear interpolation with the trapezoidal rule to calculate the area under the curve. "accuracy" calculates ``number of correct predictions / total predictions``\, but can often be misleading when used without considering the model's classification skill for each class. Defaults to "auc_macro".

        population_size (int or str, optional): Only used for the genetic algorithm grid search. Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to "auto".

        tournament_size (int, optional): For genetic algorithm grid search only. Number of individuals to perform tournament selection. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search only. If set to True, takes the ``tournament_size`` best solution to the next generation. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search only. Probability of crossover operation between two individuals. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search only. Probability of child mutation. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to 0.2.

        ga_algorithm (str, optional): For genetic algorithm grid search only. Evolutionary algorithm to use. Supported options include: {"eaMuPlusLambda", "eaMuCommaLambda", "eaSimple"}. If you need to speed up the genetic algorithm grid search, try setting ``algorithm`` to "euSimple", at the expense of evolutionary model robustness. See more details in the DEAP algorithms documentation (https://deap.readthedocs.io). Defaults to "eaMuPlusLambda".

        sim_strategy (str, optional): Strategy to use for simulating missing data. Only used to validate the accuracy of the imputation. The final model will be trained with the non-simulated dataset. Supported options include: {"random", "nonrandom", "nonrandom_weighted"}. "random" randomly simulates missing data. When set to "nonrandom", branches from ``GenotypeData.guidetree`` will be randomly sampled to generate missing data on descendant nodes. For "nonrandom_weighted", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). If using the "nonrandom" or "nonrandom_weighted" options, a guide tree is required to have been initialized in the passed ``genotype_data`` object. Defaults to "random".

        sim_prop_missing (float, optional): Proportion of missing data to use with missing data simulation. Defaults to 0.1.

        disable_progressbar (bool, optional): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly when being written to a file. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

        n_jobs (int, optional): Number of parallel jobs to use in the grid search if ``gridparams`` is not None. -1 means use all available processors. Defaults to 1.

        verbose (int, optional): Verbosity flag. The higher, the more verbose. Possible values are 0, 1, or 2. 0 = silent, 1 = progress bar, 2 = one line per epoch. Note that the progress bar is not particularly useful when logged to a file, so verbose=0 or verbose=2 is recommended when not running interactively. Setting verbose higher than 0 is useful for initial runs and debugging, but can slow down training. Defaults to 0.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.
        best_params (Dict[str, Any]): Best found parameters from grid search.

    Example:
        >>> data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>> )
        >>>
        >>> nlpca = ImputeNLPCA(
        >>>     genotype_data=data,
        >>>     learning_rate=0.001,
        >>>     epochs=200
        >>> )
        >>>
        >>> nlpca_gtdata = nlpca.imputed

    References:
    .. [1] Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. Bioinformatics, 21(20), 3887-3895.
    """

    nlpca = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
