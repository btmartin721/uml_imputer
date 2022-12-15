# Standard library imports
import sys

# Third-party imports
import numpy as np
import pandas as pd

# Custom imports
try:
    from .impute import Impute
    from .unsupervised.neural_network_imputers import UBP, SAE
    from ..utils.misc import get_processor_name
except (ModuleNotFoundError, ValueError):
    from impute.impute import Impute
    from impute.unsupervised.neural_network_imputers import UBP, SAE
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


class ImputeAutoEncoder(Impute):
    """Class to impute missing data using an unsupervised Autoencoder neural network model.

    An autoencoder model trains an input to reconstruct itself by embedding the input into a latent dimension that is a reduced-dimensional representation of the data. The embedded latent dimension is then expanded out to a reconstruction of the data. Once the model is trained, it can then be used to predict unseen data. The uml_imputer model trains on known genotypes, and once the model is trained it is then used to predict the unknown genotypes.

    Args:
        genotype_data (GenotypeData object): Input data initialized as GenotypeData object. Required positional argument.

        gridsearch_method (str or None, optional): Grid search method to use. Supported options include: {None, "gridsearch", "randomized_gridsearch", "genetic_algorithm"}. A grid search will determine the optimal parameters as those that maximize the ``scoring_metric``\. If it takes a long time, try running it with a small subset of the data just to find the optimal parameters for the model, then run a full imputation using the optimal parameters.If set to None, then a grid search is not performed, and static parameters are used. If this parameter is not None, then the search parameters must be supplied to ``gridparams``\. Setting this parameter to "gridsearch" uses scikit-learn's GridSearchCV to test every possible parameter combination. "randomized_gridsearch" uses scikit-learn's RandomizedSearchCV with ``grid_iter`` random combinations of parameters to test, instead of every possible combination. "genetic_algorithm" uses a genetic algorithm via the sklearn-genetic-opt GASearchCV module to do the grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to None (no gridsearch performed).

        gridparams (Dict[str, Any], optional): Dictionary with the keys as the neural network model parameters and the values as lists (GridSearchCV and RandomizedSearchCV) or distributions (GASearchCV) of parameter values. ``gridparams`` is used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``\, then the ``gridparams`` values can be lists or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be lists or numpy arrays or can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. See the sklearn-genetic-opt documentation for more information. Defaults to None.

        prefix (str, optional): Prefix for output directory. Defaults to "imputer".

        validation_split (float, optional): Proportion of training dataset to set aside for calculating validation loss during model training. Defaults to 0.2.

        column_subset (int or float, optional): If float is provided, a proportion of the dataset columns is randomly subset as ``int(n_sites * column_subset)``\, and must be in the range [0, 1]. If int is provided, subset ``column_subset`` columns. Defaults to 1.0 (uses all columns).

        epochs (int, optional): Number of epochs (cycles through the data) to run during training. Defaults to 100.

        batch_size (int, optional): Batch size to train the model with. Model training per epoch is performed over multiple subsets of samples (rows) of size ``batch_size``\. Defaults to 32.

        n_components (int, optional): Number of components (latent dimensions) to use for embedding the input features. Defaults to 3.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model architecture. Adjust if overfitting or underfitting occurs. Defaults to 1.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of nodes to use in the hidden layers. If a string or a list of strings is passed, the strings must be equal to  "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``\. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units, ``sqrt(n_features)``\. If "log2" is supplied, the units will be calculated as ``log2(n_features)``\. ``hidden_layer_sizes`` will calculate and set the number of output units for each hidden layer. If multiple hidden layers are supplied, each subsequent layer's dimensions are further reduced by the "midpoint", "sqrt", or "log2". E.g., if using ``num_hidden_layers=3`` and ``n_components=2``\, and there are 100 features (columns), the hidden layer sizes for ``midpoint`` will be: [51, 27, 14]. If a single string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list. The list length must be equal to the ``num_hidden_layers`` and all hidden layer sizes must be > ``n_components``\. Defaults to "midpoint".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Supported activation functions include: {"elu", "selu", "leaky_relu", "prelu", "relu"}. Each activation function has some advantages and disadvantages and determines the curve and non-linearity of gradient descent. Some are also faster than others. See https://towardsdatascience.com/7-popular-activation-functions-you-should-know-in-deep-learning-and-how-to-use-them-with-keras-and-27b4d838dfe6 for more information. Note that using ``hidden_activation="selu"`` will force ``weights_initializer`` to be "lecun_normal". Defaults to "elu".

        optimizer (str, optional): The optimizer to use with gradient descent. Supported options are: {"adam", "sgd", and "adagrad"}. See tf.keras.optimizers for more info. Defaults to "adam".

        learning_rate (float, optional): The learning rate for the optimizer. Adjust if the loss is learning too slowly or quickly. If you are getting overfitting, it is likely too high, and likewise underfitting can occur when the learning rate is too low. Defaults to 0.01.

        lr_patience (int, optional): Number of epochs without loss improvement to wait before reducing the learning rate. Defaults to 1.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply. Adjust if the model is over or underfitting. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        l2_penalty (float, optional) L2 regularization penalty to apply. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        dropout_rate (float, optional): Neuron dropout rate during training. Dropout randomly disables ``dropout_rate`` proportion of neurons during training, which can reduce overfitting. E.g., if dropout_rate is set to 0.2, then 20% of the neurons are randomly dropped out per epoch. Adjust if the model is over or underfitting. Must be a float in the range [0, 1]. . Defaults to 0.2.

        sample_weights (str, Dict[int, float], or None, optional): Weights for the 012-encoded classes during training. If None, then does not weight classes. If set to "auto", then class weights are automatically calculated for each column. If a dictionary is passed, it must contain 0, 1, and 2 as the keys and the class weights as the values. E.g., {0: 1.0, 1: 1.0, 2: 1.0}. The dictionary is then used to set the overall class weights. If you wanted to prevent the model from learning to predict heterozygotes, for example, you could set the class weights to {0: 1.0, 1: 0.0, 2: 1.0}. Defaults to None (equal weighting).

        scoring_metric (str, optional): Scoring metric to use for the grid search. Supported options include: {"auc_macro", "auc_micro", "precision_recall_macro", "precision_recall_micro", "accuracy"}. Note that all metrics are automatically calculated when doing a grid search, the results of which are logged to a CSV file. However, when refitting following the grid search, the value passed to ``scoring_metric`` is used to select the best parameters. If you wish to choose the best parameters from a different metric, that information will also be in the CSV file. "auc_macro" and "auc_micro" get the AUC (area under curve) score for the ROC (Receiver Operating Characteristic) curve. The ROC curves plot the false positive rate (X-axis) versus the true positive rate (Y-axis) for each 012-encoded class and for the macro and micro averages among classes. The false positive rate is defined as: ``False Positive Rate = False Positives / (False Positives + True Negatives)`` and the true positive rate is defined as ``True Positive Rate = True Positives / (True Positives + False Negatives)``\. Macro averaging places equal importance on each class, whereas the micro average is the global average across all classes. AUC scores allow the ROC curve, and thus the model's classification skill, to be summarized as a single number. "precision_recall_macro" and "precision_recall_micro" create Precision-Recall (PR) curves for each class plus the macro and micro averages among classes. Precision is defined as ``True Positives / (True Positives + False Positives)`` and recall is defined as ``Recall = True Positives / (True Positives + False Negatives)``\. Reviewing both precision and recall is useful in cases where there is an imbalance in the observations between the two classes. For example, if there are many examples of major alleles (class 0) and only a few examples of minor alleles (class 2). PR curves take into account the use the Average Precision (AP) instead of AUC. AUC and AP are similar metrics, but AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each probability threshold, with the increase in recall from the previous threshold used as the weight. On the contrary, AUC uses linear interpolation with the trapezoidal rule to calculate the area under the curve. "accuracy" calculates ``number of correct predictions / total predictions``\, but can often be misleading when used without considering the model's classification skill for each class. Defaults to "auc_macro".

        grid_iter (int, optional): For randomized grid search only. Number of iterations to use for randomized and genetic algorithm grid searches. For randomized grid search, ``grid_iter`` parameter combinations will be randomly sampled. For the genetic algorithm, this determines how many generations the genetic algorithm will run. Defaults to 80.

        early_stop_gen (int, optional): For genetic algorithm grid search only. Stops training early if the model sees ``early_stop_gen`` consecutive generations without improvement to the ``scoring_metric``\. This can save training time by reducing the number of epochs and generations that are performed. Only used when ``gridsearch_method="genetic_algorithm"``\. Defaults to 25.

        population_size (int or str, optional): For genetic algorithm grid search only. Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to "auto".

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
        >>> sae = ImputeAutoEncoder(
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
        gridsearch_method="gridsearch",
        gridparams=None,
        prefix="imputer",
        validation_split=0.2,
        column_subset=1.0,
        epochs=100,
        batch_size=32,
        n_components=3,
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
        scoring_metric="auc_macro",
        grid_iter=80,
        early_stop_gen=25,
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

    UBP [1]_ is an extension of NLPCA [2]_ with the input being randomly generated and of reduced dimensionality that gets trained to predict the supplied output based on only known values. It then uses the trained model to predict missing values. However, in contrast to NLPCA, UBP trains the model over three phases. The first is a single layer perceptron used to refine the randomly generated input. The second phase is a multi-layer perceptron that uses the refined reduced-dimension data from the first phase as input. In the second phase, the model weights are refined but not the input. In the third phase, the model weights and the inputs are both refined.

    Args:
        genotype_data (GenotypeData object): Input data initialized as GenotypeData object. Required positional argument.

        gridsearch_method (str or None, optional): Grid search method to use. Supported options include: {None, "gridsearch", "randomized_gridsearch", "genetic_algorithm"}. A grid search will determine the optimal parameters as those that maximize the ``scoring_metric``\. If it takes a long time, try running it with a small subset of the data just to find the optimal parameters for the model, then run a full imputation using the optimal parameters.If set to None, then a grid search is not performed, and static parameters are used. If this parameter is not None, then the search parameters must be supplied to ``gridparams``\. Setting this parameter to "gridsearch" uses scikit-learn's GridSearchCV to test every possible parameter combination. "randomized_gridsearch" uses scikit-learn's RandomizedSearchCV with ``grid_iter`` random combinations of parameters to test, instead of every possible combination. "genetic_algorithm" uses a genetic algorithm via the sklearn-genetic-opt GASearchCV module to do the grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to None (no gridsearch performed).

        gridparams (Dict[str, Any], optional): Dictionary with the keys as the neural network model parameters and the values as lists (GridSearchCV and RandomizedSearchCV) or distributions (GASearchCV) of parameter values. ``gridparams`` is used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``\, then the ``gridparams`` values can be lists or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be lists or numpy arrays or can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. See the sklearn-genetic-opt documentation for more information. Defaults to None.

        prefix (str, optional): Prefix for output directory. Defaults to "imputer".

        validation_split (float, optional): Proportion of training dataset to set aside for calculating validation loss during model training. Defaults to 0.2.

        column_subset (int or float, optional): If float is provided, a proportion of the dataset columns is randomly subset as ``int(n_sites * column_subset)``\, and must be in the range [0, 1]. If int is provided, subset ``column_subset`` columns. Defaults to 1.0 (uses all columns).

        epochs (int, optional): Number of epochs (cycles through the data) to run during training. Defaults to 100.

        batch_size (int, optional): Batch size to train the model with. Model training per epoch is performed over multiple subsets of samples (rows) of size ``batch_size``\. Defaults to 32.

        n_components (int, optional): Number of components (latent dimensions) to use for embedding the input features. Defaults to 3.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model architecture. Adjust if overfitting or underfitting occurs. Defaults to 1.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of nodes to use in the hidden layers. If a string or a list of strings is passed, the strings must be equal to  "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``\. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units, ``sqrt(n_features)``\. If "log2" is supplied, the units will be calculated as ``log2(n_features)``\. ``hidden_layer_sizes`` will calculate and set the number of output units for each hidden layer. If multiple hidden layers are supplied, each subsequent layer's dimensions are further reduced by the "midpoint", "sqrt", or "log2". E.g., if using ``num_hidden_layers=3`` and ``n_components=2``\, and there are 100 features (columns), the hidden layer sizes for ``midpoint`` will be: [51, 27, 14]. If a single string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list. The list length must be equal to the ``num_hidden_layers`` and all hidden layer sizes must be > ``n_components``\. Defaults to "midpoint".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Supported activation functions include: {"elu", "selu", "leaky_relu", "prelu", "relu"}. Each activation function has some advantages and disadvantages and determines the curve and non-linearity of gradient descent. Some are also faster than others. See https://towardsdatascience.com/7-popular-activation-functions-you-should-know-in-deep-learning-and-how-to-use-them-with-keras-and-27b4d838dfe6 for more information. Note that using ``hidden_activation="selu"`` will force ``weights_initializer`` to be "lecun_normal". Defaults to "elu".

        optimizer (str, optional): The optimizer to use with gradient descent. Supported options are: {"adam", "sgd", and "adagrad"}. See tf.keras.optimizers for more info. Defaults to "adam".

        learning_rate (float, optional): The learning rate for the optimizer. Adjust if the loss is learning too slowly or quickly. If you are getting overfitting, it is likely too high, and likewise underfitting can occur when the learning rate is too low. Defaults to 0.01.

        lr_patience (int, optional): Number of epochs without loss improvement to wait before reducing the learning rate. Defaults to 1.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply. Adjust if the model is over or underfitting. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        l2_penalty (float, optional) L2 regularization penalty to apply. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        dropout_rate (float, optional): Neuron dropout rate during training. Dropout randomly disables ``dropout_rate`` proportion of neurons during training, which can reduce overfitting. E.g., if dropout_rate is set to 0.2, then 20% of the neurons are randomly dropped out per epoch. Adjust if the model is over or underfitting. Must be a float in the range [0, 1]. . Defaults to 0.2.

        sample_weights (str, Dict[int, float], or None, optional): Weights for the 012-encoded classes during training. If None, then does not weight classes. If set to "auto", then class weights are automatically calculated for each column. If a dictionary is passed, it must contain 0, 1, and 2 as the keys and the class weights as the values. E.g., {0: 1.0, 1: 1.0, 2: 1.0}. The dictionary is then used to set the overall class weights. If you wanted to prevent the model from learning to predict heterozygotes, for example, you could set the class weights to {0: 1.0, 1: 0.0, 2: 1.0}. Defaults to None (equal weighting).

        scoring_metric (str, optional): Scoring metric to use for the grid search. Supported options include: {"auc_macro", "auc_micro", "precision_recall_macro", "precision_recall_micro", "accuracy"}. Note that all metrics are automatically calculated when doing a grid search, the results of which are logged to a CSV file. However, when refitting following the grid search, the value passed to ``scoring_metric`` is used to select the best parameters. If you wish to choose the best parameters from a different metric, that information will also be in the CSV file. "auc_macro" and "auc_micro" get the AUC (area under curve) score for the ROC (Receiver Operating Characteristic) curve. The ROC curves plot the false positive rate (X-axis) versus the true positive rate (Y-axis) for each 012-encoded class and for the macro and micro averages among classes. The false positive rate is defined as: ``False Positive Rate = False Positives / (False Positives + True Negatives)`` and the true positive rate is defined as ``True Positive Rate = True Positives / (True Positives + False Negatives)``\. Macro averaging places equal importance on each class, whereas the micro average is the global average across all classes. AUC scores allow the ROC curve, and thus the model's classification skill, to be summarized as a single number. "precision_recall_macro" and "precision_recall_micro" create Precision-Recall (PR) curves for each class plus the macro and micro averages among classes. Precision is defined as ``True Positives / (True Positives + False Positives)`` and recall is defined as ``Recall = True Positives / (True Positives + False Negatives)``\. Reviewing both precision and recall is useful in cases where there is an imbalance in the observations between the two classes. For example, if there are many examples of major alleles (class 0) and only a few examples of minor alleles (class 2). PR curves take into account the use the Average Precision (AP) instead of AUC. AUC and AP are similar metrics, but AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each probability threshold, with the increase in recall from the previous threshold used as the weight. On the contrary, AUC uses linear interpolation with the trapezoidal rule to calculate the area under the curve. "accuracy" calculates ``number of correct predictions / total predictions``\, but can often be misleading when used without considering the model's classification skill for each class. Defaults to "auc_macro".

        grid_iter (int, optional): For randomized grid search only. Number of iterations to use for randomized and genetic algorithm grid searches. For randomized grid search, ``grid_iter`` parameter combinations will be randomly sampled. For the genetic algorithm, this determines how many generations the genetic algorithm will run. Defaults to 80.

        early_stop_gen (int, optional): For genetic algorithm grid search only. Stops training early if the model sees ``early_stop_gen`` consecutive generations without improvement to the ``scoring_metric``\. This can save training time by reducing the number of epochs and generations that are performed. Only used when ``gridsearch_method="genetic_algorithm"``\. Defaults to 25.

        population_size (int or str, optional): For genetic algorithm grid search only. Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to "auto".

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
        gridsearch_method=None,
        gridparams=None,
        prefix="imputer",
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

        gridsearch_method (str or None, optional): Grid search method to use. Supported options include: {None, "gridsearch", "randomized_gridsearch", "genetic_algorithm"}. A grid search will determine the optimal parameters as those that maximize the ``scoring_metric``\. If it takes a long time, try running it with a small subset of the data just to find the optimal parameters for the model, then run a full imputation using the optimal parameters.If set to None, then a grid search is not performed, and static parameters are used. If this parameter is not None, then the search parameters must be supplied to ``gridparams``\. Setting this parameter to "gridsearch" uses scikit-learn's GridSearchCV to test every possible parameter combination. "randomized_gridsearch" uses scikit-learn's RandomizedSearchCV with ``grid_iter`` random combinations of parameters to test, instead of every possible combination. "genetic_algorithm" uses a genetic algorithm via the sklearn-genetic-opt GASearchCV module to do the grid search. If doing a grid search, "randomized_search" takes the least amount of time because it does not have to test all parameters. "genetic_algorithm" takes the longest. See the scikit-learn GridSearchCV and RandomizedSearchCV documentation for the "gridsearch" and "randomized_gridsearch" options, and the sklearn-genetic-opt GASearchCV documentation for the "genetic_algorithm" option. Defaults to None (no gridsearch performed).

        gridparams (Dict[str, Any], optional): Dictionary with the keys as the neural network model parameters and the values as lists (GridSearchCV and RandomizedSearchCV) or distributions (GASearchCV) of parameter values. ``gridparams`` is used to specify parameter ranges or distributions for the grid search. If using ``gridsearch_method="gridsearch"``\, then the ``gridparams`` values can be lists or numpy arrays. If using ``gridsearch_method="randomized_gridsearch"``\, distributions can be lists or numpy arrays or can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). If using the genetic algorithm grid search by setting ``gridsearch_method="genetic_algorithm"``\, the parameters can be specified as ``sklearn_genetic.space`` objects. See the sklearn-genetic-opt documentation for more information. Defaults to None.

        prefix (str, optional): Prefix for output directory. Defaults to "imputer".

        validation_split (float, optional): Proportion of training dataset to set aside for calculating validation loss during model training. Defaults to 0.2.

        column_subset (int or float, optional): If float is provided, a proportion of the dataset columns is randomly subset as ``int(n_sites * column_subset)``\, and must be in the range [0, 1]. If int is provided, subset ``column_subset`` columns. Defaults to 1.0 (uses all columns).

        epochs (int, optional): Number of epochs (cycles through the data) to run during training. Defaults to 100.

        batch_size (int, optional): Batch size to train the model with. Model training per epoch is performed over multiple subsets of samples (rows) of size ``batch_size``\. Defaults to 32.

        n_components (int, optional): Number of components (latent dimensions) to use for embedding the input features. Defaults to 3.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model architecture. Adjust if overfitting or underfitting occurs. Defaults to 1.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of nodes to use in the hidden layers. If a string or a list of strings is passed, the strings must be equal to  "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``\. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units, ``sqrt(n_features)``\. If "log2" is supplied, the units will be calculated as ``log2(n_features)``\. ``hidden_layer_sizes`` will calculate and set the number of output units for each hidden layer. If multiple hidden layers are supplied, each subsequent layer's dimensions are further reduced by the "midpoint", "sqrt", or "log2". E.g., if using ``num_hidden_layers=3`` and ``n_components=2``\, and there are 100 features (columns), the hidden layer sizes for ``midpoint`` will be: [51, 27, 14]. If a single string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list. The list length must be equal to the ``num_hidden_layers`` and all hidden layer sizes must be > ``n_components``\. Defaults to "midpoint".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Supported activation functions include: {"elu", "selu", "leaky_relu", "prelu", "relu"}. Each activation function has some advantages and disadvantages and determines the curve and non-linearity of gradient descent. Some are also faster than others. See https://towardsdatascience.com/7-popular-activation-functions-you-should-know-in-deep-learning-and-how-to-use-them-with-keras-and-27b4d838dfe6 for more information. Note that using ``hidden_activation="selu"`` will force ``weights_initializer`` to be "lecun_normal". Defaults to "elu".

        optimizer (str, optional): The optimizer to use with gradient descent. Supported options are: {"adam", "sgd", and "adagrad"}. See tf.keras.optimizers for more info. Defaults to "adam".

        learning_rate (float, optional): The learning rate for the optimizer. Adjust if the loss is learning too slowly or quickly. If you are getting overfitting, it is likely too high, and likewise underfitting can occur when the learning rate is too low. Defaults to 0.01.

        lr_patience (int, optional): Number of epochs without loss improvement to wait before reducing the learning rate. Defaults to 1.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply. Adjust if the model is over or underfitting. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        l2_penalty (float, optional) L2 regularization penalty to apply. If this value is too high, underfitting can occur, and vice versa. Defaults to 1e-6.

        dropout_rate (float, optional): Neuron dropout rate during training. Dropout randomly disables ``dropout_rate`` proportion of neurons during training, which can reduce overfitting. E.g., if dropout_rate is set to 0.2, then 20% of the neurons are randomly dropped out per epoch. Adjust if the model is over or underfitting. Must be a float in the range [0, 1]. . Defaults to 0.2.

        sample_weights (str, Dict[int, float], or None, optional): Weights for the 012-encoded classes during training. If None, then does not weight classes. If set to "auto", then class weights are automatically calculated for each column. If a dictionary is passed, it must contain 0, 1, and 2 as the keys and the class weights as the values. E.g., {0: 1.0, 1: 1.0, 2: 1.0}. The dictionary is then used to set the overall class weights. If you wanted to prevent the model from learning to predict heterozygotes, for example, you could set the class weights to {0: 1.0, 1: 0.0, 2: 1.0}. Defaults to None (equal weighting).

        scoring_metric (str, optional): Scoring metric to use for the grid search. Supported options include: {"auc_macro", "auc_micro", "precision_recall_macro", "precision_recall_micro", "accuracy"}. Note that all metrics are automatically calculated when doing a grid search, the results of which are logged to a CSV file. However, when refitting following the grid search, the value passed to ``scoring_metric`` is used to select the best parameters. If you wish to choose the best parameters from a different metric, that information will also be in the CSV file. "auc_macro" and "auc_micro" get the AUC (area under curve) score for the ROC (Receiver Operating Characteristic) curve. The ROC curves plot the false positive rate (X-axis) versus the true positive rate (Y-axis) for each 012-encoded class and for the macro and micro averages among classes. The false positive rate is defined as: ``False Positive Rate = False Positives / (False Positives + True Negatives)`` and the true positive rate is defined as ``True Positive Rate = True Positives / (True Positives + False Negatives)``\. Macro averaging places equal importance on each class, whereas the micro average is the global average across all classes. AUC scores allow the ROC curve, and thus the model's classification skill, to be summarized as a single number. "precision_recall_macro" and "precision_recall_micro" create Precision-Recall (PR) curves for each class plus the macro and micro averages among classes. Precision is defined as ``True Positives / (True Positives + False Positives)`` and recall is defined as ``Recall = True Positives / (True Positives + False Negatives)``\. Reviewing both precision and recall is useful in cases where there is an imbalance in the observations between the two classes. For example, if there are many examples of major alleles (class 0) and only a few examples of minor alleles (class 2). PR curves take into account the use the Average Precision (AP) instead of AUC. AUC and AP are similar metrics, but AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each probability threshold, with the increase in recall from the previous threshold used as the weight. On the contrary, AUC uses linear interpolation with the trapezoidal rule to calculate the area under the curve. "accuracy" calculates ``number of correct predictions / total predictions``\, but can often be misleading when used without considering the model's classification skill for each class. Defaults to "auc_macro".

        grid_iter (int, optional): For randomized grid search only. Number of iterations to use for randomized and genetic algorithm grid searches. For randomized grid search, ``grid_iter`` parameter combinations will be randomly sampled. For the genetic algorithm, this determines how many generations the genetic algorithm will run. Defaults to 80.

        early_stop_gen (int, optional): For genetic algorithm grid search only. Stops training early if the model sees ``early_stop_gen`` consecutive generations without improvement to the ``scoring_metric``\. This can save training time by reducing the number of epochs and generations that are performed. Only used when ``gridsearch_method="genetic_algorithm"``\. Defaults to 25.

        population_size (int or str, optional): For genetic algorithm grid search only. Size of the initial population to sample randomly generated individuals. If set to "auto", then ``population_size`` is calculated as ``15 * n_parameters``\. If set to an integer, then uses the integer value as ``population_size``\. If you need to speed up the genetic algorithm grid search, try decreasing this parameter. See GASearchCV in the sklearn-genetic-opt documentation (https://sklearn-genetic-opt.readthedocs.io) for more info. Defaults to "auto".

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
