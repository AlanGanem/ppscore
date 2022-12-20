import numpy as np

from itertools import permutations, product
  

import warnings

from tqdm import tqdm

from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from scipy import sparse

import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_object_dtype,
    is_categorical_dtype,
    is_string_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)

from sklearn.model_selection import TimeSeriesSplit

from joblib import Parallel, delayed, effective_n_jobs

from preprocessing import RobustKBinsDiscretizer
from mutual_information import MutualInformationForest



def _feature_is_id(df, x):
    "Returns Boolean if the feature column x is an ID"
    if not _dtype_represents_categories(df[x]):
        return False

    category_count = df[x].value_counts().count()
    return category_count == len(df[x])


def _is_column_in_df(column, df):
    try:
        return column in df.columns
    except:
        return False


def _check_categoricals(df):
    numericals_msk = (df.dtypes == float)|(df.dtypes == int)
    categoricals = df.dtypes[~numericals_msk].index.tolist()
    return categoricals

def make_preprocess_pipeline(df, x):    
    
    categorical_columns = _check_categoricals(df[x])
    numerical_columns = list(set(x) - set(categorical_columns))
    
    if categorical_columns:
        steps = []
        casting = preprocessing.FunctionTransformer(lambda d: d.astype(str))
        encoder = preprocessing.OneHotEncoder(sparse = False, handle_unknown = "ignore")        
        steps.append(("str casting",casting,categorical_columns))        
        steps.append(("onehotencoding",encoder,categorical_columns))
    
        if numerical_columns:
            steps.append(("passthrough",preprocessing.FunctionTransformer(),numerical_columns))
        
        pipe = ColumnTransformer(steps)
    else:
        pipe = preprocessing.FunctionTransformer()
    
    return pipe

def _ensure_list(arr):
    
    if arr is None:
        arr = []
    if not isinstance(arr, str):
        if not hasattr(arr, "__len__"):
            arr = [arr]    
    if isinstance(arr, str):
        arr = [arr]
        
    return arr

def maybe_drop_nulls(df, x,y, drop_x_nulls, drop_y_nulls):
        
    subset = list(np.array(x+y)[[drop_x_nulls]*len(x) + [drop_y_nulls]*len(y)])
    if subset:
        return df.dropna(subset = subset, )
    else:
        return df

def maybe_sample(df, x, y, sample_size, replace = False, random_seed = None):
    
    if sample_size:
        if not (sample_size >= len(df)):
            return df.sample(sample_size = sample_size, replace = replace, random_state = random_seed)
        else:
            return df
    else:
        return df
    

def _score(
    df,
    x,
    y,
    conditional = None,
    drop_x_nulls = False,
    drop_y_nulls = True,    
    sample_size = 10_000,
    replace = False,
    random_seed=None,
    #crossvalscore params
    groups=None,
    scoring=None,
    cv=None,
    n_jobs_cv=None,
    verbose=0,
    fit_params=None,
    pre_dispatch='2*n_jobs',
    error_score="raise",
    #forest args
    n_estimators=1,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=200,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
    #quantization params    
    n_bins_y=10,
    strategy_y='kmeans',
    handle_nan_X = 'handle', #error, handle, ignore        
    handle_nan_y = 'error', #error, handle, ignore
    
):
    
    
    x = _ensure_list(x)
    y = _ensure_list(y)
    conditional = _ensure_list(conditional)

    
    df = maybe_drop_nulls(df, x+conditional, y, drop_x_nulls, drop_y_nulls)
    df = maybe_sample(df, x+conditional, y, sample_size, replace = replace, random_seed = random_seed)    
    
    
    estimator = MutualInformationForest(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=random_state,
        warm_start=warm_start,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        max_samples=max_samples,
        n_bins_y=n_bins_y,
        strategy_y=strategy_y,
        handle_nan_X = handle_nan_X,
        handle_nan_y = handle_nan_y,
        verbose = False
    )
    
    
    preprocess_pipeline = make_preprocess_pipeline(df, x)
    full_estim_pipeline = make_pipeline(preprocess_pipeline, estimator)        
    
    scores = cross_val_score(
        full_estim_pipeline,
        df[x], 
        df[y],
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs_cv,
        verbose=False,
        fit_params=fit_params,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
    )    
    
    
        
    scores = np.array(scores)    
    baseline_score = 0
    
    if conditional:
        preprocess_pipeline = make_preprocess_pipeline(df, x+conditional)
        full_estim_pipeline = make_pipeline(preprocess_pipeline, clone(estimator))                

        scores_both = cross_val_score(
            full_estim_pipeline,
            df[x+conditional], 
            df[y],
            groups=groups,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs_cv,
            verbose=False,
            fit_params=fit_params,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
        )
        scores_both = np.array(scores_both)
        
        preprocess_pipeline = make_preprocess_pipeline(df, conditional)
        full_estim_pipeline = make_pipeline(preprocess_pipeline, clone(estimator))        

        scores_cond = cross_val_score(
            full_estim_pipeline,
            df[conditional], 
            df[y],
            groups=groups,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs_cv,
            verbose=False,
            fit_params=fit_params,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
        )
        scores_cond = np.array(scores_cond)
        
        if np.mean(scores) > np.mean(scores_cond):
            baseline_scores = scores
        else:
            baseline_scores = scores_cond

            
        baseline_score = np.nanmean(baseline_scores)
        scores = scores_both - baseline_scores
        scores = np.where(scores > 0, scores, 0)
    
    return {
        "x":str(x),
        "y":str(y),
        "conditional":str(conditional),
        "scores":scores,
        "average_score":np.nanmean(scores),
        "baseline_score":baseline_score
    }
    

def score(
    df,
    x,
    y,
    conditional = None,
    catch_errors = True,
    drop_x_nulls = False,
    drop_y_nulls = True,    
    sample_size = 10_000,
    replace = False,
    random_seed = None,
    #crossvalscore params
    groups=None,
    scoring=None,
    cv=None,
    n_jobs_cv=None,
    verbose=0,
    fit_params=None,
    pre_dispatch='2*n_jobs',
    error_score="raise",
    #forest args
    n_estimators=1,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=200,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
    #quantization params    
    n_bins_y=10,
    strategy_y='kmeans',
    handle_nan_X = 'handle', #error, handle, ignore        
    handle_nan_y = 'error', #error, handle, ignore
):
    """
    Calculate the Predictive Power Score (PPS) for "x predicts y"
    The score always ranges from 0 to 1 and is data-type agnostic.

    A score of 0 means that the column x cannot predict the column y better than a naive baseline model.
    A score of 1 means that the column x can perfectly predict the column y given the model.
    A score between 0 and 1 states the ratio of how much potential predictive power the model achieved compared to the baseline model.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that contains the columns x and y
    x : str
        Name of the column x which acts as the feature
    y : str
        Name of the column y which acts as the target    
    conditional : str or None
        Name of the column conditional which the predictive power score will be calculated conditioned on, that is P(Y|X, Conditional)    
    n_bins_target: int
        n_bins to be passed to RobustKBinsDiscretizer(n_bins=n_bins_target, encode='ordinal', handle_nan = 'handle', strategy='quantile') for the target variable, to transform a regression problem into a classification one.     
    n_bins_independent: int or None
        n_bins to be passed to RobustKBinsDiscretizer(n_bins=n_bins_target, encode='ordinal', handle_nan = 'handle', strategy='quantile') for the dependent variable. it usefull when using a model like a DecisionTree without regularization.
        Binning avoids models with too much variance and thus potential overfit. Also helps in computation time when calculating for many features.
    average: str
        `average` arg passed to sklearn roc_auc_score on multiclass case (when binarizing the regression problem, it yields a multiclas classification problem)    
    sample_weight:
        sample_weight column name. Passed to model.fit and roc_auc_score.    
    sample : int or `None`
        Number of rows for sampling. The sampling decreases the calculation time of the PPS.
        If `None` there will be no sampling.
    
    dropna: ["all", "target", None]
        whether to keep only rows where all features are not null. 
            - `target` removes only null targets, keeping X nulls in place
            - `all` removes all nulls in x,y and possibly conditional
            - None keeps all nulls in all columns
            
    cross_validation : int
        Number of iterations during cross-validation. This has the following implications:
        For example, if the number is 4, then it is possible to detect patterns when there are at least 4 times the same observation. If the limit is increased, the required minimum observations also increase. This is important, because this is the limit when sklearn will throw an error and the PPS cannot be calculated
    random_seed : int or `None`
        Random seed for the parts of the calculation that require random numbers, e.g. shuffling or sampling.
        If the value is set, the results will be reproducible. If the value is `None` a new random number is drawn at the start of each calculation.
    invalid_score : any
        The score that is returned when a calculation is invalid, e.g. because the data type was not supported.
    catch_errors : bool
        If `True` all errors will be catched and reported as `unknown_error` which ensures convenience. If `False` errors will be raised. This is helpful for inspecting and debugging errors.

    Returns
    -------
    Dict
        A dict that contains multiple fields about the resulting PPS.
        The dict enables introspection into the calculations that have been performed under the hood
    """
    
    conditional = _ensure_list(conditional)
    y = _ensure_list(y)
    x = _ensure_list(x)        
    

    warnings.filterwarnings("ignore")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame"
        )
    

    if random_seed is None:
        from random import random
        random_seed = int(random() * 1000)

    try:
        return _score(         
            df=df,
            x=x,
            y=y,
            conditional=conditional,
            drop_x_nulls=drop_x_nulls,
            drop_y_nulls=drop_y_nulls,
            sample_size=sample_size,
            replace=replace,
            #=#,
            groups=groups,
            scoring=scoring,
            cv=cv,
            n_jobs_cv=n_jobs_cv,
            verbose=verbose,
            fit_params=fit_params,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            #=#,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            #=#,
            n_bins_y=n_bins_y,
            strategy_y=strategy_y,
            handle_nan_X=handle_nan_X,
            handle_nan_y=handle_nan_y,
        )
    except Exception as exception:
        if catch_errors:
            return {
                "x": str(x),
                "y": str(y),
                "conditional":str(conditional),
                "scores": [],
                "average_score": np.nan,             
                "baseline_score": np.nan,                
            }
        else:
            raise exception



def _format_list_of_dicts(scores, output, sorted):
    """
    Format list of score dicts `scores`
    - maybe sort by ppscore
    - maybe return pandas.Dataframe
    - output can be one of ["df", "list"]
    """
    if sorted:
        scores.sort(key=lambda item: item["average_score"], reverse=True)

    if output == "df":
        df_columns = [
            "x",
            "y",
            "conditional",
            "scores",            
            "average_score",            
            "baseline_score",            
        ]
        data = {column: [score[column] for score in scores] for column in df_columns}
        scores = pd.DataFrame.from_dict(data)

    return scores


def predictors(
    df,
    y,
    conditional = None,
    catch_errors = True,
    drop_x_nulls = False,
    drop_y_nulls = True,    
    sample_size = 10_000,
    replace = False,
    random_seed = None,
    #crossvalscore params
    groups=None,
    scoring=None,
    cv=None,
    n_jobs_cv=None,
    verbose=0,
    fit_params=None,
    pre_dispatch='2*n_jobs',
    error_score="raise",
    #forest args
    n_estimators=1,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=200,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
    #quantization params    
    n_bins_y=10,
    strategy_y='kmeans',
    handle_nan_X = 'handle', #error, handle, ignore        
    handle_nan_y = 'error', #error, handle, ignore
):
    """
    Calculate the Predictive Power Score (PPS) of all the features in the dataframe
    against a target column

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that contains the data
    y : str
        Name of the column y which acts as the target
    conditional : str or None
        Name of the column conditional which the predictive power score will be calculated conditioned on, that is P(Y|X, Conditional)    
    n_bins_target: int
        n_bins to be passed to RobustKBinsDiscretizer(n_bins=n_bins_target, encode='ordinal', handle_nan = 'handle', strategy='quantile') for the target variable, to transform a regression problem into a classification one.     
    n_bins_independent: int or None
        n_bins to be passed to RobustKBinsDiscretizer(n_bins=n_bins_target, encode='ordinal', handle_nan = 'handle', strategy='quantile') for the dependent variable. it usefull when using a model like a DecisionTree without regularization.
        Binning avoids models with too much variance and thus potential overfit. Also helps in computation time when calculating for many features.
    average: str
        `average` arg passed to sklearn roc_auc_score on multiclass case (when binarizing the regression problem, it yields a multiclas classification problem)    
    sample_weight:
        sample_weight column name. Passed to model.fit and roc_auc_score.    
    sample : int or `None`
        Number of rows for sampling. The sampling decreases the calculation time of the PPS.
        If `None` there will be no sampling.
    
    dropna: ["all", "target", None]
        whether to keep only rows where all features are not null. 
            - `target` removes only null targets, keeping X nulls in place
            - `all` removes all nulls in x,y and possibly conditional
            - None keeps all nulls in all columns
    
    cross_validation : int
        Number of iterations during cross-validation. This has the following implications:
        For example, if the number is 4, then it is possible to detect patterns when there are at least 4 times the same observation. If the limit is increased, the required minimum observations also increase. This is important, because this is the limit when sklearn will throw an error and the PPS cannot be calculated
    random_seed : int or `None`
        Random seed for the parts of the calculation that require random numbers, e.g. shuffling or sampling.
        If the value is set, the results will be reproducible. If the value is `None` a new random number is drawn at the start of each calculation.
    invalid_score : any
        The score that is returned when a calculation is invalid, e.g. because the data type was not supported.
    catch_errors : bool
        If `True` all errors will be catched and reported as `unknown_error` which ensures convenience. If `False` errors will be raised. This is helpful for inspecting and debugging errors.

    Returns
    -------
    pandas.DataFrame or list of Dict
        Either returns a tidy dataframe or a list of all the PPS dicts. This can be influenced
        by the output argument
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame"
        )
    
    
    conditional = _ensure_list(conditional)
    y = _ensure_list(y)
    
    if sum((drop_x_nulls,drop_y_nulls)) == 0:
        #sample first to avoid overhead of sampling sub dfs
        df = maybe_sample(df, sample_size, random_seed=random_seed)
    
    
    df_columns = set(df.columns)
    extra_cols = {i for i in conditional+[sample_weight, y] if not i is None}
    df_columns = list(df_columns - extra_cols)
    
    perms = [[i,y]+list(extra_cols) for i in df_columns]
    
    #perms = [(p,df[list(set(p))]) for p in perms]
            
    n_jobs = effective_n_jobs(n_jobs)
    
    scores = Parallel(n_jobs=n_jobs)(delayed(score)(
                df=df[list(set(_ensure_list(p)))],
                x=p[0],
                y=p[1],
                conditional=conditional,
                catch_errors=catch_errors,
                drop_x_nulls=drop_x_nulls,
                drop_y_nulls=drop_y_nulls,
                sample_size=sample_size,
                replace=replace,
                random_seed = random_seed,
                #=#,
                groups=groups,
                scoring=scoring,
                cv=cv,
                n_jobs_cv=n_jobs_cv,
                verbose=verbose,
                fit_params=fit_params,
                pre_dispatch=pre_dispatch,
                error_score=error_score,
                #=#,
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
                #=#,
                n_bins_y=n_bins_y,
                strategy_y=strategy_y,
                handle_nan_X=handle_nan_X,
                handle_nan_y=handle_nan_y,
        )        
        
        for p in tqdm(perms, position=0, leave=True, disable = not verbose)
    )
    
    return _format_list_of_dicts(scores=scores, output="df", sorted=True)



def matrix(
        df,
        conditional = None,
        catch_errors = True,
        drop_x_nulls = False,
        drop_y_nulls = True,    
        sample_size = 10_000,
        replace = False,
        random_seed = None,
        #crossvalscore params
        groups=None,
        scoring=None,
        cv=None,
        n_jobs_cv=None,
        verbose=0,
        fit_params=None,
        pre_dispatch='2*n_jobs',
        error_score="raise",
        #forest args
        n_estimators=1,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=200,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        #quantization params    
        n_bins_y=10,
        strategy_y='kmeans',
        handle_nan_X = 'handle', #error, handle, ignore        
        handle_nan_y = 'error', #error, handle, ignore

):
    
    """
    Calculate the Predictive Power Score (PPS) of all the features in the dataframe
    against a target column

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that contains the data
    y : str
        Name of the column y which acts as the target
    conditional : str or None
        Name of the column conditional which the predictive power score will be calculated conditioned on, that is P(Y|X, Conditional)    
    n_bins_target: int
        n_bins to be passed to RobustKBinsDiscretizer(n_bins=n_bins_target, encode='ordinal', handle_nan = 'handle', strategy='quantile') for the target variable, to transform a regression problem into a classification one.     
    n_bins_independent: int or None
        n_bins to be passed to RobustKBinsDiscretizer(n_bins=n_bins_target, encode='ordinal', handle_nan = 'handle', strategy='quantile') for the dependent variable. it usefull when using a model like a DecisionTree without regularization.
        Binning avoids models with too much variance and thus potential overfit. Also helps in computation time when calculating for many features.
    average: str
        `average` arg passed to sklearn roc_auc_score on multiclass case (when binarizing the regression problem, it yields a multiclas classification problem)    
    sample_weight:
        sample_weight column name. Passed to model.fit and roc_auc_score.    
    sample : int or `None`
        Number of rows for sampling. The sampling decreases the calculation time of the PPS.
        If `None` there will be no sampling.
    
    dropna: ["all", "target", None]
        whether to keep only rows where all features are not null. 
            - `target` removes only null targets, keeping X nulls in place
            - `all` removes all nulls in x,y and possibly conditional
            - None keeps all nulls in all columns
    
    cross_validation : int
        Number of iterations during cross-validation. This has the following implications:
        For example, if the number is 4, then it is possible to detect patterns when there are at least 4 times the same observation. If the limit is increased, the required minimum observations also increase. This is important, because this is the limit when sklearn will throw an error and the PPS cannot be calculated
    random_seed : int or `None`
        Random seed for the parts of the calculation that require random numbers, e.g. shuffling or sampling.
        If the value is set, the results will be reproducible. If the value is `None` a new random number is drawn at the start of each calculation.
    invalid_score : any
        The score that is returned when a calculation is invalid, e.g. because the data type was not supported.
    catch_errors : bool
        If `True` all errors will be catched and reported as `unknown_error` which ensures convenience. If `False` errors will be raised. This is helpful for inspecting and debugging errors.

    Returns
    -------
    pandas.DataFrame or list of Dict
        Either returns a tidy dataframe or a list of all the PPS dicts. This can be influenced
        by the output argument
    
    """
    
    conditional = _ensure_list(conditional)
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame"
        )
        
    if sum((drop_x_nulls,drop_y_nulls)) == 0:
        #sample first to avoid overhead of sampling sub dfs
        df = maybe_sample(df, sample_size, random_seed=random_seed)
        
    
    
            
    df_columns = set(df.columns)
    extra_cols = {i for i in conditional if not i is None}
    df_columns = list(df_columns - extra_cols)
    
    perms = [[*i]+list(extra_cols) for i in product(df_columns, repeat=2)]
    #perms = [(p,df[list(set(p))]) for p in perms]
            
    n_jobs = effective_n_jobs(n_jobs)
    
    scores = Parallel(n_jobs=n_jobs)(delayed(score)(
                df=df[list(set(_ensure_list(p)))],
                x=p[0],
                y=p[1],
                conditional=conditional,
                catch_errors=catch_errors,
                drop_x_nulls=drop_x_nulls,
                drop_y_nulls=drop_y_nulls,
                sample_size=sample_size,
                replace=replace,
                random_seed=random_seed,
                #=#,
                groups=groups,
                scoring=scoring,
                cv=cv,
                n_jobs_cv=n_jobs_cv,
                verbose=verbose,
                fit_params=fit_params,
                pre_dispatch=pre_dispatch,
                error_score=error_score,
                #=#,
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
                #=#,
                n_bins_y=n_bins_y,
                strategy_y=strategy_y,
                handle_nan_X=handle_nan_X,
                handle_nan_y=handle_nan_y,
                )         
        for p in tqdm(perms, position=0, leave=True, disable = not verbose)
    )
                                                                                                                      
    return _format_list_of_dicts(scores=scores, output="df", sorted=True)


