import numpy as np

from tqdm import tqdm

from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score, r2_score
from sklearn.base import clone

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



NOT_SUPPORTED_ANYMORE = "NOT_SUPPORTED_ANYMORE"
TO_BE_CALCULATED = -1

def cross_val_predict_time_series(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch='2*n_jobs',
    method='predict',
):
    """
    Allows cross_val_predict for TimeSeriesSplit cv splitter. it basically pads to NaN the samples that are not
    used to predictions (like the first fold of a out of time kfold). cv must be an instance of TimeSeriesSplit of sklearn or
    a a list of containers following the same format as cv.split(X) results.
    
    params descriptions can be found in cross_val_predict docs
    
    """
    
    if isinstance(cv, TimeSeriesSplit):
        splits = list(cv.split(X))           
    elif isinstance(cv, (list, tuple)):
        splits = cv    
    else:
        raise TypeError(f"cv must be instance of sklearn.model_selection._split.TimeSeriesSplit or container, got {type(cv)}")
    
    #ads train with first partiton and test with the datapoints that are not included, just to make it a valid permutation
    test_indices = np.concatenate([test for _, test in splits])
    missing_indexes = np.setdiff1d(np.arange(len(X)), test_indices)
    splits = splits + [(splits[0][0], missing_indexes)]
    
    results = cross_val_predict(
        estimator = estimator,
        X = X,
        y=y,
        groups=groups,
        cv=splits,
        n_jobs=n_jobs,
        verbose=verbose,
        fit_params=fit_params,
        pre_dispatch=pre_dispatch,
        method=method,
    )
    
    #fill with NaNs the missing indexes
    results = results.astype(float)
    results[missing_indexes] = np.nan
    return results



def _calculate_model_cv_score_(
    df, target, feature, conditional, n_bins_target, n_bins_independent, average, task, model, cross_validation, random_seed, sample_weight = None, cv_n_jobs = None,  **kwargs
):
    "Calculates the mean model score based on cross-validation"
    # Sources about the used methods:
    # https://scikit-learn.org/stable/modules/tree.html
    # https://scikit-learn.org/stable/modules/cross_validation.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
    
    #use custom crossval for time series if cv is TimeSeriesSplit
    if isinstance(cross_validation, TimeSeriesSplit):
        cv_func = cross_val_predict_time_series
    else:
        cv_func = cross_val_predict
        
    if conditional in [target,feature]:
        conditional = None
    metric = task["metric_key"]
    #model = task["model"]
    # shuffle the rows - this is important for cross-validation
    # because the cross-validation just takes the first n lines
    # if there is a strong pattern in the rows eg 0,0,0,0,1,1,1,1
    # then this will lead to problems because the first cv sees mostly 0 and the later 1
    # this approach might be wrong for timeseries because it might leak information
    #df = df.sample(frac=1, random_state=random_seed, replace=False)    
    scoring_method = None
    # preprocess target
    label_encoder = preprocessing.LabelEncoder()
    if task["type"] == "classification":        
        df[target] = label_encoder.fit_transform(df[target])
        target_series = df[target].values.flatten()
        scoring_method = "predict_proba"
    else:        
        if len(np.unique(df[target])) > n_bins_target:
            target_series = preprocessing.KBinsDiscretizer(n_bins=n_bins_target, encode='ordinal', strategy='quantile').fit_transform(df[[target]]).flatten()
        else:
            target_series = label_encoder.fit_transform(df[[target]]).flatten()
        scoring_method = "predict_proba"

    labels = np.unique(target_series)
    
    # preprocess feature
    if _dtype_represents_categories(df[feature]):
        one_hot_encoder = preprocessing.OneHotEncoder()
        array = df[feature].__array__()
        sparse_matrix = one_hot_encoder.fit_transform(array.reshape(-1, 1))
        feature_input = sparse_matrix
    else:
        # reshaping needed because there is only 1 feature
        array = df[feature].values
        if not isinstance(array, np.ndarray):  # e.g Int64 IntegerArray
            array = array.to_numpy()
        
        if not n_bins_independent is None:
            if len(np.unique(array)) > n_bins_independent:
                # binarize to avoid overfitting        
                feature_input = preprocessing.KBinsDiscretizer(n_bins=n_bins_independent, encode='ordinal', strategy='quantile').fit_transform(array.reshape(-1, 1))
            else:
                feature_input = array.reshape(-1, 1)
        else:
            feature_input = array.reshape(-1, 1)
        
    
    #preprocess conditional 
    if not conditional is None:
        if _dtype_represents_categories(df[conditional]):
            one_hot_encoder = preprocessing.OneHotEncoder()
            array = df[conditional].__array__()
            feature_input_cond = one_hot_encoder.fit_transform(array.reshape(-1, 1))
            feature_input = sparse.hstack([feature_input_cond, feature_input])
            
        else:
            # reshaping needed because there is only 1 feature
            array = df[conditional].values
            if not isinstance(array, np.ndarray):  # e.g Int64 IntegerArray
                array = array.to_numpy()
            
            if not n_bins_independent is None:
                
                if len(np.unique(array)) > n_bins_independent:
                    # binarize to avoid overfitting        
                    feature_input_cond = preprocessing.KBinsDiscretizer(n_bins=n_bins_independent, encode='ordinal', strategy='quantile').fit_transform(array.reshape(-1, 1))
                else:
                    feature_input_cond = array.reshape(-1, 1)
            else:
                feature_input_cond = array.reshape(-1, 1)
            
            
            if sparse.issparse(feature_input):
                feature_input = sparse.hstack([feature_input, feature_input_cond])
            else:
                feature_input = np.hstack([feature_input, feature_input_cond])
        
        
        

    # Cross-validation is stratifiedKFold for classification, KFold for regression
    # CV on one core (n_job=1; default) has shown to be fastest
    sample_weight = None if sample_weight is None else df[sample_weight].values.flatten()
    fit_params = {"sample_weight":sample_weight} if not sample_weight is None else None

    preds = cv_func(
            clone(model), feature_input, target_series.flatten(), cv=cross_validation, n_jobs=cv_n_jobs,
            fit_params=fit_params, pre_dispatch='2*n_jobs', method=scoring_method)
    
    if (len(labels) <= 2) and (preds.ndim > 1):
        if preds.shape[1] == 2:
            preds = preds[:,1]
        elif preds.shape[1] == 1:
            preds = preds[:,0]

    preds_nan_msk = ~np.isnan(preds)
    if len(preds_nan_msk.shape) > 1:
        preds_nan_msk = ~np.any(~preds_nan_msk, axis = 1)
        
    model_score = roc_auc_score(
            target_series[preds_nan_msk],
            preds[preds_nan_msk],
            average=average,
            sample_weight=sample_weight if sample_weight is None else sample_weight[preds_nan_msk],
            max_fpr=None,
            multi_class="ovr",
            labels=None
        )

    baseline_score = 0.5
    
    if not conditional is None:
        cond_preds = cv_func(
                clone(model), feature_input_cond, target_series.flatten(), cv=cross_validation, n_jobs=cv_n_jobs,
                fit_params=fit_params, pre_dispatch='2*n_jobs', method=scoring_method)


        cond_preds_nan_msk = ~np.isnan(cond_preds)
        if len(cond_preds_nan_msk.shape) > 1:
            cond_preds_nan_msk = ~np.any(~cond_preds_nan_msk, axis = 1)
            
        if (len(labels) <= 2) and (cond_preds.ndim > 1):
            if cond_preds.shape[1] == 2:
                cond_preds = cond_preds[:,1]
            elif cond_preds.shape[1] == 1:
                cond_preds = cond_preds.flatten()
        
        cond_score = roc_auc_score(
                target_series[cond_preds_nan_msk],
                cond_preds[cond_preds_nan_msk],
                average=average,
                sample_weight=sample_weight if sample_weight is None else sample_weight[cond_preds_nan_msk],
                max_fpr=None,
                multi_class="ovr",
                labels=None
            )
        
        baseline_score = cond_score #baseline is the 
        #subtract cond effect from joint effect
        #model_score = model_score - max(0, cond_score - 0.5)

        
    
    return model_score, baseline_score


def _normalized_r2_score(model_r2, naive_r2):
    "Normalizes the model R2 score, given the baseline score"
    if model_r2 < naive_r2:
        return 0
    else:
        return model_r2


def _r2_normalizer(df, y, model_score, **kwargs):
    "In case of MAE, calculates the baseline score for y and derives the PPS."        
    baseline_score = 0
    ppscore = _normalized_r2_score(model_score, baseline_score)
    return ppscore, baseline_score


    
def _normalized_auc_score(model_auc, baseline_auc):
    "Normalizes the model auc score, given the baseline score"
    # # AUC ranges from 0 to 1
    # # 1 is best
    if model_auc < baseline_auc:
        return 0        
    else:
        scale_range = 1.0 - baseline_auc  # eg 0.3        
        auc_diff = model_auc - baseline_auc
        return auc_diff / scale_range  # 0.1/0.3 = 0.33




def _auc_normalizer(df, y, model_score, baseline_score = 0.5, **kwargs):
    "calculates the baseline score for y and derives the PPS. Custom score should follow the same API as f1_score"    
    ppscore = _normalized_auc_score(model_score, baseline_score)
    return ppscore, baseline_score


VALID_CALCULATIONS = {
    "regression": {
        "type": "regression",
        "is_valid_score": True,
        "model_score": TO_BE_CALCULATED,
        "baseline_score": TO_BE_CALCULATED,
        "ppscore": TO_BE_CALCULATED,
        "unnormalized_ppscore":TO_BE_CALCULATED,
        "metric_name": "weighted AUC",
        "metric_key": "roc_auc",
        "model": tree.DecisionTreeClassifier(),
        "score_normalizer": _auc_normalizer,
    },
    "classification": {
        "type": "classification",
        "is_valid_score": True,
        "model_score": TO_BE_CALCULATED,
        "baseline_score": TO_BE_CALCULATED,
        "ppscore": TO_BE_CALCULATED,
        "unnormalized_ppscore":TO_BE_CALCULATED,
        "metric_name": "weighted AUC",
        "metric_key": "roc_auc",
        "model": tree.DecisionTreeClassifier(),
        "score_normalizer": _auc_normalizer,
    },
    "predict_itself": {
        "type": "predict_itself",
        "is_valid_score": True,
        "model_score": 1,
        "baseline_score": 0,
        "ppscore": 1,
        "unnormalized_ppscore":1,
        "metric_name": None,
        "metric_key": None,
        "model": None,
        "score_normalizer": None,
    },
    "target_is_constant": {
        "type": "target_is_constant",
        "is_valid_score": True,
        "model_score": 1,
        "baseline_score": 1,
        "ppscore": 0,
        "unnormalized_ppscore":0,
        "metric_name": None,
        "metric_key": None,
        "model": None,
        "score_normalizer": None,
    },
    "target_is_id": {
        "type": "target_is_id",
        "is_valid_score": True,
        "model_score": 0,
        "baseline_score": 0,
        "ppscore": 0,
        "unnormalized_ppscore":0,
        "metric_name": None,
        "metric_key": None,
        "model": None,
        "score_normalizer": None,
    },
    "feature_is_id": {
        "type": "feature_is_id",
        "is_valid_score": True,
        "model_score": 0,
        "baseline_score": 0,
        "ppscore": 0,
        "unnormalized_ppscore":0,
        "metric_name": None,
        "metric_key": None,
        "model": None,
        "score_normalizer": None,
    },
}

INVALID_CALCULATIONS = [
    "target_is_datetime",
    "target_data_type_not_supported",
    "empty_dataframe_after_dropping_na",
    "unknown_error",
]


def _dtype_represents_categories(series) -> bool:
    "Determines if the dtype of the series represents categorical values"
    return (
        is_bool_dtype(series)
        or is_object_dtype(series)
        or is_string_dtype(series)
        or is_categorical_dtype(series)
    )


def _determine_case_and_prepare_df(df, x, y, sample_weight = None,  conditional = None, sample=5_000, dropna = True, random_seed=123):
    "Returns str with the name of the determined case based on the columns x and y"
    if x == y:
        return df, "predict_itself"

    cols = [x, y]
    
    if conditional is None:
        pass
    else:
        if y == conditional:            
            conditional = None
            pass
        else:
            cols = cols + [conditional]
    
    if sample_weight is None:
        pass
    else:
        cols = cols + [sample_weight]
    
    df = df[cols]
    # IDEA: log.warning when values have been dropped
    # dro duplciated columns
    df = df.loc[:,~df.columns.duplicated()].copy()
    if dropna:
        df = df.dropna()    
    else:
        pass
    
    if len(df) == 0:
        return df, "empty_dataframe_after_dropping_na"
        # IDEA: show warning
        # raise Exception(
        #     "After dropping missing values, there are no valid rows left"
        # )

    df = _maybe_sample(df, sample, random_seed=random_seed)

    if _feature_is_id(df, x):
        return df, "feature_is_id"

    category_count = df[y].value_counts().count()
    if category_count == 1:
        # it is helpful to separate this case in order to save unnecessary calculation time
        return df, "target_is_constant"
    if _dtype_represents_categories(df[y]) and (category_count == len(df[y])):
        # it is important to separate this case in order to save unnecessary calculation time
        return df, "target_is_id"

    if _dtype_represents_categories(df[y]):
        return df, "classification"
    if is_numeric_dtype(df[y]):
        # this check needs to be after is_bool_dtype (which is part of _dtype_represents_categories) because bool is considered numeric by pandas
        return df, "regression"

    if is_datetime64_any_dtype(df[y]) or is_timedelta64_dtype(df[y]):
        # IDEA: show warning
        # raise TypeError(
        #     f"The target column {y} has the dtype {df[y].dtype} which is not supported. A possible solution might be to convert {y} to a string column"
        # )
        return df, "target_is_datetime"

    # IDEA: show warning
    # raise Exception(
    #     f"Could not infer a valid task based on the target {y}. The dtype {df[y].dtype} is not yet supported"
    # )  # pragma: no cover
    return df, "target_data_type_not_supported"


def _feature_is_id(df, x):
    "Returns Boolean if the feature column x is an ID"
    if not _dtype_represents_categories(df[x]):
        return False

    category_count = df[x].value_counts().count()
    return category_count == len(df[x])


def _maybe_sample(df, sample, random_seed=None):
    """
    Maybe samples the rows of the given df to have at most `sample` rows
    If sample is `None` or falsy, there will be no sampling.
    If the df has fewer rows than the sample, there will be no sampling.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that might be sampled
    sample : int or `None`
        Number of rows to be sampled
    random_seed : int or `None`
        Random seed that is forwarded to pandas.DataFrame.sample as `random_state`

    Returns
    -------
    pandas.DataFrame
        DataFrame after potential sampling
    """
    if sample and (len(df) > sample):
        # this is a problem if x or y have more than sample=5000 categories
        # TODO: dont sample when the problem occurs and show warning
        np.random.seed(random_seed)
        ixs = np.random.choice(np.arange(len(df)), replace=False, size = sample)
        ixs = np.sort(ixs)
        #order idxs to keep ordering of df
        df = df.iloc[ixs]
    return df


def _is_column_in_df(column, df):
    try:
        return column in df.columns
    except:
        return False


def _score(
    df, x, y, conditional,n_bins_target, n_bins_independent, average, task, sample, dropna, model, cross_validation, random_seed, invalid_score, catch_errors, sample_weight,cv_n_jobs, **kwargs
):
    df, case_type = _determine_case_and_prepare_df(
        df, x, y, sample_weight = sample_weight, conditional = conditional, sample=sample, dropna= dropna, random_seed=random_seed
    )
    task = _get_task(case_type, invalid_score)
    
    if case_type in ["classification", "regression"]:
        model_score, baseline_score = _calculate_model_cv_score_(
            df,
            target=y,
            feature=x,
            conditional=conditional,
            n_bins_target = n_bins_target,
            n_bins_independent = n_bins_independent,
            average = average,
            task=task,
            model=model,
            cross_validation=cross_validation,
            random_seed=random_seed,
            sample_weight = sample_weight,
            cv_n_jobs = cv_n_jobs,
        )
        # IDEA: the baseline_scores do sometimes change significantly, e.g. for F1 and thus change the PPS
        # we might want to calculate the baseline_score 10 times and use the mean in order to have less variance
        unnormalized_ppscore = max(0, model_score - baseline_score)
        ppscore, baseline_score = task["score_normalizer"](
            df, y, model_score, baseline_score=baseline_score, random_seed=random_seed
        )
    else:
        model_score = task["model_score"]
        baseline_score = task["baseline_score"]
        ppscore = task["ppscore"]
        unnormalized_ppscore = max(0, model_score - baseline_score)

    return {
        "x": x,
        "y": y,
        "conditional":conditional,
        "ppscore": ppscore,
        "unnormalized_ppscore":unnormalized_ppscore,
        "case": case_type,
        "is_valid_score": task["is_valid_score"],
        "metric": task["metric_name"],
        "baseline_score": baseline_score,
        "model_score": abs(model_score),  # sklearn returns negative mae
        "model": model,
    }


def score(
    df,
    x,
    y,
    sample_weight = None,
    conditional = None,
    n_bins_target = 10,
    n_bins_independent = None,
    average = "weighted",    
    task=NOT_SUPPORTED_ANYMORE,
    model=tree.DecisionTreeClassifier(),
    sample=5_000,
    dropna=True,
    cross_validation=4,
    random_seed=123,
    invalid_score=0,
    catch_errors=True,
    cv_n_jobs = None,
    **kwargs,
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
        n_bins to be passed to KBinsDiscretizer for the target variable, to transform a regression problem into a classification one.     
    n_bins_independent: int or None
        n_bins to be passed to KBinsDiscretizer for the dependent variable. it usefull when using a model like a DecisionTree without regularization.
        Binning avoids models with too much variance and thus potential overfit. Also helps in computation time when calculating for many features.
    average: str
        `average` arg passed to sklearn roc_auc_score on multiclass case (when binarizing the regression problem, it yields a multiclas classification problem)    
    sample_weight:
        sample_weight column name. Passed to model.fit and roc_auc_score.    
    sample : int or `None`
        Number of rows for sampling. The sampling decreases the calculation time of the PPS.
        If `None` there will be no sampling.
    dropna: bool
        whether to keep only rows where all features are not null        
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

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame"
        )
    if not _is_column_in_df(x, df):
        raise ValueError(
            f"The 'x' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe"
        )
    if len(df[[x]].columns) >= 2:
        raise AssertionError(
            f"The dataframe has {len(df[[x]].columns)} columns with the same column name {x}\nPlease adjust the dataframe and make sure that only 1 column has the name {x}"
        )
    if not _is_column_in_df(y, df):
        raise ValueError(
            f"The 'y' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe"
        )
    if len(df[[y]].columns) >= 2:
        raise AssertionError(
            f"The dataframe has {len(df[[y]].columns)} columns with the same column name {y}\nPlease adjust the dataframe and make sure that only 1 column has the name {y}"
        )
    if task is not NOT_SUPPORTED_ANYMORE:
        raise AttributeError(
            "The attribute 'task' is no longer supported because it led to confusion and inconsistencies.\nThe task of the model is now determined based on the data types of the columns. If you want to change the task please adjust the data type of the column.\nFor more details, please refer to the README"
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
            n_bins_target=n_bins_target,
            n_bins_independent=n_bins_independent,
            average=average,
            task=task,
            sample=sample,
            dropna=dropna,
            model=model,
            cross_validation=cross_validation,
            random_seed=random_seed,
            invalid_score=invalid_score,
            catch_errors=catch_errors,
            sample_weight = sample_weight,
            cv_n_jobs=cv_n_jobs,
        )
    except Exception as exception:
        if catch_errors:
            case_type = exception#"unknown_error"
            task = _get_task(case_type="unknown_error", invalid_score=invalid_score)
            return {
                "x": x,
                "y": y,
                "conditional":conditional,
                "ppscore": task["ppscore"],
                "unnormalized_ppscore":task["unnormalized_ppscore"],
                "case": case_type,
                "is_valid_score": task["is_valid_score"],
                "metric": task["metric_name"],
                "baseline_score": task["baseline_score"],
                "model_score": task["model_score"],  # sklearn returns negative mae
                "model": model,
            }
        else:
            raise exception


def _get_task(case_type, invalid_score):
    if case_type in VALID_CALCULATIONS.keys():
        return VALID_CALCULATIONS[case_type]
    elif case_type in INVALID_CALCULATIONS:
        return {
            "type": case_type,
            "is_valid_score": False,
            "model_score": invalid_score,
            "baseline_score": invalid_score,
            "ppscore": invalid_score,
            "unnormalized_ppscore": invalid_score,
            "metric_name": None,
            "metric_key": None,
            "model": None,
            "score_normalizer": None,
        }
    raise Exception(f"case_type {case_type} is not supported")


def _format_list_of_dicts(scores, output, sorted):
    """
    Format list of score dicts `scores`
    - maybe sort by ppscore
    - maybe return pandas.Dataframe
    - output can be one of ["df", "list"]
    """
    if sorted:
        scores.sort(key=lambda item: item["ppscore"], reverse=True)

    if output == "df":
        df_columns = [
            "x",
            "y",
            "conditional",
            "ppscore",            
            "unnormalized_ppscore",
            "case",
            "is_valid_score",
            "metric",
            "baseline_score",
            "model_score",
            "model",
        ]
        data = {column: [score[column] for score in scores] for column in df_columns}
        scores = pd.DataFrame.from_dict(data)

    return scores


def predictors(df,
               y,
               sample_weight = None,
               conditional = None,
               n_bins_target = 10,
               n_bins_independent = None,
               average = "weighted",                   
               model=tree.DecisionTreeClassifier(),
               sample=5_000,
               dropna=True,
               cross_validation=4,
               random_seed=123,
               invalid_score=0,
               catch_errors=True,
               verbose = False,
               cv_n_jobs = None,
               **kwargs):
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
        n_bins to be passed to KBinsDiscretizer for the target variable, to transform a regression problem into a classification one.     
    n_bins_independent: int or None
        n_bins to be passed to KBinsDiscretizer for the dependent variable. it usefull when using a model like a DecisionTree without regularization.
        Binning avoids models with too much variance and thus potential overfit. Also helps in computation time when calculating for many features.
    average: str
        `average` arg passed to sklearn roc_auc_score on multiclass case (when binarizing the regression problem, it yields a multiclas classification problem)    
    sample_weight:
        sample_weight column name. Passed to model.fit and roc_auc_score.    
    sample : int or `None`
        Number of rows for sampling. The sampling decreases the calculation time of the PPS.
        If `None` there will be no sampling.
    dropna: bool
        whether to keep only rows where all features are not null        
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
    if not _is_column_in_df(y, df):
        raise ValueError(
            f"The 'y' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe"
        )
    if len(df[[y]].columns) >= 2:
        raise AssertionError(
            f"The dataframe has {len(df[[y]].columns)} columns with the same column name {y}\nPlease adjust the dataframe and make sure that only 1 column has the name {y}"
        )
    
    output = "df"
    if not output in ["df", "list"]:
        raise ValueError(
            f"""The 'output' argument should be one of ["df", "list"] but you passed: {output}\nPlease adjust your input to one of the valid values"""
        )
    
    sorted = True
    if not sorted in [True, False]:
        raise ValueError(
            f"""The 'sorted' argument should be one of [True, False] but you passed: {sorted}\nPlease adjust your input to one of the valid values"""
        )

    
    if not dropna:
        #sample first to avoid overhead of sampling sub dfs
        df = _maybe_sample(df, sample, random_seed=random_seed)
        
    scores=[]
    n_cols_remove = len([i for i in [conditional,sample_weight] if not i is None])
    
    with tqdm(total=df.shape[1] - n_cols_remove - 1, position=0, leave=True, disable = not verbose) as pbar:       
        for x in df.columns:
            if (x != sample_weight) and (x != conditional) and (x != y):
                s = score(
                        df=df,
                        x=x,
                        y=y,
                        conditional=conditional,
                        model=model,
                        n_bins_target=n_bins_target,
                        n_bins_independent=n_bins_independent,
                        average=average,
                        sample=sample,
                        sample_weight = sample_weight,
                        dropna=dropna,
                        catch_errors = catch_errors,
                        cv_n_jobs = cv_n_jobs,
                        **kwargs
                        ) 
                scores.append(s)
                pbar.update()
    
    return _format_list_of_dicts(scores=scores, output=output, sorted=sorted)



def matrix(df,
           sample_weight = None,
           conditional = None,
           n_bins_target = 10,
           n_bins_independent = None,
           average = "weighted",                   
           model=tree.DecisionTreeClassifier(),
           sample=5_000,
           dropna=True,
           cross_validation=4,
           random_seed=123,
           invalid_score=0,
           catch_errors=True,
           verbose = False,
           cv_n_jobs=None,
           **kwargs):
    
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
        n_bins to be passed to KBinsDiscretizer for the target variable, to transform a regression problem into a classification one.     
    n_bins_independent: int or None
        n_bins to be passed to KBinsDiscretizer for the dependent variable. it usefull when using a model like a DecisionTree without regularization.
        Binning avoids models with too much variance and thus potential overfit. Also helps in computation time when calculating for many features.
    average: str
        `average` arg passed to sklearn roc_auc_score on multiclass case (when binarizing the regression problem, it yields a multiclas classification problem)    
    sample_weight:
        sample_weight column name. Passed to model.fit and roc_auc_score.    
    sample : int or `None`
        Number of rows for sampling. The sampling decreases the calculation time of the PPS.
        If `None` there will be no sampling.
    dropna: bool
        whether to keep only rows where all features are not null        
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
    
    output = "df"
    if not output in ["df", "list"]:
        raise ValueError(
            f"""The 'output' argument should be one of ["df", "list"] but you passed: {output}\nPlease adjust your input to one of the valid values"""
        )
        
    sorted = True
    if not sorted in [True, False]:
        raise ValueError(
            f"""The 'sorted' argument should be one of [True, False] but you passed: {sorted}\nPlease adjust your input to one of the valid values"""
        )
    
    if not dropna:
        #sample first to avoid overhead of sampling sub dfs
        df = _maybe_sample(df, sample, random_seed=random_seed)
    
    scores = []
    n_cols_remove = len([i for i in [conditional,sample_weight] if not i is None])
    with tqdm(total=(df.shape[1]-n_cols_remove)**2, position=0, leave=True, disable = not verbose) as pbar:       
        for x in df.columns:
            if (x != sample_weight) and (x != conditional):                    
                for y in df.columns:
                    if (y != sample_weight) and (y != conditional):
                        s = score(
                            df=df,
                            x=x,
                            y=y,
                            conditional=conditional,
                            model=model,
                            n_bins_target=n_bins_target,
                            n_bins_independent=n_bins_independent,
                            average=average,
                            sample=sample,
                            sample_weight = sample_weight,
                            dropna=dropna,
                            catch_errors = catch_errors,
                            cv_n_jobs = cv_n_jobs,
                            **kwargs) 
                        scores.append(s)
                        pbar.update()

    return _format_list_of_dicts(scores=scores, output=output, sorted=sorted)
