from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, clone
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier

from preprocessing import RobustKBinsDiscretizer

class MutualInformationForest(RandomForestClassifier):
    
    """A decision tree classifier where the target is discretized. usefull to transform regression into classification.
    One can choose to discretize X as well or not. Discretization is performed using RobustKBinsDiscretizer. 
    This estimator is robust to np.inf, -np.inf and np.nans if handle_nan is set to True. if one chooses to discretize X, it is also robust in the same cases.
    Also, this implementation exposesget_leaf_entropies and get_information_gain as usefull methods for mutual information estimation.
        
    
    """
    
    
    def __init__(        
        self,
        n_estimators=1,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        #quantization params
        n_bins_X=None,        
        n_bins_y=10,        
        strategy_X='kmeans',
        strategy_y='kmeans',
        handle_nan_X = 'handle', #error, handle, ignore        
        handle_nan_y = 'error', #error, handle, ignore        
    ):
        
        
        
        self.n_bins_X = n_bins_X
        self.n_bins_y = n_bins_y
        self.strategy_X = strategy_X
        self.strategy_y = strategy_y
        self.handle_nan_X = handle_nan_X
        self.handle_nan_y = handle_nan_y
        super().__init__(
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
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
        return
    
    def _ensure_2d(self, arr):
        if arr.ndim == 1:
            arr = arr.reshape(-1,1)
        return arr
    
    def _fit_preprocess_data(self, X, y = None):        
        
        X = np.array(X)
        
        X = self._ensure_2d(X)
        max_X = np.finfo(np.float32).max
        min_X = np.finfo(np.float32).min        

        if not self.n_bins_X is None:
            nanimputer = FunctionTransformer()
            infimputer = FunctionTransformer()
            neginfimputer = FunctionTransformer()
            x_quant = RobustKBinsDiscretizer(self.n_bins_X, encode = "ordinal", handle_nan = self.handle_nan_X, strategy=self.strategy_X, return_sparse = False)
        else:
            if self.handle_nan_X == "handle":
                nanimputer = FunctionTransformer(lambda x: np.where(x!=x,min_X, x))                     
                infimputer = FunctionTransformer(lambda x: np.where(x==np.inf,max_X, x))
                neginfimputer = FunctionTransformer(lambda x: np.where(x==-np.inf,min_X+1, x))             
                x_quant = FunctionTransformer()
            else:
                nanimputer = FunctionTransformer()
                infimputer = FunctionTransformer()
                neginfimputer = FunctionTransformer()
                x_quant = FunctionTransformer()
                
                
        
        if not y is None:
            y = np.array(y)
            y = self._ensure_2d(y)
            if not y.dtype in (float,int):
                caster = FunctionTransformer(lambda d: d.astype(str))
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                self.y_preprocessor = make_pipeline(caster,enc).fit(y)
            
            else:

                max_y = np.finfo(np.float32).max
                min_y = np.finfo(np.float32).min        

                y_infimputer = FunctionTransformer(lambda x: np.where(x==np.inf,max_y, x))
                y_neginfimputer = FunctionTransformer(lambda x: np.where(x==-np.inf,min_y+1, x))             

                y_quant = RobustKBinsDiscretizer(self.n_bins_y, encode = "ordinal", handle_nan = self.handle_nan_y, strategy=self.strategy_y, return_sparse = False)
                self.y_preprocessor = make_pipeline(y_infimputer, y_neginfimputer, y_quant).fit(y)
        
        self.X_preprocessor = make_pipeline(infimputer, neginfimputer, nanimputer, x_quant).fit(X)
    
    def _preprocess_data(self, X, y = None):
        X = self._ensure_2d(X)
        X = self.X_preprocessor.transform(X)
        if not y is None:
            y = self._ensure_2d(y)
            y = self.y_preprocessor.transform(y)
        else:
            y = None
        
        return X, y
    
    def fit(self, X, y = None, sample_weight = None):
        self._fit_preprocess_data(X,y)
        X,y = self._preprocess_data(X,y)
        super().fit(X,y, sample_weight)
        return self
    
    def predict(self, X, **kwargs):
        X,y = self._preprocess_data(X,None)
        return super().predict(X, **kwargs)
    
    def apply(self, X, **kwargs):
        X,y = self._preprocess_data(X,None)
        return super().apply(X, **kwargs)
    
    def predict_proba(self, X, **kwargs):
        X,y = self._preprocess_data(X,None)
        return super().predict_proba(X, **kwargs)
    
    def predict_log_proba(self, X, **kwargs):
        X,y = self._preprocess_data(X,None)
        return super().predict_log_proba(X, **kwargs)
    
    def cost_complexity_pruning_path(self, X, y = None, **kawrgs):
        X,y = self._preprocess_data(X,y)
        return super().cost_complexity_pruning_path(X, y, **kwargs)

    def decision_path(self, X):
        X,y = self._preprocess_data(X,None)
        return super().decision_path(X)
    
    def get_leaf_entropies(self, X, y, sample_weight = None):
        
        X,y = self._preprocess_data(X,y)
        leafs = self.apply(X)
        #flatten avoiding colisions
        n_trees = leafs.shape[1]        
        leafs = leafs + (leafs.max(0)+1).cumsum()
        leafs = leafs.T.flatten()
        y=np.hstack(n_trees*[y.flatten()])
        
        if sample_weight is None:
            total_sample_size = len(y)
            a = np.array([leafs,y]).T
            a = a[a[:, 0].argsort()]
            entropies = np.array([(len(i)/total_sample_size)*entropy(i) for i in np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:])])
        else:
            total_sample_size = np.sum(sample_weight)
            a = np.array([leafs,y, sample_weight]).T
            a = a[a[:, 0].argsort()]
            groups = np.split(a[:, 1:], np.unique(a[:, 0], return_index=True)[1][1:])
            entropies = np.array([(len(i)/total_sample_size)*entropy(i[:,0], i[:,1]) for i in groups])
        return entropies
    
    def get_information_gain(self, X, y, sample_weight = None):
        
        y = np.array(y)
        leaf_entropies_avg = np.sum(self.get_leaf_entropies(X, y, sample_weight))                
        
        shuffled_idx = np.random.choice(np.arange(len(y.flatten())), len(y.flatten()), replace = False)
        
        y_shuffled = y[shuffled_idx]
        if not sample_weight is None:
            sample_weight = sample_weight[shuffled_idx]
            
        sample_entropy = np.sum(self.get_leaf_entropies(X, y_shuffled, sample_weight))
        information_gain = (sample_entropy - leaf_entropies_avg)/sample_entropy
        return max(0, information_gain)
    
    def score(self, X, y, sample_weight = None):
        return self.get_information_gain(X, y, sample_weight)
        
def entropy(leaf_targets, sample_weight = None):
    if sample_weight is None:
        _, p = np.unique(leaf_targets, return_counts=True)
        p = p/p.sum()

    else:        
        p = np.zeros((leaf_targets.size, int(leaf_targets.max() + 1)), dtype = float)
        p[np.arange(leaf_targets.size), leaf_targets.astype(int)] = 1
        p = p*(sample_weight.reshape(-1,1))
        p = p.sum(0)
        p = p[p>0]
        p = p/p.sum()        
    return np.sum(-p*np.log(p))        