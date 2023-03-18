import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# make parameter grids
param_lr = {
    'classifier__penalty' : ['l2', 'none'], 
    'classifier__C'       : np.logspace(2,-9,5),
    'classifier__solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}
param_svc = {
    'classifier__C'       : [1, 10, 100],
    'classifier__kernel'  : ['linear', 'rbf', 'sigmoid'],
    'classifier__gamma'   : ['auto'],
}
param_rf = {
    'classifier__max_depth'    : [10, 20, 30],
    'classifier__min_samples_split' : [2, 5, 10],
    'classifier__min_samples_leaf'  : [1, 2],
#     'classifier__max_depth'    : [10],
#     'classifier__min_samples_split' : [5],
#     'classifier__min_samples_leaf'  : [1],
}
param_gbc = {
    'classifier__loss' : ['deviance', 'exponential'],
    'classifier__max_depth' : [10, 20, 30],
    'classifier__min_samples_split' : [2, 5, 10],
    'classifier__min_samples_leaf' : [1, 2, 5],
}
param_xgb = {
    # 'classifier__n_jobs' : -1,
    # 'classifier__max_depth' : [6, 12, 15],
    'classifier__max_depth' : [6],
    'classifier__gamma' : [ 0.1, 0.2, 0.3],
    #  'classifier__gamma' : [0.2],
    # 'classifier__eval_metric' : ['auc', 'logloss', 'merror', 'mlogloss', 'rmse'],
    'classifier__eval_metric' : ['auc'],
    'classifier__reg_alpha' : np.append(np.logspace(0,-6,3),0),
    'classifier__reg_lambda' : np.append(np.logspace(0,-6,3),0),
    'classifier__booster' : ['gbtree', 'gblinear', 'dart'],
    # 'classifier__booster' : ['gbtree'],
}
param_bag = {
    'classifier__n_estimators' : [300],
    'classifier__base_estimator' : [DecisionTreeClassifier(max_depth=1),
                                    DecisionTreeClassifier(max_depth=2),
                                    DecisionTreeClassifier(max_depth=3),
                                    LogisticRegression(penalty='l2', C=1, solver='liblinear'),
                                    ],
    # 'classifier__base_estimator' : [DecisionTreeClassifier(max_depth=2)],

}
param_dt = {
    'classifier__max_depth' : [5, 10, 15, 20, 25],
    'classifier__min_samples_split' : [2, 5, 10, 15, 100],
    'classifier__min_samples_leaf'  : [1, 2, 5, 10, 15],
}
param_sgd = {
    'classifier__loss' : ['hinge', 'log', 'squared_hinge'],
    'classifier__penalty' : ['l2', 'l1', 'elasticnet', 'none'],
    'classifier__alpha' : [0.0001, 0.001, 0.01],
    # 'classifier__learning_rate' : ['constant', 'optimal', 'invscaling', 'adaptive'],
}
param_knn = {
    'classifier__n_neighbors' : [3, 5, 7],
    'classifier__weights' : ['uniform', 'distance'],
    'classifier__algorithm' : ['auto'],
    'classifier__leaf_size' : [10, 20, 30],
    'classifier__p' : [2],
}
param_nb = {'classifier__var_smoothing': np.logspace(0,-9, 10)
}
param_cnb = {
    'classifier__alpha' : np.append(np.logspace(0,-9,10),0),
}
param_ada = {
    'classifier__n_estimators' : [300],
    'classifier__base_estimator' : [DecisionTreeClassifier(max_depth=1),
                                    DecisionTreeClassifier(max_depth=2),
                                    DecisionTreeClassifier(max_depth=3),
                                    LogisticRegression(penalty='l2', C=1, solver='liblinear'),
                                    ],
    # 'classifier__base_estimator' : [DecisionTreeClassifier(max_depth=2)],
}  
param_lgbm = {
    'classifier__boosting_type': ['gbdt', 'dart', 'goss'],
    # 'classifier__num_leaves': [31, 50, 100],
    # 'classifier__learning_rate': [0.01, 0.05, 0.1],
    # 'classifier__n_estimators': [100, 200, 500],
    # 'classifier__subsample': [0.8, 1],
    # 'classifier__colsample_bytree': [0.8, 1],
    'classifier__reg_alpha' : np.append(np.logspace(0,-6,3),0),
    'classifier__reg_lambda' : np.append(np.logspace(0,-6,3),0)
}
param_cat = {
    # 'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__depth': [4, 6, 8],
    'classifier__l2_leaf_reg': [1, 3, 5],
    # 'classifier__iterations': [100, 200, 500],
    # 'classifier__border_count': [32, 128, 254],
    # 'classifier__bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS', 'No'],
}  
param_extra = {
    'classifier__n_estimators': [300],
    'classifier__max_depth': [None, 6, 12],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__class_weight': [None, 'balanced', 'balanced_subsample'],
}
param_rgf = {
    'classifier__max_leaf': [1000, 2000, 3000],
    'classifier__algorithm': ["RGF", "RGF_Opt", "RGF_Sib"],
}
