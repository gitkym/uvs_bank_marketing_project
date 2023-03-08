import numpy as np

# make parameter grids
param_lr = {
    'classifier__penalty' : ['l2', 'none'], 
    'classifier__C'       : np.logspace(-5,5,5),
    'classifier__solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}
param_svc = {
    'classifier__C'       : [1, 10, 100],
    'classifier__kernel'  : ['linear', 'rbf', 'sigmoid'],
    'classifier__gamma'   : ['auto'],
}
param_rf = {
    'classifier__n_estimators' : [100, 200, 300],
    'classifier__max_depth'    : [10, 20, 30],
    'classifier__min_samples_split' : [2, 5, 10],
    'classifier__min_samples_leaf'  : [1, 2],
}
param_gbc = {
    'classifier__loss' : ['deviance', 'exponential'],
    'classifier__n_estimators' : [100, 200, 300],
    'classifier__max_depth' : [10, 20, 30],
    'classifier__min_samples_split' : [2, 5, 10],
    'classifier__min_samples_leaf' : [1, 2, 5],
}
param_dt = {
    'classifier__max_depth' : [5, 10, 15, 20, 25],
    'classifier__min_samples_split' : [2, 5, 10, 15, 100],
    'classifier__min_samples_leaf'  : [1, 2, 5, 10, 15],
}
param_sgd = {
    'classifier__loss' : ['hinge', 'log', 'squared_hinge'],
    'classifier__penalty' : ['l2', 'l1', 'elasticnet', 'none'],
    'classifier__alpha' : [0.0001, 0.001, 0.01, 0.1, 1],
    'classifier__learning_rate' : ['constant', 'optimal', 'invscaling', 'adaptive'],
}
param_knn = {
    'classifier__n_neighbors' : [3, 5, 7],
    'classifier__weights' : ['uniform', 'distance'],
    'classifier__algorithm' : ['auto'],
    'classifier__leaf_size' : [10, 20, 30, 40],
    'classifier__p' : [2],
}
