from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import time as time
import pandas as pd


def make_model(X_train, y_train, X_test, y_test, model, param_grid, folds=5, scoring = 'roc_auc'):
    '''Function to fit a model and return the best parameters and accuracy score'''
    
    # Create a pipeline for categorical features
    cat_features = X_train.select_dtypes(include=['object']).columns
    cat_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # df_cat= cat_pipeline.fit_transform(df[cat_features])

    # Create a pipeline for numerical features
    num_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])

    # Create a column transformer
    preprocessor = ColumnTransformer([
            ('cat', cat_pipeline, cat_features),
            ('num', num_pipeline, num_features)
])
    # Create a pipeline for the model
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Create a grid search object
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    clf_grid = GridSearchCV(clf,                    # model
                       param_grid = param_grid,   # hyperparameters
                       scoring=scoring,        # metric for scoring
                       cv=cv)                     # number of folds
    
    # Fit the model
    start_time = time.time()
    clf_grid.fit(X_train,y_train)
    print("Tuned Hyperparameters :", clf_grid.best_params_)
    print("Accuracy :",clf_grid.best_score_)
    # print training time in minutes
    print('Training Time : {} minutes'.format(round((time.time() - start_time)/60,3)))
    print("Test Score :",clf_grid.score(X_test,y_test))
    
    # Get the results
    df_param = pd.DataFrame(clf_grid.cv_results_)
    
    return clf_grid, df_param