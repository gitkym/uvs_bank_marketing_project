from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
import time as time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc


def make_model_nl(df, model, param_grid, test_size = 0.2, folds=5, scoring = 'roc_auc'):        # Non-Linear models
    '''Function to fit a model and return the best parameters and accuracy score'''
    y = df['y']
    X=df.drop('y', axis=1)
    # Create a pipeline for categorical features
    cat_features = X.select_dtypes(include=['object']).columns
    cat_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # df_cat= cat_pipeline.fit_transform(df[cat_features])

    # Create a pipeline for numerical features
    num_features = X.select_dtypes(include=['int64', 'float64']).columns

    if isinstance(model, ComplementNB):     # ComplimentNB() does not support negative values
        num_pipeline = Pipeline([
            ('minmax_scaler', MinMaxScaler())
        ])
    else:
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
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit the model
    start_time = time.time()
    clf_grid.fit(X_train,y_train)
    end_time = time.time()
    train_time = round((end_time - start_time)/60,3)
    # Print the tuned parameters and score
    print("Tuned Hyperparameters :", clf_grid.best_params_)
    print("Accuracy :",clf_grid.best_score_)
    # print training time in minutes
    print('Training Time : {} minutes'.format(round(train_time)))
    print("Test Score :",clf_grid.score(X_test,y_test))
    
    # Get the results
    test_score = clf_grid.score(X_test,y_test)

    '''plot roc curve'''
    # Convert string labels to binary values
    le = LabelEncoder()
    y_test = le.fit_transform(y_test)
    y_prob = clf_grid.predict_proba(X_test)[:, 1]   # use predict_proba to get the probability scores for non-linear models
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return clf_grid, train_time, test_score

############################################################################################################

def make_model_l(df, model, param_grid, test_size = 0.2, folds=5, scoring = 'roc_auc'):       # Linear models
    '''Function to fit a model and return the best parameters and accuracy score'''
    y = df['y']
    X=df.drop('y', axis=1)
    # Create a pipeline for categorical features
    cat_features = X.select_dtypes(include=['object']).columns
    cat_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # df_cat= cat_pipeline.fit_transform(df[cat_features])

    # Create a pipeline for numerical features
    num_features = X.select_dtypes(include=['int64', 'float64']).columns
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
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit the model
    start_time = time.time()
    clf_grid.fit(X_train,y_train)
    end_time = time.time()
    train_time = round((end_time - start_time)/60,3)
    # Print the tuned parameters and score
    print("Tuned Hyperparameters :", clf_grid.best_params_)
    print("Accuracy :",clf_grid.best_score_)
    # print training time in minutes
    print('Training Time : {} minutes'.format(round(train_time)))
    print("Test Score :",clf_grid.score(X_test,y_test))
    
    # Get the results
    test_score = clf_grid.score(X_test,y_test)

    '''plot roc curve'''
    # Convert string labels to binary values
    le = LabelEncoder()
    y_test = le.fit_transform(y_test)
    y_score = clf_grid.decision_function(X_test)    # use decision function to get the probability scores for linear models
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return clf_grid, train_time, test_score