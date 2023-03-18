from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             matthews_corrcoef, log_loss, confusion_matrix, classification_report, balanced_accuracy_score)
from sklearn.naive_bayes import ComplementNB
import time as time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

'''Apply sampling technique for imbalanced - SMOTE'''

from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def make_model(df, model, param_grid, test_size = 0.2, folds=5, scoring = 'roc_auc', over_size = 0.2, under_size = 0.5):        # Non-Linear models
    '''Function to fit a model and return the best parameters and accuracy score'''

    y = df['y']
    le = LabelEncoder()
    y = le.fit_transform(y)
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

    
    # Create Resampling pipeline
    over = SMOTE(sampling_strategy=over_size)
    if under_size!=None: 
        under = RandomUnderSampler(sampling_strategy=under_size)
        clf = Pipeline([
        ('preprocessor', preprocessor),
        ('over', over), 
        ('under', under),
        ('classifier', model)
    ])
    else:
        clf = Pipeline([
        ('preprocessor', preprocessor),
        ('over', over), 
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
    linear = hasattr(model, 'decision_function')        # check if the model has a decision function (linear models)
    if linear:
        y_score = clf_grid.decision_function(X_test)  # use decision function to get the probability scores for linear models
    else:
        y_score = clf_grid.predict_proba(X_test)[:, 1]  # use predict_proba to get the probability scores for non-linear models

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
    
    # Calculate the classification metrics
    y_pred = clf_grid.predict(X_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),     
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),   
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc,  # Use the roc_auc variable from the ROC curve plot
        'MCC': matthews_corrcoef(y_test, y_pred),
        'Log-Loss': log_loss(y_test, y_pred)
    }
    
    metrics_df = pd.DataFrame(metrics, index=[0])
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf_grid, train_time, test_score, metrics_df