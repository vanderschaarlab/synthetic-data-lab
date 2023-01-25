# Imports
# Standard
import itertools

# 3rd Party
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb


def FTU_score(loader):

    # Input: data to be evaluated (synthetic or real)

    # Split the data into D_tr and D_eval for training and validation
    X, y = loader[loader.static_features].copy(), loader[loader.target_column].copy()
    X.drop(columns=["is_dead_at_time_horizon=14"], inplace=True)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, train_size=0.6, random_state=4
    )
    X_train.reset_index(drop=True, inplace=True)
    X_eval.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_eval.reset_index(drop=True, inplace=True)

    # Train a classifier on D_tr to learn prediction rule (X, A) -> Y
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=1,
        gamma=1,
        objective="binary:logistic",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    # for each pair of feature values
    ftu_list = []
    sensitive_feature_values = loader[loader.sensitive_features[0]].unique()
    for val1, val2 in itertools.combinations(sensitive_feature_values, 2):
        # Create a new dataset D_eval(1) by setting A := 1 for all records in D_eval, the X is kept the same.
        # Create D_eval(0) similarly.
        X_eval_1 = X_eval.copy()
        X_eval_2 = X_eval.copy()

        X_eval_1[loader.sensitive_features[0]] = val1
        X_eval_2[loader.sensitive_features[0]] = val2

        # Pass D_eval(1) to the classifier to get predictions \hat{Y}(1).
        # Similarly, make predictions on D_eval(0) to get \hat{Y}(0).
        y_hat_1 = xgb_model.predict(X_eval_1)
        y_hat_2 = xgb_model.predict(X_eval_2)

        # Calculate the proportion of \hat{Y}(1) = 1, denoted as P1.
        # Calculate the proportion of \hat{Y}(0) = 1, denoted as P0.
        p1 = y_hat_1.sum() / len(y_hat_1)
        p2 = y_hat_2.sum() / len(y_hat_2)

        # FTU = abs(P0 - P1)
        ftu = abs(p1 - p2)
        ftu_list.append(ftu)
    return np.median(ftu_list)


def demographic_parity_score(loader):

    # Input: data to be evaluated (synthetic or real)

    # Split the data into D_tr and D_eval for training and validation
    X, y = loader[loader.static_features].copy(), loader[loader.target_column].copy()
    X.drop(columns=["is_dead_at_time_horizon=14"], inplace=True)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, train_size=0.6, random_state=4
    )
    X_train.reset_index(drop=True, inplace=True)
    X_eval.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_eval.reset_index(drop=True, inplace=True)

    # Train a classifier on D_tr to learn prediction rule (X, A) -> Y
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=1,
        gamma=1,
        objective="binary:logistic",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    # for each pair of feature values
    demographic_parity_list = []
    sensitive_feature_values = loader[loader.sensitive_features[0]].unique()
    for val1, val2 in itertools.combinations(sensitive_feature_values, 2):

        # Create a new dataset D_eval(1) by only keeping the records in D_eval that have A = 1. Remove the records with A = 0.
        # Create D_eval(0) similarly.
        X_eval_0 = X_eval.loc[X_eval[loader.sensitive_features[0]] == val1].copy()
        X_eval_1 = X_eval.loc[X_eval[loader.sensitive_features[0]] == val2].copy()
        # display(X_eval_0)
        # display(X_eval_1)

        # Pass D_eval(1) to the classifier to get predictions \hat{Y}(1).
        # Similarly, make predictions on D_eval(0) to get \hat{Y}(0).
        y_hat_0 = xgb_model.predict(X_eval_0)
        y_hat_1 = xgb_model.predict(X_eval_1)

        # Calculate the proportion of \hat{Y}(1) = 1, denoted as P1.
        # Calculate the proportion of \hat{Y}(0) = 1, denoted as P0.
        p0 = y_hat_0.sum() / len(y_hat_0)
        p1 = y_hat_1.sum() / len(y_hat_1)

        # DP = abs(P0 - P1)
        DP = abs(p0 - p1)
        demographic_parity_list.append(DP)
    return np.mean(demographic_parity_list)
