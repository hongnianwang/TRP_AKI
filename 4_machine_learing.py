#!/usr/bin/env python3

import argparse
import datetime
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main(path_data):

    X = pd.read_csv(os.path.join(path_data, "X_train.csv"))
    y = pd.read_csv(os.path.join(path_data, "y_train.csv"))["label"]

    X_test = pd.read_csv(os.path.join(path_data, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(path_data, "y_test.csv"))["label"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid for hyperparameter tuning
    # Note: The ranges for each parameter are determined based on the specific
    #       task. It's recommended to perform exploratory searches to
    #       narrow down the ranges, and then use grid search for fine-tuning.
    #       For demonstration purposes, we are performing a simple search here.
    param_grid = {
        'learning_rate': [0.03, 0.04, 0.05],
        'max_depth': [7, 9],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.1, 0.5, 1]
    }

    # Initialize the XGBClassifier with a default set of parameters
    best_params = {
        'max_depth': 7,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50
    }

    best_auc = 0

    param_combinations = [
        (lr, md, sub, col, reg_a, reg_l)
        for lr in param_grid['learning_rate']
        for md in param_grid['max_depth']
        for sub in param_grid['subsample']
        for col in param_grid['colsample_bytree']
        for reg_a in param_grid['reg_alpha']
        for reg_l in param_grid['reg_lambda']
    ]
    for lr, md, sub, col, reg_a, reg_l in tqdm(param_combinations, desc="Hyperparameter tuning"):
        xgb_model = xgb.XGBClassifier(
            max_depth=md,
            learning_rate=lr,
            n_estimators=1000,
            min_child_weight=1,
            gamma=0,
            subsample=sub,
            colsample_bytree=col,
            reg_alpha=reg_a,
            reg_lambda=reg_l,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=1,
            seed=27,
            eval_metric='auc',
            early_stopping_rounds=50
        )

        # Fit the model
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

        y_val_pred = xgb_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred)

        # Update best parameters if current model is better
        if auc > best_auc:
            best_auc = auc
            best_params = {
                'max_depth': md,
                'learning_rate': lr,
                'subsample': sub,
                'colsample_bytree': col,
                'reg_alpha': reg_a,
                'reg_lambda': reg_l,
                'eval_metric': 'auc',
                'early_stopping_rounds': 50
            }

    print(f"Best parameters: {best_params}")

    # Re-train the model using the best parameters
    best_xgb_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=1000,
        min_child_weight=1,
        gamma=0,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27
    )

    best_xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

    # Predict probabilities for the test set
    y_pred = best_xgb_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred)
    test_f1 = f1_score(y_test, np.round(y_pred))

    print(f"AUC: {auc}")
    print(f"F1 Score: {test_f1}")

    # Save predictions and true labels with best parameters and AUC in the filename
    output_dir = os.path.join(path_data, 'result')
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    filename = f"{current_time}_auc{auc:.3f}_lr{best_params['learning_rate']}_md{best_params['max_depth']}_sub{best_params['subsample']}_col{best_params['colsample_bytree']}_regA{best_params['reg_alpha']}_regL{best_params['reg_lambda']}.csv"
    predictions_path = os.path.join(output_dir, filename)
    predictions_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train machine learning model.')
    parser.add_argument('--path_data', type=str, required=True, help='Base path to the data directory')
    parser.add_argument('--hour', type=int, choices=[24, 48, 72], help='Time window for data (24, 48, or 72 hours)')
    parser.add_argument('--group', type=int, choices=[1, 2, 3], help='Data group to use: 1 for recent values, 2 for recent + min/max, 3 for recent + min/max + shapelet')

    args = parser.parse_args()

    if args.hour and args.group:
        path_data = os.path.join(args.path_data, str(args.hour), str(args.group))
    else:
        path_data = args.path_data

    main(path_data)
