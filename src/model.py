from typing import Optional, Callable
import joblib
import pickle
import lightgbm as lgb
import xgboost as xgb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.model_selection import learning_curve


class LightGBMModel:
    def __init__(self, model_params: dict = {}, custom_loss: Optional[Callable] = None, custom_eval: Optional[Callable] = None, client = None):
        """
        Initialize the LightGBM model using low-level API.
        :param task: str, one of ["classification", "regression"]
        :param model_params: dict, parameters for the model
        :param custom_loss: callable, custom loss function
        """
        self.model_params = model_params
        self.custom_loss = custom_loss
        self.custom_eval = custom_eval
        self.model = None
        self.evals_result = {}
        self.client = client

        # カスタム損失関数がある場合、objective に設定
        if custom_loss:
            self.model_params["objective"] = self.custom_loss

    def train(self, X_train, y_train, eval_set, early_stopping_rounds=None):
        """
        Train the model using LightGBM lowe-level API.
        :param X_train: array-like, feature matrix for training
        :param y_train: array-like, target vector for training
        :param X_val: array-like, feature matrix for validation 
        :param y_val: array-like, target vector for validation
        :param early_stopping_rounds: int, stops training early if no improvement
        """
        if isinstance(X_train, pd.DataFrame):
            train_data = lgb.Dataset(X_train, label=y_train)
            eval_data = lgb.Dataset(eval_set[0], label=eval_set[1], reference=train_data) if eval_set else None

            self.model = lgb.train(
                params=self.model_params,
                train_set=train_data,
                valid_sets=[eval_data] if eval_data else None,
                feval=self.custom_eval if self.custom_eval else None,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
                    lgb.log_evaluation(False),
                    lgb.record_evaluation(self.evals_result)
                ],
            )


    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        if isinstance(self.model, (lgb.DaskLGBMRegressor, lgb.DaskLGBMClassifier)):
            booster = self.model.to_local().booster_
            booster.save_model(filepath)
        else:
            joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        if self.client:
            local_model = lgb.Booster(model_file=filepath)
            if "binary" in self.model_params.get("objective", ""):
                self.model = lgb.DaskLGBMClassifier(client=self.client, **self.model_params)
            else:
                self.model = lgb.DaskLGBMRegressor(client=self.client, **self.model_params)
            self.model.set_params(**local_model.params)
        else:
            self.model = joblib.load(filepath)

    def visualize_feature_importance(self):
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        lgb.plot_importance(self.model)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()

    def visualize_learning_curve(self):
        if not self.evals_result:
            raise ValueError("No evaluation results available.")
        lgb.plot_metric(self.evals_result)
        plt.title("Learning Curve")
        plt.tight_layout()
        plt.show()



# class MLModel:
#     def __init__(self, model_type="sklearn", task="classification", model_params=None, custom_loss=None) -> None:
#         """
#         Initialize the ML model.
#         :param model_type: str, one of ["sklearn", "lightgbm", "xgboost"]
#         :param task: str, one of ["classification", "regression"]
#         :param model_params: dict, parameters for the model
#         :param custom_loss: callable, custom loss function (LightGBM/XGBoost only)
#         """
#         self.model_type = model_type.lower()
#         self.task = task.lower()
#         self.model_params = model_params or {}
#         self.model = self._initialize_model()
#         self.eval_results = None
# 
#     def _initialize_model(self):
#         if self.model_type == "sklearn" and self.task == "classification":
#             return RandomForestClassifier(**self.model_params)
#         elif self.model_type == "sklearn" and self.task == "regression":
#             return RandomForestRegressor(**self.model_params) 
#         elif self.model_type == "lightgbm" and self.task == "classification":
#             return lgb.LGBMClassifier(**self.model_params)
#         elif self.model_type == "lightgbm" and self.task == "regression":
#             return lgb.LGBMRegressor(**self.model_params)
#         elif self.model_type == "xgboost" and self.task == "classification":
#             return xgb.XGBClassifier(**self.model_params)
#         elif self.model_type == "xgboost" and self.task == "regression":
#             return xgb.XGBRegressor(**self.model_params)
#         else:
#             raise ValueError("Unsupported combination of task and model type.")
# 
#     def train(self, X_train, y_train, eval_set=None, early_stopping_rounds=None):
#         """
#         Train the model.
#         :param X_train: array-like, feature matrix for training
#         :param y_train: array-like, target vector for training
#         :param eval_set: tuple, validation data (X_val, y_val)
#         :param early_stopping_rounds: int, stops training early if no improvement
#         """
#         if self.model_type in ["lightgbm", "xgboost"] and eval_set is not None:
#             eval_set = [(eval_set[0], eval_set[1])]
#             self.eval_results = {}
# 
#         if self.model_type == "lightgbm":
#             params = self.model.get_params()
#             if self.custom_loss:
#                 params["objective"] = self.custom_loss
# 
#             self.model.fit(
#                 X_train, 
#                 y_train, 
#                 eval_set=eval_set, 
#                 callbacks=[
#                     lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
#                     lgb.log_evaluation(False)
#                 ],
#                 eval_metric=self.custom_loss if self.custom_loss else "rmse",
#             )
#             self.eval_results = self.model.evals_result_
#         elif self.model_type == "xgboost":
#             self.model.fit(
#                 X_train, 
#                 y_train, 
#                 eval_set=eval_set, 
#                 early_stopping_rounds=early_stopping_rounds, 
#                 verbose=False,
#                 eval_metric=self.custom_loss if self.custom_loss else "rmse" if self.task == "regression" else "logloss",
#             )
#             self.eval_results = self.model.evals_result()
#         else: # sklearn
#             self.model.fit(X_train, y_train)
# 
#     def predict(self, X):
#         """
#         Make predictions.
#         :param X: array-like, feature matrix
#         :return: array-like, predictions
# 
#         """
#         return self.model.predict(X)
#     
#     def evaluate(self, X, y):
#         """
#         Evaluate the model's performance.
#         :param X: array-like, feature matrix for evaluation
#         :param y: array-like, true target values
#         :return: float, evaluation score
#         """
#         predictions = self.predict(X)
#         if self.task == "classification":
#             return accuracy_score(y, predictions)
#         elif self.task == "regression":
#             return mean_squared_error(y, predictions, squared=False)
#     
#     def save_model(self, filepath):
#         """
#         Save the model to a file.
#         :param filepath: str, path to save the model
#         """
#         with open(filepath, "wb") as file:
#             pickle.dump(self.model, file)
#     
#     def load_model(self, filepath):
#         """
#         Load the model from a file.
#         :param filepath: str, path to the model file
#         """
#         with open(filepath, "rb") as file:
#             self.model = pickle.load(file)
#     
#     def visualize_feature_importance(self, feature_names=None, max_features=20):
#         """
#         Visualize feature importance.
#         :param feature_names: list, names of the features
#         :param max_features: int, number of top features to display
#         """
#         if hasattr(self.model, "feature_importances_"):
#             importances = self.model.feature_importances_
#         elif self.model_type == "xgboost":
#             importances = self.model.get_booster().get_score(importance_type="weight")
#             importances = list(importances.values())
#         else:
#             raise AttributeError("Feature importance is not available for the current model.")
# 
#         sorted_idx = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)[:max_features]
#         sorted_importances = [importances[i] for i in sorted_idx]
#         sorted_features = [feature_names[i] if feature_names else f"Feature {i}" for i in sorted_idx]
# 
#         plt.figure(figsize=(10, 6))
#         plt.barh(sorted_features, sorted_importances, color="skyblue")
#         plt.xlabel("Importance")
#         plt.title("Feature Importance")
#         plt.gca().invert_yaxis()
#         plt.tight_layout()
#         plt.show()
# 
#     def visualize_learning_curve(self, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
#         """
#         Visualize the learning curve.
#         :param X: arrray-like, feature matrix
#         :param y: array-like, target vector
#         :param cv: int, number of cross-validation folds
#         :param train_sizes: array-like, relative or absolute numbers of training examples
#         """
#         if self.model_type in ["lightgbm", "xgboost"] and self.eval_results:
#             plt.figure(figsize=(10, 6))
#             for dataset in self.eval_results:
#                 for metric in self.eval_results[dataset]:
#                     plt.plot(self.eval_results[dataset][metric], label=f"{dataset} {metric}")
#             plt.xlabel("Iteration")
#             plt.ylabel("Metric")
#             plt.title("Learning Curve")
#             plt.legend()
#             plt.show()
#         else:
#             train_sizes, train_scores, val_scores = learning_curve(
#                 self.model, X, y, cv=cv, train_sizes=train_sizes, scoring="accuracy" if self.task == "classification" else "neg_root_mean_squared_error"
#             )
#             train_scores_mean = np.mean(train_scores, axis=1)
#             val_scores_mean = np.mean(val_scores, axis=1)
# 
#             plt.figure(figsize=(10, 6))
#             plt.plot(train_sizes, train_scores_mean, label="Training Score", marker="o")
#             plt.plot(train_sizes, val_scores_mean, label="Validation Score", marker="o")
#             plt.xlabel("Training Examples")
#             plt.ylabel("Score")
#             plt.title("Learning Curve")
#             plt.legend()
#             plt.show()
# 
#     def visualize_confusion_matrix(self, X, y, class_names=[], normalize=False):
#         """
#         Visualize the confusion matrix.
#         :param X: array-like, feature matrix for evaluation
#         :param y: array-like, true target values
#         :param class_names: list, names of the classes
#         :param normalize: bool, whether to normalize the confusion matrix
#         """
#         if self.task != "classification":
#             raise ValueError("Confusion matrix is only applicable for classification tasks.")
# 
#         predictions = self.predict(X)
#         cm = confusion_matrix(y, predictions, normalize="true" if normalize else None)
# 
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
#                     xticklabels=class_names, yticklabels=class_names)
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.title("Confusion Matrix")
#         plt.show()


def custom_loss_function(y_pred, dataset):
    y_true = dataset.get_label()
    
    # ピアソン相関係数の計算
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    
    numerator = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    denominator = np.sqrt(np.sum((y_true - y_true_mean)**2) * np.sum((y_pred - y_pred_mean)**2)) + 1e-10
    corr = numerator / denominator

    # 勾配とヘッセ行列の計算
    grad = -(y_true - y_pred) * (1 - corr)
    hess = np.ones_like(y_true) * 0.1  # 小さな値で安定化

    return grad, hess


def custom_eval_function(y_pred, dataset):
    y_true = dataset.get_label()

    # 配列の reshape を適切に適用
    y_pred = y_pred.reshape(-1)
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    
    numerator = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    denominator = np.sqrt(np.sum((y_true - y_true_mean)**2) * np.sum((y_pred - y_pred_mean)**2)) + 1e-10
    correlation = numerator / denominator

    return 'pearson_corr', correlation, True  # 高いほど良い指標

