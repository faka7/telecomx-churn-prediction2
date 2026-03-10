import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

class ModelTrainer:
    def __init__(self):
        self.models = self.get_base_models()
        self.trained_models = {}

    def get_base_models(self):
        return {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(),
            'SVM': SVC(probability=True),
            'Gradient Boosting': GradientBoostingClassifier()
        }

    def train_model(self, model_name, X, y):
        model = self.models[model_name]
        model.fit(X, y)
        self.trained_models[model_name] = model

    def make_prediction(self, model_name, X):
        model = self.trained_models[model_name]
        return model.predict(X)

    def cross_validation(self, model_name, X, y, cv=5):
        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv)
        return scores.mean()

    def hyperparameter_optimization(self, model_name, X, y, param_grid, cv=5):
        model = self.models[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=cv)
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def save_model(self, model_name):
        model = self.trained_models[model_name]
        with open(f'{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, model_name):
        with open(f'{model_name}.pkl', 'rb') as f:
            self.trained_models[model_name] = pickle.load(f)

    def get_feature_importance(self, model_name, feature_names):
        model = self.trained_models[model_name]
        importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        return dict(zip(feature_names, importances)) if importances is not None else None
