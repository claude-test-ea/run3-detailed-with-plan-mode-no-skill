"""Training module.

Provides model definitions, hyperparameter grids, and training functions
for the loan eligibility prediction pipeline.
"""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_models():
    """Return a dictionary of model name -> model instance for all candidate models.

    Models included:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting (GradientBoostingClassifier)
    - SVM
    - KNN

    Returns
    -------
    dict
        Mapping of model name (str) to sklearn/xgboost estimator instance.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
            n_estimators=100,
        ),
        "SVM": SVC(random_state=42, probability=True),
        "KNN": KNeighborsClassifier(),
    }
    return models


def get_grid_params():
    """Return hyperparameter grids for models that undergo GridSearchCV.

    Grids are provided for:
    - Random Forest: n_estimators, max_depth, min_samples_split
    - Gradient Boosting: n_estimators, max_depth, learning_rate

    Returns
    -------
    dict
        Mapping of model name (str) to parameter grid (dict).
    """
    grid_params = {
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
        },
    }
    return grid_params


def train_all_models(X_train, y_train):
    """Train all candidate models on the training data.

    For Random Forest and Gradient Boosting, GridSearchCV with 5-fold
    cross-validation and F1 scoring is used to select the best hyperparameters.
    Other models are trained directly with .fit().

    Parameters
    ----------
    X_train : array-like
        Training feature matrix (already preprocessed).
    y_train : array-like
        Training target vector.

    Returns
    -------
    dict
        Mapping of model name (str) to trained model instance.
        For grid-searched models, the best estimator is returned.
    """
    models = get_models()
    grid_params = get_grid_params()
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name in grid_params:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=grid_params[name],
                cv=5,
                scoring="f1",
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)
            trained_models[name] = grid_search.best_estimator_
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Best CV F1:  {grid_search.best_score_:.4f}")
        else:
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"  Training complete.")

    return trained_models
