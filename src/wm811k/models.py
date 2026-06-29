from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def create_baseline_models(random_state: int = 42):
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

