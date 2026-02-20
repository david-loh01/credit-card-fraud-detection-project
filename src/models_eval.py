# src/models_eval.py
from typing import Tuple

import ap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
classification_report,
confusion_matrix,
precision_recall_curve,
average_precision_score
)

from src.plot_utils import plot_path


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def _plot_confusion_matrix(y_true, y_pred, title: str, subdir: str, filename: str):
    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(y_true, y_pred, normalize="true"),
        annot=True,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True Value")
    ax.set_xlabel("Predicted Value")

    out_path = plot_path(subdir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plot_precision_recall_curve(y_true,
                                 y_scores,
                                 title: str, subdir: str,
                                 filename: str
                                 ):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots()
    ax.step(recall,
            precision,
            color = "b",
            alpha = 0.2,
            where = "post",
            label = f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc = "upper right")
    ax.grid(True)

    out_path = plot_path(subdir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"{title} - Average Precision Score (AUPRC): {average_precision:.4f}")


def eval_logreg(X: np.ndarray, y: np.ndarray, title_suffix: str):
    train_x, val_x, train_y, val_y = _train_val_split(X, y)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(train_x, train_y)
    pred_y = clf.predict(val_x)
    y_scores = clf.predict_proba(val_x)[:,1] #probability of fraud class

    print(f"=== Logistic Regression ({title_suffix}) ===")
    print(classification_report(val_y, pred_y))

    _plot_confusion_matrix(
        val_y,
        pred_y,
        f"Confusion Matrix (LogReg, {title_suffix})",
        subdir="confusion_logreg",
        filename=f"logreg_{title_suffix.replace(':', '_')}.png",
    )

    _plot_precision_recall_curve(
        val_y,
        y_scores,
        f"Precision Recall Curve (LogReg, {title_suffix})",
        subdir="pr_logreg",
        filename=f"pr_logreg_{title_suffix.replace(':', '_')}.png",
    )


def eval_decision_tree(X: np.ndarray, y: np.ndarray, title_suffix: str):
    train_x, val_x, train_y, val_y = _train_val_split(X, y)
    model = DecisionTreeClassifier(max_depth=6, criterion="entropy", random_state=24)
    model.fit(train_x, train_y)
    y_pred = model.predict(val_x)
    y_scores = model.predict_proba(val_x)[:,1]

    print(f"=== Decision Tree ({title_suffix}) ===")
    print(classification_report(val_y, y_pred))
    _plot_confusion_matrix(
        val_y,
        y_pred,
        f"Confusion Matrix (DT, {title_suffix})",
        subdir="confusion_dt",
        filename=f"dt_{title_suffix.replace(':', '_')}.png",
    )
    _plot_precision_recall_curve(
        val_y,
        y_scores,
        f"PR Curve (DT, {title_suffix})",
        subdir="pr_dt",
        filename=f"pr_dt_{title_suffix.replace(':', '_')}.png",
    )

def eval_xgboost(X: np.ndarray, y: np.ndarray, title_suffix: str):
    train_x, val_x, train_y, val_y = _train_val_split(X, y)
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=24,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    y_pred = model.predict(val_x)
    y_scores = model.predict_proba(val_x)[:,1]

    print(f"=== XGBoost ({title_suffix}) ===")
    print(classification_report(val_y, y_pred))
    _plot_confusion_matrix(
        val_y,
        y_pred,
        f"Confusion Matrix (XGB, {title_suffix})",
        subdir="confusion_xgb",
        filename=f"xgb_{title_suffix.replace(':', '_')}.png",
    )
    _plot_precision_recall_curve(
        val_y,
        y_scores,
        f"PR Curve (XGB, {title_suffix})",
        subdir="pr_xgb",
        filename=f"pr_xgb_{title_suffix.replace(':', '_')}.png",
    )

def eval_random_forest(X: np.ndarray, y: np.ndarray, title_suffix: str):
    train_x, val_x, train_y, val_y = _train_val_split(X, y)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=24,
        n_jobs=-1,
    )
    rf.fit(train_x, train_y)
    y_pred = rf.predict(val_x)
    y_scores = rf.predict_proba(val_x)[:,1]

    print(f"=== Random Forest ({title_suffix}) ===")
    print(classification_report(val_y, y_pred))
    _plot_confusion_matrix(
        val_y,
        y_pred,
        f"Confusion Matrix (RF, {title_suffix})",
        subdir="confusion_rf",
        filename=f"rf_{title_suffix.replace(':', '_')}.png",
    )
    _plot_precision_recall_curve(
        val_y,
        y_scores,
        f"PR Curve (RF, {title_suffix})",
        subdir="pr_rf",
        filename=f"pr_rf_{title_suffix.replace(':', '_')}.png",
    )