import os

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nAccuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nPrecision:")
    print(precision_score(y_test, predictions))

    print("\nRecall:")
    print(recall_score(y_test, predictions))

    print("\nF1 Score:")
    print(f1_score(y_test, predictions))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    os.makedirs("../outputs", exist_ok=True)

    # Confusion Matrix Heatmap

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine","Fraud"],
        yticklabels=["Genuine","Fraud"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig(
        "../outputs/confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.show()

    # ROC Curve

    fpr, tpr, _ = roc_curve(y_test, probabilities)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot([0,1],[0,1],'r--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.savefig(
        "../outputs/roc_curve.png",
        bbox_inches="tight"
    )

    plt.show()

    # Feature Importance

    importances = model.feature_importances_

    feature_names = X_test.columns

    indices = importances.argsort()[::-1]

    plt.figure(figsize=(10,6))

    plt.bar(
        range(len(importances)),
        importances[indices]
    )

    plt.xticks(
        range(len(importances)),
        feature_names[indices],
        rotation=90
    )

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(
        "../outputs/feature_importance.png"
    )

    plt.show()

    # Save trained model

    joblib.dump(
        model,
        "../outputs/random_forest_model.pkl"
    )

    print("\nModel saved successfully!")