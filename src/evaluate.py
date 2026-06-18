from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.metrics import roc_auc_score


def evaluate_model(y_true, y_pred,y_prob):
    y_true_binary = (
    y_true == "Yes"
    ).astype(int)
    roc_auc = roc_auc_score(
    y_true_binary,
    y_prob
)

    print(
        f"ROC-AUC : {roc_auc:.4f}"
)
    print(
        f"Accuracy : {accuracy_score(y_true, y_pred):.4f}"
    )

    print(
        f"Precision: {precision_score(y_true, y_pred, pos_label='Yes'):.4f}"
    )

    print(
        f"Recall   : {recall_score(y_true, y_pred, pos_label='Yes'):.4f}"
    )

    print(
        f"F1 Score : {f1_score(y_true, y_pred, pos_label='Yes'):.4f}"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred
        )
    )
    cm = confusion_matrix(
    y_true,
    y_pred
)

    print("\nConfusion Matrix:\n")
    print(cm)