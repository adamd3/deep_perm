from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score


def calculate_metrics(y_true, y_pred_proba):
    """Calculate various classification metrics"""
    y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)

    metrics = {
        "auroc": roc_auc_score(y_true, y_pred_proba[:, 1]),
        "accuracy": (y_pred == y_true).mean(),
        "f1": f1_score(y_true, y_pred),
    }

    # Calculate AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    metrics["auprc"] = auc(recall, precision)

    return metrics
