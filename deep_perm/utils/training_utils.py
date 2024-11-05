def update_metrics_per_epoch(metrics_per_epoch, epoch, train_loss, train_acc, val_metrics, dataiq):
    """Update metrics dictionary with current epoch results"""
    metrics_per_epoch["epoch"].append(epoch)
    metrics_per_epoch["train_loss"].append(train_loss)
    metrics_per_epoch["val_loss"].append(val_metrics["loss"])
    metrics_per_epoch["train_accuracy"].append(train_acc)
    metrics_per_epoch["val_accuracy"].append(val_metrics["accuracy"])
    metrics_per_epoch["val_auroc"].append(val_metrics["auroc"])
    metrics_per_epoch["val_auprc"].append(val_metrics["auprc"])
    metrics_per_epoch["val_f1"].append(val_metrics["f1"])

    # DataIQ specific metrics
    metrics_per_epoch["confidence"].append(dataiq.confidence.copy())
    metrics_per_epoch["aleatoric"].append(dataiq.aleatoric.copy())
    metrics_per_epoch["correctness"].append(dataiq.correctness.copy())
    metrics_per_epoch["entropy"].append(dataiq.entropy.copy())
    metrics_per_epoch["mi"].append(dataiq.mi.copy())
    metrics_per_epoch["variability"].append(dataiq.variability.copy())
