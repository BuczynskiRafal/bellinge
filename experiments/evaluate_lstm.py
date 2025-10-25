import sys
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.lstm import FlashFloodLSTM
from src.data.dataloader import create_dataloaders
from src.evaluation.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_roc_curve,
    compute_pr_curve
)


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to: {save_path}")


def plot_roc_curve(fpr, tpr, auc_score, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve to: {save_path}")


def plot_pr_curve(precision, recall, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved PR curve to: {save_path}")


def main():
    with open('configs/lstm_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    _, _, test_loader = create_dataloaders(
        train_X_path=config['data']['train_X'],
        train_y_path=config['data']['train_y'],
        val_X_path=config['data']['val_X'],
        val_y_path=config['data']['val_y'],
        test_X_path=config['data']['test_X'],
        test_y_path=config['data']['test_y'],
        batch_size=config['training']['batch_size']
    )

    print("Loading model...")
    model = FlashFloodLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    checkpoint_path = Path(config['artifacts']['models_dir']) / f"lstm_epoch_{config['training']['epochs']}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from: {checkpoint_path}")

    print("Evaluating on test set...")
    y_pred_proba, y_true = evaluate_model(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred_proba)
    print("\nTest Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    results_dir = Path(config['artifacts']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    viz_dir = Path(config['artifacts']['viz_dir'])
    viz_dir.mkdir(parents=True, exist_ok=True)

    cm = compute_confusion_matrix(y_true, y_pred_proba)
    plot_confusion_matrix(cm, viz_dir / 'confusion_matrix.png')

    fpr, tpr, _ = compute_roc_curve(y_true, y_pred_proba)
    plot_roc_curve(fpr, tpr, metrics['roc_auc'], viz_dir / 'roc_curve.png')

    precision, recall, _ = compute_pr_curve(y_true, y_pred_proba)
    plot_pr_curve(precision, recall, viz_dir / 'precision_recall_curve.png')

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
