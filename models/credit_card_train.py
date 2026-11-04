import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve
)

from data.loader import load_client_data, input_dim, get_pos_weight
from models.credit_card import build_model


N_BANKS = 5
EPOCHS = 3
BATCH_SIZE = 512
LEARNING_RATE = 0.001
THRESHOLD = 0.5
SEED = 42


def find_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[np.argmax(f1_scores)])


def train_local_bank(bank_id):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train, X_val, y_val = load_client_data(
        bank_id=bank_id, fraud_type="credit_card"
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    model = build_model(input_dim("credit_card"))
    pos_weight = get_pos_weight(bank_id, "credit_card")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nBank {bank_id} | Train: {len(y_train)} | Validation: {len(y_val)}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(train_dataset):.4f}")

    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        probabilities = torch.sigmoid(model(X_val_tensor)).cpu().numpy()

    best_threshold = find_best_threshold(y_val, probabilities)
    default_predictions = (probabilities >= THRESHOLD).astype(int)
    tuned_predictions = (probabilities >= best_threshold).astype(int)

    results = {
        "bank_id": bank_id,
        "best_threshold": best_threshold,
        "default_precision": precision_score(y_val, default_predictions, zero_division=0),
        "default_recall": recall_score(y_val, default_predictions, zero_division=0),
        "default_f1": f1_score(y_val, default_predictions, zero_division=0),
        "tuned_precision": precision_score(y_val, tuned_predictions, zero_division=0),
        "tuned_recall": recall_score(y_val, tuned_predictions, zero_division=0),
        "tuned_f1": f1_score(y_val, tuned_predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_val, probabilities),
        "pr_auc": average_precision_score(y_val, probabilities)
    }

    print(f"Threshold: {best_threshold:.4f} | F1: {results['default_f1']:.4f} -> {results['tuned_f1']:.4f}")
    print(f"Precision: {results['tuned_precision']:.4f} | Recall: {results['tuned_recall']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f} | PR-AUC: {results['pr_auc']:.4f}")

    return results


def main():
    all_results = [train_local_bank(bank_id) for bank_id in range(N_BANKS)]

    print("\nLocal baseline summary")
    print(f"{'Bank':<6} {'Threshold':<11} {'Default F1':<12} {'Tuned F1':<10} {'ROC-AUC':<10} {'PR-AUC':<10}")

    for result in all_results:
        print(
            f"{result['bank_id']:<6} "
            f"{result['best_threshold']:<11.4f} "
            f"{result['default_f1']:<12.4f} "
            f"{result['tuned_f1']:<10.4f} "
            f"{result['roc_auc']:<10.4f} "
            f"{result['pr_auc']:<10.4f}"
        )



def main():
    all_results = [train_local_bank(bank_id) for bank_id in range(N_BANKS)]

    print("\nLocal baseline summary")
    print(f"{'Bank':<6} {'Threshold':<11} {'Default F1':<12} {'Tuned F1':<10} {'ROC-AUC':<10} {'PR-AUC':<10}")

    for result in all_results:
        print(
            f"{result['bank_id']:<6} "
            f"{result['best_threshold']:<11.4f} "
            f"{result['default_f1']:<12.4f} "
            f"{result['tuned_f1']:<10.4f} "
            f"{result['roc_auc']:<10.4f} "
            f"{result['pr_auc']:<10.4f}"
        )

    output_dir = Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "credit_card_local_baseline.csv"

    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
