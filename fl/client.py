import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class SimpleFraudModel(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def create_dummy_data(bank_id, num_samples=200):
    # Different seed gives each simulated bank different private data
    torch.manual_seed(42 + bank_id)

    # 10 input features for each transaction
    X = torch.randn(num_samples, 10)

    # Create binary fraud labels: 0 = normal, 1 = fraud
    y = (X.sum(dim=1) > 0).float()

    # 80% training, 20% validation
    split = int(0.8 * num_samples)

    train_dataset = TensorDataset(
        X[:split],
        y[:split]
    )

    val_dataset = TensorDataset(
        X[split:],
        y[split:]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader, val_loader


def get_weights(model):
    return [
        value.cpu().numpy()
        for _, value in model.state_dict().items()
    ]
def set_weights(model, weights):
    keys = model.state_dict().keys()

    state_dict = {
        key: torch.tensor(value)
        for key, value in zip(keys, weights)
    }

    model.load_state_dict(state_dict, strict=True)


def train_model(model, train_loader, epochs=1):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for X, y in train_loader:
            optimizer.zero_grad()

            outputs = model(X)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Local epoch {epoch + 1}: "
            f"loss = {average_loss:.4f}"
        )


def evaluate_model(model, val_loader):
    criterion = nn.BCEWithLogitsLoss()

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in val_loader:
            outputs = model(X)

            loss = criterion(outputs, y)
            total_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= 0.5).float()

            correct += (predictions == y).sum().item()
            total += y.size(0)

    average_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return average_loss, accuracy


class FraudClient(fl.client.NumPyClient):
    def __init__(self, bank_id, model, train_loader, val_loader):
        self.bank_id = bank_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        print(f"Bank {self.bank_id}: sending model parameters")
        return get_weights(self.model)

    def fit(self, parameters, config):
        print(f"Bank {self.bank_id}: starting local training")

        # Receive the current global model from the server
        set_weights(self.model, parameters)

        # Train using this bank's private data
        train_model(
            self.model,
            self.train_loader,
            epochs=1
        )

        print(f"Bank {self.bank_id}: local training complete")

        # Return updated parameters and number of training samples
        return (
            get_weights(self.model),
            len(self.train_loader.dataset),
            {"bank_id": self.bank_id}
        )

    def evaluate(self, parameters, config):
        # Load the latest global model
        set_weights(self.model, parameters)

        # Evaluate using this bank's private validation data
        loss, accuracy = evaluate_model(
            self.model,
            self.val_loader
        )

        print(
            f"Bank {self.bank_id}: "
            f"validation accuracy = {accuracy:.4f}"
        )

        return (
            float(loss),
            len(self.val_loader.dataset),
            {"accuracy": float(accuracy)}
        )

def create_client(bank_id):
    model = SimpleFraudModel(input_dim=10)

    train_loader, val_loader = create_dummy_data(
        bank_id=bank_id
    )

    return FraudClient(
        bank_id=bank_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )