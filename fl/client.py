import flwr as fl
import numpy as np
import torch
import torch.nn as nn

from data.loader import load_client_data, input_dim, get_pos_weight

from models.credit_card import build_model as build_credit_card_model
from models.account_takeover import build_model as build_account_takeover_model
from models.identity import build_model as build_identity_model


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

FRAUD_TYPE = "credit_card"
NUM_BANKS = 5
LOCAL_EPOCHS = 1


# ---------------------------------------------------------
# Model selection
# ---------------------------------------------------------

def build_fraud_model(fraud_type):
    """
    Build the correct fraud detection model
    for the selected fraud type.
    """

    dim = input_dim(
        fraud_type,
        n_banks=NUM_BANKS
    )

    if fraud_type == "credit_card":
        return build_credit_card_model(dim)

    if fraud_type == "account_takeover":
        return build_account_takeover_model(dim)

    if fraud_type == "identity_fraud":
        return build_identity_model(dim)

    raise ValueError(
        f"Unknown fraud type: {fraud_type}"
    )


# ---------------------------------------------------------
# Model weight exchange
# ---------------------------------------------------------

def get_weights(model):
    """
    Convert PyTorch model parameters into NumPy arrays
    so Flower can send them to the server.
    """

    return [
        parameter.detach().cpu().numpy().copy()
        for parameter in model.parameters()
    ]


def set_weights(model, weights):
    """
    Load global model parameters received from Flower
    into the local PyTorch model.
    """

    parameters = list(model.parameters())

    if len(parameters) != len(weights):
        raise ValueError(
            "Number of received weight arrays "
            "does not match model parameters."
        )

    with torch.no_grad():

        for parameter, weight in zip(
            parameters,
            weights
        ):

            weight = np.asarray(
                weight,
                dtype=np.float32
            )

            if tuple(weight.shape) != tuple(parameter.shape):
                raise ValueError(
                    "Weight shape does not match "
                    "model parameter shape."
                )

            parameter.copy_(
                torch.as_tensor(
                    weight,
                    dtype=parameter.dtype,
                    device=parameter.device
                )
            )


# ---------------------------------------------------------
# Local training
# ---------------------------------------------------------

def train_model(
    model,
    X_train,
    y_train,
    pos_weight,
    epochs=1
):
    """
    Train one client's model locally using only
    that simulated bank's private training data.
    """

    model.train()

    # copy=True avoids PyTorch warnings caused by
    # read-only NumPy arrays loaded from parquet.
    X_tensor = torch.from_numpy(
        np.array(
            X_train,
            dtype=np.float32,
            copy=True
        )
    )

    y_tensor = torch.from_numpy(
        np.array(
            y_train,
            dtype=np.float32,
            copy=True
        )
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight],
            dtype=torch.float32
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    batch_size = 512

    for epoch in range(epochs):

        permutation = torch.randperm(
            len(X_tensor)
        )

        total_loss = 0.0

        for start in range(
            0,
            len(X_tensor),
            batch_size
        ):

            indices = permutation[
                start:start + batch_size
            ]

            X_batch = X_tensor[indices]
            y_batch = y_tensor[indices]

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * len(indices)
            )

        average_loss = (
            total_loss / len(X_tensor)
        )

        print(
            f"Local epoch {epoch + 1}: "
            f"loss = {average_loss:.4f}"
        )


# ---------------------------------------------------------
# Local evaluation
# ---------------------------------------------------------

def evaluate_model(
    model,
    X_val,
    y_val
):
    """
    Evaluate the global model on one bank's
    private validation partition.
    """

    model.eval()

    # copy=True avoids read-only NumPy array warnings.
    X_tensor = torch.from_numpy(
        np.array(
            X_val,
            dtype=np.float32,
            copy=True
        )
    )

    y_tensor = torch.from_numpy(
        np.array(
            y_val,
            dtype=np.float32,
            copy=True
        )
    )

    criterion = nn.BCEWithLogitsLoss()

    batch_size = 2048

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            batch_size
        ):

            X_batch = X_tensor[
                start:start + batch_size
            ]

            y_batch = y_tensor[
                start:start + batch_size
            ]

            outputs = model(X_batch)

            loss = criterion(
                outputs,
                y_batch
            )

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            total_loss += (
                loss.item() * len(y_batch)
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += len(y_batch)

    average_loss = total_loss / total

    accuracy = correct / total

    return average_loss, accuracy


# ---------------------------------------------------------
# Flower client
# ---------------------------------------------------------

class FraudClient(fl.client.NumPyClient):
    """
    Flower client representing one simulated bank.
    """

    def __init__(
        self,
        bank_id,
        fraud_type,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        pos_weight
    ):

        self.bank_id = bank_id
        self.fraud_type = fraud_type

        self.model = model

        self.X_train = X_train
        self.y_train = y_train

        self.X_val = X_val
        self.y_val = y_val

        self.pos_weight = pos_weight


    def get_parameters(self, config):
        """
        Return this client's current model parameters.
        """

        print(
            f"Bank {self.bank_id}: "
            "sending model parameters"
        )

        return get_weights(
            self.model
        )


    def fit(
        self,
        parameters,
        config
    ):
        """
        Receive the global model, train locally,
        and return the updated model parameters.
        """

        print(
            f"Bank {self.bank_id}: "
            "starting local training"
        )

        # Load the latest global model.
        set_weights(
            self.model,
            parameters
        )

        # Train only on this bank's private data.
        train_model(
            self.model,
            self.X_train,
            self.y_train,
            self.pos_weight,
            epochs=LOCAL_EPOCHS
        )

        print(
            f"Bank {self.bank_id}: "
            "local training complete"
        )

        # len(X_train) is used by Flower for
        # sample-count-weighted FedAvg.
        return (
            get_weights(self.model),
            len(self.X_train),
            {
                "bank_id": self.bank_id
            }
        )


    def evaluate(
        self,
        parameters,
        config
    ):
        """
        Evaluate the latest global model using
        this bank's private validation data.
        """

        set_weights(
            self.model,
            parameters
        )

        loss, accuracy = evaluate_model(
            self.model,
            self.X_val,
            self.y_val
        )

        print(
            f"Bank {self.bank_id}: "
            f"validation accuracy = "
            f"{accuracy:.4f}"
        )

        return (
            float(loss),
            len(self.X_val),
            {
                "accuracy":
                    float(accuracy)
            }
        )


# ---------------------------------------------------------
# Flower client factory
# ---------------------------------------------------------

def create_client(
    bank_id,
    fraud_type=FRAUD_TYPE
):
    """
    Create one Flower client using the real
    IEEE-CIS partition for a simulated bank.
    """

    print(
        f"Loading real data for "
        f"Bank {bank_id} "
        f"({fraud_type})"
    )

    X_train, y_train, X_val, y_val = (
        load_client_data(
            bank_id=bank_id,
            fraud_type=fraud_type,
            n_banks=NUM_BANKS
        )
    )

    model = build_fraud_model(
        fraud_type
    )

    pos_weight = get_pos_weight(
        bank_id=bank_id,
        fraud_type=fraud_type,
        n_banks=NUM_BANKS
    )

    print(
        f"Bank {bank_id}: "
        f"{len(X_train)} training samples, "
        f"{len(X_val)} validation samples, "
        f"{X_train.shape[1]} features"
    )

    return FraudClient(
        bank_id=bank_id,
        fraud_type=fraud_type,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        pos_weight=pos_weight
    )