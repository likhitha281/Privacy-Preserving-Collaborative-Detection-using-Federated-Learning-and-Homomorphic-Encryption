
import numpy as np
import torch
import torch.nn as nn


class CreditCardFraudMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

    def get_weights(self):
        return get_weights(self)

    def set_weights(self, weights):
        set_weights(self, weights)


def build_model(input_dim):
    return CreditCardFraudMLP(input_dim)


def get_weights(model):
    return [param.detach().cpu().numpy().copy() for param in model.parameters()]


def set_weights(model, weights):
    params = list(model.parameters())

    if len(params) != len(weights):
        raise ValueError("Number of weight arrays does not match model parameters")

    with torch.no_grad():
        for param, weight in zip(params, weights):
            weight = np.asarray(weight, dtype=np.float32)

            if weight.shape != tuple(param.shape):
                raise ValueError("Weight shape does not match model parameter")

            param.copy_(torch.as_tensor(weight, device=param.device, dtype=param.dtype))
