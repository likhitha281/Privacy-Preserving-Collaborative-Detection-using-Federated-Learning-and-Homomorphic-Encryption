import flwr as fl

from fl.client import create_client
from fl.strategy import FraudFedAvg


NUM_CLIENTS = 5
NUM_ROUNDS = 3


def client_fn(context):
    """
    Create one Flower client for each simulated bank.
    """

    bank_id = int(context.node_config["partition-id"])

    print(f"Creating Flower client for Bank {bank_id}")

    client = create_client(bank_id)

    return client.to_client()


def weighted_average(metrics):
    """
    Aggregate evaluation accuracy across clients,
    weighted by the number of validation samples.
    """

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    weighted_accuracy = sum(
        num_examples * metric["accuracy"]
        for num_examples, metric in metrics
    ) / total_examples

    return {
        "accuracy": weighted_accuracy
    }


def run_simulation():
    """
    Run federated learning using three simulated banks.
    """

    strategy = FraudFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,

        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,

        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print("\nStarting Federated Learning Simulation")
    print(f"Number of banks: {NUM_CLIENTS}")
    print(f"Number of rounds: {NUM_ROUNDS}\n")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),

        strategy=strategy,
    )


if __name__ == "__main__":
    run_simulation()