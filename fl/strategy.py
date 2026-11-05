import flwr as fl


class FraudFedAvg(fl.server.strategy.FedAvg):
    """
    Federated Averaging strategy for the fraud detection project.

    Flower's built-in FedAvg performs weighted aggregation based on
    the number of training examples returned by each client.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):
        """
        Aggregate model updates received from the Flower clients.
        """

        print(
            f"\nRound {server_round}: "
            f"received updates from {len(results)} clients"
        )

        if failures:
            print(
                f"Round {server_round}: "
                f"{len(failures)} client failure(s)"
            )

        # Use Flower's built-in weighted FedAvg aggregation.
        aggregated_result = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        if aggregated_result is not None:
            print(
                f"Round {server_round}: "
                "FedAvg aggregation completed successfully"
            )
        else:
            print(
                f"Round {server_round}: "
                "aggregation returned no result"
            )

        return aggregated_result