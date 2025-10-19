import random
from flwr.common import (
    NDArrays,
)
import numpy as np
from functools import reduce


class Server:
    def __init__(self, 
                 clients, 
                 model, 
                 aggregation=None, 
                 test_func=None, 
                 num_rounds = 1, 
                 file_name = "FL.csv",
                 server_testset = None,
                 Hyperparameters = None
                 ):
        self.clients = clients
        self.global_model = model
        self.server_testset = server_testset
        self.test_func = test_func
        self.Hyperparameters = Hyperparameters
        self.num_rounds = num_rounds
        self.file_name = file_name
        if aggregation is None:
            self.aggregation = self.models_aggregation


    def sample_clients(self, fraction, num_clients):
        clients_id = [client.client_id for client in self.clients]
        num_sampled_clients = min(int(fraction * num_clients), 1)
        sampled_clients_id = random.sample(clients_id, num_sampled_clients)
        return sampled_clients_id
    

    def train(self, num_rounds, fraction_fit):
        for round in range(num_rounds):
            sampled_clients_id = self.sample_clients(fraction_fit, len(self.clients))
            
            training_config = {"round": round} + self.Hyperparameters
            results = [self.clients[client_id].local_train(self.global_model, training_config) for client_id in sampled_clients_id]

            self.global_model = self.aggregation(results)

    
    def models_aggregation(results: list[tuple[NDArrays, float]]) -> NDArrays:
        # FedAvg aggregation
        num_examples_total = sum(num_examples for (_, num_examples) in results)

        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime


    def server_test(self):
        if self.server_test is None or self.test_func is None:
            return None

        results = self.test_func(self.global_model, self.server_testset)
        return results



    def client_test(self):
        test_config = {}
        results = [client.local_test(self.global_model, test_config) for client in self.clients]
        return results


    def evaluate_aggregation(self):
        pass


    def save_results(self):
        pass