import Server
import Client
from utils import get_paramters


def Start_simulation(fl_type, 
                     num_clients, 
                     num_rounds, 
                     model, 
                     trainsets,
                     testsets,
                     file_name = "FL.csv",
                     server = None, 
                     clients = None, 
                     aggregation = None,
                     train_func = None,
                     client_test_func = None,
                     server_test_func = None):
    
    model = model

    if fl_type == 'centralized':
        if clients is None:
            clients = [Client(client_id = id,
                               model = model,
                               trainset = trainsets[id],
                               testset = testsets[id],
                               train_func = train_func,
                               client_test_func = client_test_func) 
                        for id in range(num_clients)]
        
        if server is None:
            server = Server(clients = clients, 
                            model = model,
                            test_func = server_test_func,
                            num_rounds = num_rounds,
                            aggregation = aggregation)

        for round in range(num_rounds):
            results = Server.train()
            results = Server.evaluate()
            client_results = client.local_test()

        if server_test_func is not None:
            results = Server.evaluate()
            print("Final evaluation results:", results)

        if client_test_func is not None:
            for client in clients:
                client_results = client.local_test()
                print(f"Client {client.client_id} evaluation results:", client_results)


    elif fl_type == 'decentralized':
        pass

    elif fl_type == 'asynchronous':
        pass

    elif fl_type == 'continual':
        pass

    else:
        raise ValueError("Invalid FL type. Choose from 'centralized', 'decentralized', 'asynchronous', or 'continual'.")