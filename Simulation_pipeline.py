from Server import Server
from Client import Client
from utils import get_paramters


def Start_simulation(fl_type, num_clients, num_rounds, model, Server = None, Clients = None, aggregation = 'fedavg'):
    if fl_type == 'centralized':
        if Clients is None:
            Clients = [Clients() for _ in range(num_clients)]
        
        if Server is None:
            Server = Server(clients = Clients)

        parameters = get_paramters(model)

        Server.train(num_rounds = num_rounds,
                     parameters = parameters,
                     aggregation = aggregation)
        
        Server.evaluate()

        # Lưu kết quả huấn luyện


        # Lưu model (Optional)


    elif fl_type == 'decentralized':
        pass

    elif fl_type == 'asynchronous':
        pass

    elif fl_type == 'continual':
        pass

    else:
        raise ValueError("Invalid FL type. Choose from 'centralized', 'decentralized', 'asynchronous', or 'continual'.")