from Server import Server
from Client import Client
from utils import get_paramters


def Start_simulation(fl_type, 
                     num_clients, 
                     num_rounds, 
                     model, 
                     trainsets,
                     testsets,
                     Server = None, 
                     Clients = None, 
                     aggregation = None,
                     train_func = None,
                     client_test_func = None,
                     server_test_func = None):
    
    model = model

    if fl_type == 'centralized':
        if Clients is None:
            Clients = [Clients(client_id = id,
                               model = model,
                               trainset = trainsets[id],
                               testset = testsets[id],
                               train_func = train_func,
                               client_test_func = client_test_func) 
                        for id in range(num_clients)]
        
        if Server is None:
            Server = Server(clients = Clients, 
                            model = model,
                            test_func = server_test_func)

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