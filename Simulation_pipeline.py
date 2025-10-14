from Server import Server
from Client import Client


def Start_simulation(fl_type, Server, Client):
    
    if fl_type == 'centralized':
        pass

    elif fl_type == 'decentralized':
        pass

    elif fl_type == 'asynchronous':
        pass

    elif fl_type == 'continual':
        pass

    else:
        raise ValueError("Invalid FL type. Choose from 'centralized', 'decentralized', 'asynchronous', or 'continual'.")