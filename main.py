from Data_Processing import Load_data, Data_partition
from Client import Client
from Server import Server
from Start_simulation import Start_simulation

# Hyperparameters
dataset = "MNIST"
model = "CNN"
num_clients = 10
partition_method = "dirichlet"  


# Data Processing

transform = None
data = Load_data(dataset, transform)

clients_data = Data_partition(      )



# Khởi tạo Client-Server

server = Server()

client = Client()


# Simulation Pipeline

Start_simulation(
    Server = server,
    Client = client,
)