from utils import get_parameters, set_parameters

class Client:
    def __init__(self, 
                 client_id, 
                 model, 
                 trainset, 
                 testset,
                 train_func,
                 client_test_func):
        self.client_id = client_id
        self.model = model  
        self.trainset = trainset
        self.testset = testset
        self.train_func = train_func
        self.test_func = client_test_func


    def local_train(self, config):
        model, results = self.train_func(self.trainset, self.model, config)
        return get_parameters(model), results


    def local_test(self, config):
        results = self.test_func(self.testset, self.model, config)
        return results