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


    def local_train(self, model, config):
        self.model = set_parameters(self.model, model)
        model, results = self.train_func(self.trainset, self.model, config)
        return {"client_id": self._id, "params": get_parameters(model), "training results": results}


    def local_test(self, model, config):
        self.model = set_parameters(self.model, model)
        results = self.test_func(self.testset, self.model, config)
        return {"client_id": self._id, "test results": results}