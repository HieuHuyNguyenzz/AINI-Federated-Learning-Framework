from utils import get_parameters, set_parameters

class Client:
    def __init__(self, name, client_id, model, trainset, testset):
        self.name = name
        self.client_id = client_id
        self.model = model  
        self.trainset = trainset
        self.testset = testset


    def local_train(self, train, config):
        model, results = train(self.trainset, config)
        return get_parameters(model), results


    def local_test(self, test, config):
        results = test(self.testset, config)
        return results