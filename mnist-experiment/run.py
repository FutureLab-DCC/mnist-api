from programming_api.common import context, logger, metrics, measures, run
from programming_api import Model, Dataset, Experiment
import MNISTDataset, MNISTModel, MNISTExperiment


def generate_client_fn(**kwargs):

    def fn(id):
        model = MNISTModel.MNISTModel(suffix = id)
        
        dataset = MNISTDataset.MNISTDataset("./data/{}.npz".format(id), batch_size = 10, shuffle = False, num_workers = 0)
        
        return MNISTExperiment.MNISTExperiment(model, dataset)
    
    return fn

def evaluate_fn():
    def fn(server_round, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        model = MNISTModel.MNISTModel()
        model.set_parameters(parameters)
        
        dataset = MNISTDataset.MNISTDataset('./data/1.npz', batch_size = 10, shuffle = False, num_workers = 0)
        
        experiment = MNISTExperiment.MNISTExperiment(model, dataset)

        
        #params_dict = zip(model.state_dict().keys(), parameters)
        #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        #model.load_state_dict(state_dict, strict=True)
        
        loss, accuracy = experiment.validation_loop(dataset.dataloader(validation=True)) 

        return loss, {"accuracy": accuracy}

    return fn

if __name__ == '__main__':

    id = 0
    model = MNISTModel.MNISTModel(suffix = id)
    dataset = MNISTDataset.MNISTDataset("./data/{}.npz".format(id), batch_size = 10, shuffle = False, num_workers = 0)
    experiment = MNISTExperiment.MNISTExperiment(model, dataset, epochs = 1)

    experiment.train(model.get_parameters())
    experiment.evaluate(model.get_parameters())

    #for img, lbl in dataset.dataloader():
    #    print(len(img))

    #run(generate_client_fn, evaluate_fn)
