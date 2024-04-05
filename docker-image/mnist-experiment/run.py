from programming_api.common import run, get_argparser
from programming_api import Model, Dataset, Experiment
import MNISTDataset, MNISTModel, MNISTExperiment



def generate_client_fn(context, measures):

    def create_client_fn(id):

        model = MNISTModel.MNISTModel(context, suffix = id)
        
        dataset = MNISTDataset.MNISTDataset("./data/{}.npz".format(id), batch_size = 10, shuffle = False, num_workers = 0)
        
        return MNISTExperiment.MNISTExperiment(model, dataset, measures)
        
    return create_client_fn
    

def evaluate_fn(context, measures):
    def fn(server_round, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        model = MNISTModel.MNISTModel(context)
        model.set_parameters(parameters)
        
        dataset = MNISTDataset.MNISTDataset('./data/1.npz', batch_size = 10, shuffle = False, num_workers = 0)
        
        experiment = MNISTExperiment.MNISTExperiment(model, dataset, measures)

        
        #params_dict = zip(model.state_dict().keys(), parameters)
        #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        #model.load_state_dict(state_dict, strict=True)
        
        loss, accuracy = experiment.validation_loop(dataset.dataloader(validation=True)) 

        return loss, {"accuracy": accuracy}

    return fn

if __name__ == '__main__':

    parser, context, backend, logger, measures = get_argparser()

    #id = 0
    #model = MNISTModel.MNISTModel(context, suffix = id)
    #dataset = MNISTDataset.MNISTDataset("./data/{}.npz".format(id), batch_size = 10, shuffle = False, num_workers = 0)
    #experiment = MNISTExperiment.MNISTExperiment(model, dataset, logger, measures, epochs = 1)
    
    #experiment.fit(model.get_parameters())
    #experiment.evaluate(model.get_parameters())
    
    
    client_fn_callback = generate_client_fn(context, measures)
    evaluate_fn_callback = evaluate_fn(context, measures)

    run(client_fn_callback, evaluate_fn_callback)
