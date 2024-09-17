from flautim.pytorch.common import run_centralized, get_argparser
from flautim.pytorch import Model, Dataset
from flautim.pytorch.federated import Experiment
import MNISTDataset, MNISTModel, MNISTExperiment

if __name__ == '__main__':

    parser, context, backend, logger, measures = get_argparser()
    
    model = MNISTModel.MNISTModel(context, suffix = 'FL-Global')

    files = [context.path +"data/{}.npz".format(i) for i in range(3)]
    
    dataset = MNISTDataset.MNISTDataset(files, batch_size = 10, shuffle = False, num_workers = 0)
    
    experiment = MNISTExperiment.MNISTExperiment(model, dataset, measures, logger, context)

    run_centralized(experiment)
