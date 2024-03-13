from programming_api.Experiment import Experiment
from programming_api.common import context, logger, metrics, measures
import MNISTModel, MNISTDataset
import numpy as np

import torch

class MNISTExperiment(Experiment):
    def __init__(self, model, dataset, **kwargs):
        super(MNISTExperiment, self).__init__(model, dataset)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.epochs = kwargs.get('epochs', 1)

    def training_loop(self, data_loader):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        for epoca in range(self.epochs):
            correct, custo = 0, 0.0
            for images, labels in data_loader:
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                custo += np.mean(loss.cpu().item())
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            measures.log(self, metrics.CROSSENTROPY, custo, epoch = epoca)
            measures.log(self, metrics.ACCURACY, correct / len(self.dataset), epoch = epoca)

    def validation_loop(self, data_loader):
        correct, loss = 0, 0.0
        self.model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()/ len(labels)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(self.dataset)
        measures.log(self, metrics.CROSSENTROPY, loss, validation=True)
        measures.log(self, metrics.ACCURACY, accuracy, validation=True)

        return loss, accuracy
