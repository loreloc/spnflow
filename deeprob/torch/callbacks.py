import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given number of consecutive epochs"""
    def __init__(self, model, patience=1, delta=1e-4):
        """
        Instantiate an EarlyStopping object.

        :param model: The model to monitor.
        :param patience: The number of consecutive epochs to wait.
        :param delta: The minimum change of the monitored quantity.
        """
        self.model = model
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.should_stop = False
        self.counter = 0
        self.best_state = None

    def __call__(self, loss):
        """
        Call the object.

        :param loss: The validation loss measured.
        """
        # Check if an improved of the loss happened
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0

            # Update the best model state parameters
            self.best_state = self.model.state_dict()
        else:
            self.counter += 1

        # Check if the training should stop
        if self.counter >= self.patience:
            self.should_stop = True
