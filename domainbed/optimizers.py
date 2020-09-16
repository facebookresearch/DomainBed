import torch

class GenericAdam():
    def __init__(self, parameters, hparams):
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=hparams["lr"],
            weight_decay=hparams['weight_decay']
        )

    """
    Returns the scheduler which is intended for the current algorithm.
    """
    def get_scheduler(self, algorithm):
        if algorithm == "RSC":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, threshold=0.0001, verbose=True)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, threshold=0.0001, verbose=True)
        return scheduler
