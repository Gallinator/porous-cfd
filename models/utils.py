class PhysicsWarmup:
    def __init__(self, start_epoch, n_epochs):
        self.start_epoch = start_epoch
        self.n_epochs = n_epochs

    def weight(self, epoch):
        return max(0, min((epoch - self.start_epoch) / self.n_epochs, 1))
