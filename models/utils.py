class PhysicsWarmup:
    def __init__(self, start_step, n_steps):
        self.start_step = start_step
        self.n_steps = n_steps

    @property
    def weight(self, step):
        return max(0, min((step - self.start_step) / self.n_steps, 1))
