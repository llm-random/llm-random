class LinearTokenReductionScheduler:
    def __init__(self, start, stop, steps):
        self.start = start
        self.stop = stop
        self.steps = steps
        self.current_step = 0

    def set_step(self, new_step):
        self.current_step = new_step

    def step(self):
        self.current_step += 1

    @property
    def value(self):
        if self.current_step < self.steps:
            return round(
                self.start + (self.stop - self.start) * self.current_step / self.steps
            )
        else:
            return self.stop
