class ConstScheduler:
    """ Scheduler for unstructured pruning.
    Prune with fixed probability every predefined number of steps.
    """
    def __init__(self, prob: float, n_steps):
        self.current_step = 0
        self.prob = prob
        self.n_steps = n_steps

    def step(self):
        # Returns probability (if time to prune) and None in the other case
        if self.current_step % self.n_steps == 0:
            res = self.prob
        else:
            res = None
        self.current_step += 1
        return res
