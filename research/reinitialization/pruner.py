from typing import TYPE_CHECKING

# to avoid cycle import while using hints
if TYPE_CHECKING:
    from research.reinitialization.linears import RandomPruneLayer


class Pruner:
    def __init__(self, n_steps_prune: int, prob: float):
        self.n_steps_prune = n_steps_prune
        self.prob = prob
        self.current_step = 0
        self.layers = []

    def register(self, layer: "RandomPruneLayer"):
        self.layers.append(layer)

    def step(self):
        if self.current_step % self.n_steps_prune == 0:
            print("Pruning step")
            for layer in self.layers:
                layer.prune(self.prob)
        self.current_step += 1
