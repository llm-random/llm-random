import torch
from lizrd.core import nn

from lizrd.support import ash


class MetricWriter(object):
    def __init__(self, tb_writer=None, step=0):
        self.tb_writer = tb_writer
        self.metrics = dict()
        self.step = step
        self.registered_loggers = []

    def update_step(self, step):
        self.step = step

    def get_and_add_metric(self, name):
        if name not in self.metrics:
            self.metrics[name] = 0
        new_name = f"{name}#{self.metrics[name]}"
        self.metrics[name] += 1
        return new_name

    def add_scalar(self, name, value):
        self.tb_writer.add_scalar(name, value, self.step)

    def register_logger(self, logger):
        self.registered_loggers.append(logger)

    def write_log(self):
        for logger in self.registered_loggers:
            logger.write_log()


METRIC_WRITER = MetricWriter()


class GenericLog(nn.Module):
    def __init__(self, name, metric_writer=None, subname=None):
        super(GenericLog, self).__init__()
        self.name = name
        if metric_writer is None:
            metric_writer = METRIC_WRITER
        self.metric_writer = metric_writer
        self.subname = subname
        self.last_value = None
        self.number = self.metric_writer.get_and_add_metric(f"{name}_{subname}")
        self.metric_writer.register_logger(self)

    def add_scalar(self, value):
        self.last_value = value

    def write_log(self):
        self.log()  # potentially adding scalars etc.
        if self.last_value is not None:
            full_name = f"{self.name}_{self.subname}/{self.number}"
            self.metric_writer.add_scalar(full_name, self.last_value)
        self.last_value = None

    def log(self):
        pass


@ash.check("...-> ...")
class LogValue(GenericLog):
    def __init__(self, name, metric_writer=None, aggregate=torch.mean, subname=None):
        if subname is None:
            subname = "val_" + aggregate.__name__
        super(LogValue, self).__init__(name, metric_writer, subname)
        self.aggregate = aggregate

    def forward(self, x):
        mean_x = self.aggregate(x)
        self.add_scalar(mean_x)
        return x


@ash.check("...-> ...")
class LogGradient(GenericLog):
    def __init__(self, name, metric_writer=None, aggregate=torch.mean, subname=None):
        if subname is None:
            subname = "grad_" + aggregate.__name__
        super(LogGradient, self).__init__(name, metric_writer, subname)
        self.aggregate = aggregate

        self.register_full_backward_hook(LogGradient.backward_hook_log_gradient)

    def forward(self, x):
        return x

    def backward_hook_log_gradient(self, grad_input, grad_output):
        grad = grad_output[0].detach()
        grad = self.aggregate(grad)
        self.add_scalar(grad)


class LogWeightValue(GenericLog):
    def __init__(
        self, name, weight_fn, metric_writer=None, aggregate=torch.mean, subname=None
    ):
        if subname is None:
            subname = "val_" + aggregate.__name__
        super(LogWeightValue, self).__init__(name, metric_writer, subname)
        self.aggregate = aggregate

        self.weight_fn = weight_fn

    def log(self):
        weight = self.weight_fn()
        weight = self.aggregate(weight)
        self.add_scalar(weight)


class LogWeightGradient(GenericLog):
    def __init__(
        self, name, weight_fn, metric_writer=None, aggregate=torch.mean, subname=None
    ):
        if subname is None:
            subname = "grad_" + aggregate.__name__
        super(LogWeightGradient, self).__init__(name, metric_writer, subname)
        self.aggregate = aggregate

        self.weight_fn = weight_fn

    def log(self):
        weight = self.weight_fn()
        grad = weight.grad
        if grad is None:
            print("Warning: weight.grad is None")
            return
        grad = self.aggregate(grad)
        self.add_scalar(grad)
