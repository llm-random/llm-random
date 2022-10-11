# Here be dragons.
import torch.nn

# This whole thing is a bit of a hack.
# When we create a new layer, we want it to inherit from the `Module` class.
# However, we also want to add a child class to the layer. It's like a reverse inheritance.
# In any case, when we create a new layer, let's say FeedForward, which inherits from core.nn.Module,
# by default we will have this inheritance chain:
# FeedForward -> core.nn.Module -> torch.nn.Module
# However, with this hack, we will have this inheritance chain when we create an object:
# fake_FeedForward -> OverModule -> original_FeedForward -> core.nn.Module -> torch.nn.Module
# Note that both fake_FeedForward and original_FeedForward have the same name.

# The above will allow us to add a fake-stack, profiling, etc. Ideally, this whole abstraction will
# be completely invisible outside this file.

# Caveats:
# * We cannot use __new__ in new classes. This is not a problem 99% of the time.
# * We will have to "recreate" all the classes of torch.nn. Just add them here.

# Here you can read more about inheritance and method resolution order:
# http://python-history.blogspot.com/2010/06/method-resolution-order.html


MEMOIZED_CLASS_DICT = dict()


def get_memoized_class(cls_name, parents):
    if cls_name not in MEMOIZED_CLASS_DICT:
        MEMOIZED_CLASS_DICT[cls_name] = type(cls_name, parents, dict())
    return MEMOIZED_CLASS_DICT[cls_name]


# This class will be a parent of all our modules. It also adds OverModule as a child.
class Module(torch.nn.Module):
    def __new__(cls, *args, **kwargs):
        if issubclass(cls, OverModule):
            return super(Module, cls).__new__(cls)
        new_cls = get_memoized_class(cls.__name__, (OverModule, cls))
        return new_cls.__new__(new_cls)

    def __init__(self, *args, **kwargs):
        super(Module, self).__init__(*args, **kwargs)


# This class will be a child of all our modules. It will allow us to add a stack,
# and to profile all the layers. We just cannot use __new__ method in Layers.
class OverModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(OverModule, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        result = super(OverModule, self).forward(*args, **kwargs)
        return result


Parameter = torch.nn.Parameter  # it inherits from Tensor


# Maybe we should create those classes as functions? I'm not sure. Probably it doesn't matter.
class ModuleList(torch.nn.ModuleList, Module):
    pass


class Sequential(torch.nn.Sequential, Module):
    pass


class Linear(torch.nn.Linear, Module):
    pass


class ReLU(torch.nn.ReLU, Module):
    pass


class LayerNorm(torch.nn.LayerNorm, Module):
    pass


class Embedding(torch.nn.Embedding, Module):
    pass
