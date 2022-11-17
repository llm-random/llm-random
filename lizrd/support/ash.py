import einops
from lizrd.core import nn

DISABLE_CHECKS = False


def assert_shape(pattern, tensor, **kwargs):
    if DISABLE_CHECKS:
        return
    einops.rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)


class Check(nn.Module):
    def __init__(self, signature, layer, **kwargs):
        super(Check, self).__init__()
        self.inp_sig, self.out_sig = signature.split("->")
        self.inp_sig = self.inp_sig.split()
        self.out_sig = self.out_sig.split()
        self.constants = kwargs
        self.layer = layer

    def _check_and_add(self, shape, current, past, index):
        if current.isnumeric():
            assert shape[index] == int(current)
        elif current.isalpha():
            if current in past:
                assert shape[index] == past[current]
            else:
                past[current] = shape[index]
        elif current == "...":
            if "..." in past:
                assert shape[: index + 1] == past["..."]
            else:
                past["..."] = shape[: index + 1]

    def _check_and_add_all(self, shape, signature, past):
        for index, current in enumerate(signature, start=len(signature) - len(shape)):
            self._check_and_add(shape, current, past, index)

    def get_past(self):
        if DISABLE_CHECKS:
            return None
        past = {k: v for k, v in self.constants.items()}
        return past

    def before_layer(self, x, past):
        if DISABLE_CHECKS:
            return None
        self._check_and_add_all(x.shape, self.inp_sig, past)
        return past

    def after_layer(self, y, past):
        if DISABLE_CHECKS:
            return None
        self._check_and_add_all(y.shape, self.out_sig, past)

    def forward(self, x):
        past = self.get_past()
        self.before_layer(x, past)
        y = self.layer(x)
        self.after_layer(y, past)
        return y


def check(signature, **kwargs_shape):
    def noop_decorator(class_or_fun):
        return class_or_fun

    def decorator(class_or_fun):
        if "forward" in dir(class_or_fun):
            return class_decorator(class_or_fun)
        else:
            return function_decorator(class_or_fun)

    def class_decorator(module_class):
        old_forward = module_class.forward
        old_init = module_class.__init__

        def new_init(self, *args, **kwargs):
            old_init(self, *args, **kwargs)
            self._shape_checker = Check(signature, layer=None, **kwargs_shape)

        def new_forward(self, x):
            past = self._shape_checker.get_past()
            self._shape_checker.before_layer(x, past)
            y = old_forward(self, x)
            self._shape_checker.after_layer(y, past)
            return y

        module_class.__init__ = new_init
        module_class.forward = new_forward
        return module_class

    def function_decorator(function):
        def new_function(*args, **kwargs):
            return Check(signature, function(*args, **kwargs), **kwargs_shape)

        return new_function

    if DISABLE_CHECKS:
        return noop_decorator
    return decorator
