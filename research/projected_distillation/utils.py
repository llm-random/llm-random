import numpy as np

FREEZE_PARAMS_REGULES = [
    "logging_ff_pre_relu_p11",
    "logging_ff_pre_relu_p12",
    "logging_ff_pre_relu_p21",
    "logging_ff_pre_relu_p22",

]


def freez_projected_params(model):
    for name, param in model.named_parameters():
        if any([reg in name for reg in FREEZE_PARAMS_REGULES]):  # Check if the parameter belongs to layer1
            param.requires_grad = False
    return model 

