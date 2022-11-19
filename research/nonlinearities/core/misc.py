def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)