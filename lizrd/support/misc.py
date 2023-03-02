import datetime
from typing import Optional, List


def tags_to_name(tags: Optional[List[str]]) -> str:
    return "_".join(tags) if tags else ""


def make_concise_datetime() -> str:
    now = datetime.datetime.now()
    return str(now.year)[-2:] + "_" + now.strftime("%m-%d_%H:%M:%S")


def count_parameters(model, args, VOCAB_SIZE):
    model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = 2 * VOCAB_SIZE * args.dmodel
    last_layer_params = args.cutoff * args.dmodel
    model_n_params -= embedding_params + last_layer_params
    return model_n_params
