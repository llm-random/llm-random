from model import FunctionsRegistry
from token_reduction.model import (
    create_token_dropping_function,
    create_token_merging_function,
)

function_registry = FunctionsRegistry()
function_registry.register(
    "embedding", "token_dropping", create_token_dropping_function
)
function_registry.register("embedding", "token_merging", create_token_merging_function)
