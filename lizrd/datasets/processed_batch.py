from abc import ABC


class ProcessedBatch(ABC):
    def __init__(self, processed_examples):
        pass

    def __iter__(self):
        all_attrs = vars(self).values()
        return iter([attr for attr in all_attrs if hasattr(attr, "shape")])
        