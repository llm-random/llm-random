from enum import Enum, auto


# Class QualityFFMode auto enum:
class QualityFFMode(Enum):
    BASELINE = auto()
    QUALITY = auto()
    MAGNITUDE = auto()


QUALITY_TRAIN_MODEL = QualityFFMode.BASELINE
