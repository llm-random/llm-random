# autoflake: skip_file
"""
This implementation is a copy of cc_train.py from the research.conditional.utils package.
Though, it is not a direct copy, as it has been modified to fit the needs of the current project.
It is missing:
- the logging of the example batch
- model parallelism
- setting anomaly ( set_detect_anomaly)
- distributed (DDP/FSDP) training
- checkpointing
- the logging of the model's learnable parameters
- the logging of the model's non-embedding learnable parameters
- compiling model
- profiler
"""

from .__main__ import main
