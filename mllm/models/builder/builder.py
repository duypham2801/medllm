from typing import Dict, Any, Tuple

from torch import nn

from .build_perceptionGPT import load_pretrained_perceptionGPT
from .build_medgemma import load_pretrained_medgemma

PREPROCESSOR = Dict[str, Any]


# TODO: Registry
def load_pretrained(model_args, training_args) -> Tuple[nn.Module, PREPROCESSOR]:
    type_ = model_args.type
    if 'shikra' in type_ or 'perceptionGPT' in type_:
        return load_pretrained_perceptionGPT(model_args, training_args)
    elif 'medgemma' in type_:
        return load_pretrained_medgemma(model_args, training_args)
    else:
        raise ValueError(f"Unknown model type: {type_}. Supported types: 'shikra', 'perceptionGPT', 'medgemma'")
