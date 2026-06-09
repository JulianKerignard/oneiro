"""
src_jax/model — modules JAX/Flax NNX du mini-DreamerV3.
"""

from .encoder import Encoder, CNNEncoder
from .decoder import Decoder, CNNDecoder
from .actor import Actor
from .critic import Critic
from .heads import (
    RewardHead,
    ContinueHead,
    symlog,
    symexp,
    twohot_encode,
    twohot_decode,
    N_BINS,
    BIN_MIN_SYMLOG,
    BIN_MAX_SYMLOG,
)
from .rssm import RSSM, sample_categorical_straight_through
from .rnd import RNDModule, RNDStats

__all__ = [
    "Encoder",
    "CNNEncoder",
    "Decoder",
    "CNNDecoder",
    "Actor",
    "Critic",
    "RewardHead",
    "ContinueHead",
    "RSSM",
    "sample_categorical_straight_through",
    "RNDModule",
    "RNDStats",
    "symlog",
    "symexp",
    "twohot_encode",
    "twohot_decode",
    "N_BINS",
    "BIN_MIN_SYMLOG",
    "BIN_MAX_SYMLOG",
]
