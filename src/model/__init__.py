"""
Composants du World Model.

Modules à venir :
    buffer.py   - Replay buffer (stockage des transitions)
    encoder.py  - Encoder obs → embedding
    rssm.py     - Recurrent State Space Model (cœur de la dynamique)
    decoder.py  - Decoder latent → obs (reconstruction pour la loss)
    heads.py    - Heads reward, continue
    actor.py    - Policy (l'agent qui décide)
    critic.py   - Value function
"""

from .buffer import ReplayBuffer, ImageReplayBuffer
from .encoder import Encoder, CNNEncoder
from .decoder import Decoder, CNNDecoder
from .rssm import RSSM, sample_categorical_straight_through
from .heads import RewardHead, ContinueHead
from .actor import Actor
from .critic import Critic
from .rnd import RNDModule

__all__ = [
    "ReplayBuffer", "ImageReplayBuffer",
    "Encoder", "CNNEncoder", "Decoder", "CNNDecoder",
    "RSSM", "sample_categorical_straight_through",
    "RewardHead", "ContinueHead",
    "Actor", "Critic",
    "RNDModule",
]
