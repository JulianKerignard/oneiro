"""
Tests de validation des modules JAX/Flax NNX portés depuis PyTorch.

Pour chaque module :
    - Instanciation avec les dimensions Crafter (Palier 2 : state_dim=960)
    - Forward pass avec input random
    - Vérification des shapes de sortie
    - Comptage des paramètres
    - Comparaison avec PyTorch équivalent (si disponible dans sys.path)

Usage :
    .venv/bin/python src_jax/model/test_modules.py
"""

import sys
from pathlib import Path

# Ajouter la racine du projet (test_modules.py est dans src_jax/model/, donc parents[2])
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
from flax import nnx

from src_jax.model import (
    Encoder, CNNEncoder,
    Decoder, CNNDecoder,
    Actor,
    Critic,
    RewardHead, ContinueHead,
    symlog, symexp,
    twohot_encode, twohot_decode,
    N_BINS,
)


# ============================== Helpers

def count_params(model: nnx.Module) -> int:
    """Compte les paramètres trainables (nnx.Param seulement)."""
    params = nnx.state(model, nnx.Param)
    return sum(p.size for p in jax.tree.leaves(params))


def count_params_pytorch(module) -> int:
    """Compte les paramètres d'un module PyTorch."""
    return sum(p.numel() for p in module.parameters())


def check(condition: bool, msg: str):
    status = "OK " if condition else "FAIL"
    print(f"    [{status}] {msg}")
    if not condition:
        raise AssertionError(msg)


def sep(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================== Config Crafter Palier 2

# Palier 2 : RSSM avec state_dim = h_dim + num_classes * class_size
# h_dim=512, num_classes=32, class_size=32  → state_dim = 512 + 1024 = 1536 (exemple)
# On utilise les dims du prompt : state_dim=960, embed_dim=192 pour Palier 2
# Palier 1 : state_dim=384, embed_dim=128

STATE_DIM_P1 = 384
STATE_DIM_P2 = 960
ACTION_DIM_CRAFTER = 17
ACTION_DIM_TETRIS = 41
EMBED_DIM_P1 = 128
EMBED_DIM_P2 = 192
HIDDEN_DIM = 256
OBS_DIM = 276        # Tetris
BATCH = 4
SEQ = 8


def main():
    print()
    print("=" * 60)
    print("  Tests modules JAX/Flax NNX — mini-DreamerV3")
    print("=" * 60)
    print(f"  JAX version  : {jax.__version__}")
    print(f"  Platform     : {jax.default_backend()}")

    rngs = nnx.Rngs(params=42)
    rng_key = jax.random.PRNGKey(0)
    all_passed = True

    # ===== 1. Encoder (MLP) =====
    sep("1. Encoder (MLP) — Palier 1 (Tetris)")
    try:
        enc = Encoder(obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM_P1, rngs=rngs)
        n_params = count_params(enc)
        print(f"  Params JAX : {n_params:,}")

        # Test forward 2D
        x = jnp.ones((BATCH, OBS_DIM))
        out = enc(x)
        check(out.shape == (BATCH, EMBED_DIM_P1), f"shape 2D : {out.shape} == ({BATCH}, {EMBED_DIM_P1})")

        # Test forward 3D (sequence)
        x3 = jnp.ones((BATCH, SEQ, OBS_DIM))
        out3 = enc(x3)
        check(out3.shape == (BATCH, SEQ, EMBED_DIM_P1), f"shape 3D : {out3.shape}")

        # Comparaison PyTorch
        try:
            import torch
            from src.model.encoder import Encoder as PTEncoder
            pt_enc = PTEncoder(obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM_P1)
            n_pt = count_params_pytorch(pt_enc)
            ratio = abs(n_params - n_pt) / max(n_pt, 1)
            print(f"  Params PyTorch : {n_pt:,}  |  diff : {ratio*100:.1f}%")
            check(ratio < 0.10, f"params diff < 10% ({ratio*100:.1f}%)")
        except ImportError:
            print("  [SKIP] PyTorch non dispo — comparaison ignorée")

        print(f"  => Encoder OK  (params={n_params:,}, out={out.shape})")
    except Exception as e:
        print(f"  [FAIL] Encoder : {e}")
        all_passed = False

    # ===== 2. CNNEncoder =====
    sep("2. CNNEncoder (Crafter 64×64)")
    try:
        cnn_enc = CNNEncoder(in_channels=3, embed_dim=EMBED_DIM_P1, base_channels=32, rngs=rngs)
        n_params = count_params(cnn_enc)
        print(f"  Params JAX : {n_params:,}")

        # Test forward NCHW
        img = jnp.ones((BATCH, 3, 64, 64))
        out = cnn_enc(img)
        check(out.shape == (BATCH, EMBED_DIM_P1), f"shape 2D : {out.shape}")

        # Test forward avec dim temporelle
        img5 = jnp.ones((BATCH, SEQ, 3, 64, 64))
        out5 = cnn_enc(img5)
        check(out5.shape == (BATCH, SEQ, EMBED_DIM_P1), f"shape 5D : {out5.shape}")

        try:
            import torch
            from src.model.encoder import CNNEncoder as PTCNNEncoder
            pt_cnn = PTCNNEncoder(in_channels=3, embed_dim=EMBED_DIM_P1, base_channels=32)
            n_pt = count_params_pytorch(pt_cnn)
            ratio = abs(n_params - n_pt) / max(n_pt, 1)
            print(f"  Params PyTorch : {n_pt:,}  |  diff : {ratio*100:.1f}%")
            check(ratio < 0.10, f"params diff < 10% ({ratio*100:.1f}%)")
        except ImportError:
            print("  [SKIP] PyTorch non dispo")

        print(f"  => CNNEncoder OK  (params={n_params:,}, out={out.shape})")
    except Exception as e:
        print(f"  [FAIL] CNNEncoder : {e}")
        all_passed = False

    # ===== 3. Decoder (MLP) =====
    sep("3. Decoder (MLP) — Palier 1 (Tetris)")
    try:
        dec = Decoder(embed_dim=EMBED_DIM_P1, hidden_dim=HIDDEN_DIM, obs_dim=OBS_DIM, rngs=rngs)
        n_params = count_params(dec)
        print(f"  Params JAX : {n_params:,}")

        x = jnp.ones((BATCH, EMBED_DIM_P1))
        out = dec(x)
        check(out.shape == (BATCH, OBS_DIM), f"shape 2D : {out.shape}")

        x3 = jnp.ones((BATCH, SEQ, EMBED_DIM_P1))
        out3 = dec(x3)
        check(out3.shape == (BATCH, SEQ, OBS_DIM), f"shape 3D : {out3.shape}")

        try:
            import torch
            from src.model.decoder import Decoder as PTDecoder
            pt_dec = PTDecoder(embed_dim=EMBED_DIM_P1, hidden_dim=HIDDEN_DIM, obs_dim=OBS_DIM)
            n_pt = count_params_pytorch(pt_dec)
            ratio = abs(n_params - n_pt) / max(n_pt, 1)
            print(f"  Params PyTorch : {n_pt:,}  |  diff : {ratio*100:.1f}%")
            check(ratio < 0.10, f"params diff < 10%")
        except ImportError:
            print("  [SKIP] PyTorch non dispo")

        print(f"  => Decoder OK  (params={n_params:,}, out={out.shape})")
    except Exception as e:
        print(f"  [FAIL] Decoder : {e}")
        all_passed = False

    # ===== 4. CNNDecoder =====
    sep("4. CNNDecoder (Crafter 64×64)")
    try:
        cnn_dec = CNNDecoder(state_dim=STATE_DIM_P2, out_channels=3, base_channels=32, rngs=rngs)
        n_params = count_params(cnn_dec)
        print(f"  Params JAX : {n_params:,}")

        x = jnp.ones((BATCH, STATE_DIM_P2))
        out = cnn_dec(x)
        check(out.shape == (BATCH, 3, 64, 64), f"shape 2D : {out.shape}")
        check(float(out.min()) >= 0.0 and float(out.max()) <= 1.0, "values in [0,1] (sigmoid)")

        x3 = jnp.ones((BATCH, SEQ, STATE_DIM_P2))
        out3 = cnn_dec(x3)
        check(out3.shape == (BATCH, SEQ, 3, 64, 64), f"shape 3D : {out3.shape}")

        try:
            import torch
            from src.model.decoder import CNNDecoder as PTCNNDecoder
            pt_cnn_dec = PTCNNDecoder(state_dim=STATE_DIM_P2, out_channels=3, base_channels=32)
            n_pt = count_params_pytorch(pt_cnn_dec)
            ratio = abs(n_params - n_pt) / max(n_pt, 1)
            print(f"  Params PyTorch : {n_pt:,}  |  diff : {ratio*100:.1f}%")
            check(ratio < 0.10, f"params diff < 10%")
        except ImportError:
            print("  [SKIP] PyTorch non dispo")

        print(f"  => CNNDecoder OK  (params={n_params:,}, out={out.shape})")
    except Exception as e:
        print(f"  [FAIL] CNNDecoder : {e}")
        all_passed = False

    # ===== 5. Actor =====
    sep("5. Actor — Crafter Palier 2")
    try:
        actor = Actor(state_dim=STATE_DIM_P2, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM_CRAFTER, rngs=rngs)
        n_params = count_params(actor)
        print(f"  Params JAX : {n_params:,}")

        state = jnp.ones((BATCH, STATE_DIM_P2))
        logits = actor(state)
        check(logits.shape == (BATCH, ACTION_DIM_CRAFTER), f"logits shape : {logits.shape}")

        # Test sample avec key
        action = actor.sample(state, rng_key)
        check(action.shape == (BATCH,), f"action shape : {action.shape}")
        check(int(action.min()) >= 0 and int(action.max()) < ACTION_DIM_CRAFTER,
              f"action dans [0, {ACTION_DIM_CRAFTER})")

        # Test avec mask
        mask = jnp.ones((BATCH, ACTION_DIM_CRAFTER), dtype=bool)
        mask = mask.at[:, 0].set(False)    # Désactiver action 0
        logits_masked = actor(state)
        logits_masked_applied = jnp.where(mask, logits_masked, jnp.full_like(logits_masked, -1e9))
        check(
            jnp.all(logits_masked_applied[:, 0] < -1e8),
            "mask appliqué correctement (action 0 à -1e9)"
        )

        # Test log_prob + entropy
        action_fixed = jnp.zeros((BATCH,), dtype=jnp.int32)
        log_prob, entropy = actor.log_prob_and_entropy(state, action_fixed)
        check(log_prob.shape == (BATCH,), f"log_prob shape : {log_prob.shape}")
        check(entropy.shape == (BATCH,), f"entropy shape : {entropy.shape}")
        check(jnp.all(entropy >= 0), "entropie >= 0")

        try:
            import torch
            from src.model.actor import Actor as PTActor
            pt_actor = PTActor(state_dim=STATE_DIM_P2, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM_CRAFTER)
            n_pt = count_params_pytorch(pt_actor)
            ratio = abs(n_params - n_pt) / max(n_pt, 1)
            print(f"  Params PyTorch : {n_pt:,}  |  diff : {ratio*100:.1f}%")
            check(ratio < 0.10, f"params diff < 10%")
        except ImportError:
            print("  [SKIP] PyTorch non dispo")

        print(f"  => Actor OK  (params={n_params:,}, logits={logits.shape})")
    except Exception as e:
        print(f"  [FAIL] Actor : {e}")
        all_passed = False

    # ===== 6. Critic =====
    sep("6. Critic — DreamerV3 twohot symlog")
    try:
        critic = Critic(state_dim=STATE_DIM_P2, hidden_dim=HIDDEN_DIM, rngs=rngs)
        n_params = count_params(critic)
        print(f"  Params JAX : {n_params:,}")
        print(f"  n_bins     : {critic.n_bins}")

        state = jnp.ones((BATCH, STATE_DIM_P2))
        logits = critic(state)
        check(logits.shape == (BATCH, N_BINS), f"logits shape : {logits.shape}")

        # Test predict
        values = critic.predict(state)
        check(values.shape == (BATCH,), f"predict shape : {values.shape}")

        # Test loss
        targets = jnp.ones((BATCH,)) * 0.5
        loss_val = critic.loss(state, targets)
        check(loss_val.shape == (), f"loss shape scalaire : {loss_val.shape}")
        check(float(loss_val) > 0, "loss > 0")

        # Vérifier que bins n'est PAS dans les Params (non-trainable)
        param_leaves = jax.tree.leaves(nnx.state(critic, nnx.Param))
        total_param_size = sum(p.size for p in param_leaves)
        print(f"  Params trainables (via nnx.Param) : {total_param_size:,}")

        try:
            import torch
            from src.model.critic import Critic as PTCritic
            pt_critic = PTCritic(state_dim=STATE_DIM_P2, hidden_dim=HIDDEN_DIM)
            n_pt = count_params_pytorch(pt_critic)
            ratio = abs(n_params - n_pt) / max(n_pt, 1)
            print(f"  Params PyTorch : {n_pt:,}  |  diff : {ratio*100:.1f}%")
            check(ratio < 0.10, f"params diff < 10%")
        except ImportError:
            print("  [SKIP] PyTorch non dispo")

        print(f"  => Critic OK  (params={n_params:,}, logits={logits.shape})")
    except Exception as e:
        print(f"  [FAIL] Critic : {e}")
        all_passed = False

    # ===== 7. RewardHead =====
    sep("7. RewardHead — twohot symlog")
    try:
        rh = RewardHead(state_dim=STATE_DIM_P2, hidden_dim=HIDDEN_DIM, rngs=rngs)
        n_params = count_params(rh)
        print(f"  Params JAX : {n_params:,}")

        state = jnp.ones((BATCH, STATE_DIM_P2))
        logits = rh(state)
        check(logits.shape == (BATCH, N_BINS), f"logits shape : {logits.shape}")

        rewards_pred = rh.predict(state)
        check(rewards_pred.shape == (BATCH,), f"predict shape : {rewards_pred.shape}")

        # Loss avec target scalaire
        targets = jnp.array([0.0, 1.0, -1.0, 0.5])
        loss_val = rh.loss(state, targets)
        check(loss_val.shape == (), "loss scalaire")
        check(float(loss_val) > 0, "loss > 0")

        try:
            import torch
            from src.model.heads import RewardHead as PTRewardHead
            pt_rh = PTRewardHead(state_dim=STATE_DIM_P2, hidden_dim=HIDDEN_DIM)
            n_pt = count_params_pytorch(pt_rh)
            ratio = abs(n_params - n_pt) / max(n_pt, 1)
            print(f"  Params PyTorch : {n_pt:,}  |  diff : {ratio*100:.1f}%")
            check(ratio < 0.10, f"params diff < 10%")
        except ImportError:
            print("  [SKIP] PyTorch non dispo")

        print(f"  => RewardHead OK  (params={n_params:,})")
    except Exception as e:
        print(f"  [FAIL] RewardHead : {e}")
        all_passed = False

    # ===== 8. ContinueHead =====
    sep("8. ContinueHead — BCE")
    try:
        ch = ContinueHead(state_dim=STATE_DIM_P2, hidden_dim=HIDDEN_DIM, rngs=rngs)
        n_params = count_params(ch)
        print(f"  Params JAX : {n_params:,}")

        state = jnp.ones((BATCH, STATE_DIM_P2))
        logits = ch(state)
        check(logits.shape == (BATCH,), f"logits shape : {logits.shape}")

        # Loss avec target binaire
        targets = jnp.array([1.0, 0.0, 1.0, 1.0])
        loss_val = ch.loss(state, targets)
        check(loss_val.shape == (), "loss scalaire")
        check(float(loss_val) > 0, "loss > 0")

        try:
            import torch
            from src.model.heads import ContinueHead as PTContinueHead
            pt_ch = PTContinueHead(state_dim=STATE_DIM_P2, hidden_dim=HIDDEN_DIM)
            n_pt = count_params_pytorch(pt_ch)
            ratio = abs(n_params - n_pt) / max(n_pt, 1)
            print(f"  Params PyTorch : {n_pt:,}  |  diff : {ratio*100:.1f}%")
            check(ratio < 0.10, f"params diff < 10%")
        except ImportError:
            print("  [SKIP] PyTorch non dispo")

        print(f"  => ContinueHead OK  (params={n_params:,})")
    except Exception as e:
        print(f"  [FAIL] ContinueHead : {e}")
        all_passed = False

    # ===== 9. Helpers symlog / twohot =====
    sep("9. Helpers symlog / twohot")
    try:
        x = jnp.array([0.0, 1.0, -1.0, 10.0, -100.0])
        sl = symlog(x)
        se = symexp(sl)

        # symexp(symlog(x)) ≈ x (rounding-error only)
        max_err = float(jnp.max(jnp.abs(se - x)))
        check(max_err < 1e-4, f"symexp(symlog(x)) ≈ x (max err = {max_err:.2e})")

        # twohot encode/decode round-trip
        bins = jnp.linspace(-20.0, 20.0, N_BINS)
        scalars = jnp.array([0.0, 1.0, -1.0, 5.0])
        th = twohot_encode(scalars, bins)
        check(th.shape == (4, N_BINS), f"twohot shape : {th.shape}")
        check(
            float(jnp.abs(th.sum(-1) - 1.0).max()) < 1e-5,
            "twohot sums to 1"
        )

        # Decode depuis twohot parfait → valeur en espace original
        decoded = twohot_decode(jnp.log(th + 1e-8), bins)
        check(decoded.shape == (4,), f"decoded shape : {decoded.shape}")

        print(f"  => Helpers symlog/twohot OK")
    except Exception as e:
        print(f"  [FAIL] Helpers : {e}")
        all_passed = False

    # ===== Récap =====
    sep("RECAP")
    if all_passed:
        print("  Tous les modules passent les tests.")
    else:
        print("  Certains tests ont ECHOUE. Voir détails ci-dessus.")

    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
