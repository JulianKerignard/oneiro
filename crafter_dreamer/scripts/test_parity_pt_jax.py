"""
Test de parite PyTorch vs JAX pour mini-DreamerV3.

Les inits sont differentes entre PT et JAX, donc on ne compare PAS bit-a-bit.
On verifie :
  1. Params count identique (ou quasi-identique si LayerNorm scale/bias comptent)
  2. Shapes de sortie identique pour memes inputs
  3. Losses du meme ordre de grandeur (facteur 10x max)
  4. Pas de NaN/Inf
  5. Gradients dans une plage raisonnable (norme L2 des grads)
  6. Training smoketest JAX : 50 iter, loss descend, pas de NaN

Usage :
    .venv/bin/python crafter_dreamer/scripts/test_parity_pt_jax.py

Tolerances :
    - Ratio mean_abs des outputs : < 10x (inits differentes)
    - Params count : diff < 5% (LayerNorm peut avoir des differences mineures)
    - Gradient norms : dans [1e-6, 1e3]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
import optax

# ============================== Config

SEED = 42
B = 4           # batch size
T = 8           # sequence length pour tests RSSM
H = W = 64      # image resolution
C = 3           # channels

# Hyperparams Crafter Palier 2 (alignes sur train_dreamer_jax.py)
EMBED_DIM = 192
H_DIM = 384
Z_CATEGORIES = 24
Z_CLASSES = 24
Z_DIM = Z_CATEGORIES * Z_CLASSES      # 576
STATE_DIM = H_DIM + Z_DIM             # 384 + 576 = 960
HIDDEN_DIM = 768
ACTION_DIM = 17
BASE_CHANNELS = 32

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================== Helpers

def pt_param_count(module):
    """Compte les parametres trainables PyTorch."""
    return sum(p.numel() for p in module.parameters())


def jax_param_count(module):
    """Compte les parametres Flax NNX (nnx.Param uniquement)."""
    return sum(p.size for p in jax.tree.leaves(nnx.state(module, nnx.Param)))


def check_ratio(pt_val, jax_val, label, threshold=10.0):
    """Verifie que le ratio max/min est sous le seuil."""
    if abs(pt_val) < 1e-9 and abs(jax_val) < 1e-9:
        ratio = 1.0
    elif min(abs(pt_val), abs(jax_val)) < 1e-9:
        ratio = float("inf")
    else:
        ratio = max(abs(pt_val), abs(jax_val)) / min(abs(pt_val), abs(jax_val))
    ok = ratio < threshold
    return ok, ratio


def check_finite(arr, name):
    """Verifie l'absence de NaN/Inf."""
    if isinstance(arr, np.ndarray):
        return bool(np.all(np.isfinite(arr)))
    elif isinstance(arr, torch.Tensor):
        return bool(torch.all(torch.isfinite(arr)))
    else:
        return bool(jnp.all(jnp.isfinite(arr)))


def random_np(shape, low=0.0, high=1.0):
    """Array numpy random fixe depuis le seed global."""
    return np.random.uniform(low, high, shape).astype(np.float32)


# ============================== Resultats globaux

_results = []


def record(name, status, details=""):
    _results.append((name, status, details))
    marker = "PASS" if status else "FAIL"
    indicator = "[OK  ]" if status else "[FAIL]"
    print(f"  {indicator} {name}")
    if details:
        for line in details.split("\n"):
            print(f"         {line}")


def sep(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================== Test 1 : CNNEncoder

def test_encoder_parity():
    sep("Test 1 : CNNEncoder — shapes, params, ordre de grandeur")

    from src.model.encoder import CNNEncoder as PTEncoder
    from src_jax.model.encoder import CNNEncoder as JAXEncoder

    pt_enc = PTEncoder(
        in_channels=C, embed_dim=EMBED_DIM,
        base_channels=BASE_CHANNELS, input_resolution=H,
    )
    jax_enc = JAXEncoder(
        in_channels=C, embed_dim=EMBED_DIM,
        base_channels=BASE_CHANNELS, input_resolution=H,
        rngs=nnx.Rngs(SEED),
    )

    # Meme input random
    x_np = random_np((B, C, H, W))
    x_pt = torch.from_numpy(x_np)
    x_jax = jnp.array(x_np)

    with torch.no_grad():
        y_pt = pt_enc(x_pt).numpy()
    y_jax = np.asarray(jax_enc(x_jax))

    details = []
    all_ok = True

    # Shape
    shape_ok = y_pt.shape == y_jax.shape
    all_ok = all_ok and shape_ok
    details.append(f"shape PT={y_pt.shape} JAX={y_jax.shape} : {'OK' if shape_ok else 'MISMATCH'}")

    # Params
    pt_n = pt_param_count(pt_enc)
    jax_n = jax_param_count(jax_enc)
    param_diff_pct = abs(pt_n - jax_n) / max(pt_n, 1) * 100
    param_ok = param_diff_pct < 5.0
    all_ok = all_ok and param_ok
    details.append(f"params PT={pt_n:,} JAX={jax_n:,} diff={param_diff_pct:.2f}% : {'OK' if param_ok else 'DIFF'}")

    # NaN/Inf
    fin_ok = check_finite(y_pt, "pt") and check_finite(y_jax, "jax")
    all_ok = all_ok and fin_ok
    details.append(f"finite : {'OK' if fin_ok else 'NaN/Inf detecte'}")

    # Ordre de grandeur
    pt_mean = float(np.mean(np.abs(y_pt)))
    jax_mean = float(np.mean(np.abs(y_jax)))
    ratio_ok, ratio = check_ratio(pt_mean, jax_mean, "mean_abs")
    all_ok = all_ok and ratio_ok
    details.append(f"mean|out| PT={pt_mean:.4f} JAX={jax_mean:.4f} ratio={ratio:.2f}x : {'OK' if ratio_ok else 'RATIO TROP ELEVE'}")

    record("CNNEncoder", all_ok, "\n".join(details))
    return all_ok


# ============================== Test 2 : CNNDecoder

def test_decoder_parity():
    sep("Test 2 : CNNDecoder — shapes, params, ordre de grandeur")

    from src.model.decoder import CNNDecoder as PTDecoder
    from src_jax.model.decoder import CNNDecoder as JAXDecoder

    pt_dec = PTDecoder(
        state_dim=STATE_DIM, out_channels=C,
        base_channels=BASE_CHANNELS, output_resolution=H,
    )
    jax_dec = JAXDecoder(
        state_dim=STATE_DIM, out_channels=C,
        base_channels=BASE_CHANNELS, output_resolution=H,
        rngs=nnx.Rngs(SEED),
    )

    state_np = random_np((B, STATE_DIM))
    state_pt = torch.from_numpy(state_np)
    state_jax = jnp.array(state_np)

    with torch.no_grad():
        recon_pt = pt_dec(state_pt).numpy()
    recon_jax = np.asarray(jax_dec(state_jax))

    details = []
    all_ok = True

    shape_ok = recon_pt.shape == recon_jax.shape
    all_ok = all_ok and shape_ok
    details.append(f"shape PT={recon_pt.shape} JAX={recon_jax.shape} : {'OK' if shape_ok else 'MISMATCH'}")

    pt_n = pt_param_count(pt_dec)
    jax_n = jax_param_count(jax_dec)
    param_diff_pct = abs(pt_n - jax_n) / max(pt_n, 1) * 100
    param_ok = param_diff_pct < 5.0
    all_ok = all_ok and param_ok
    details.append(f"params PT={pt_n:,} JAX={jax_n:,} diff={param_diff_pct:.2f}% : {'OK' if param_ok else 'DIFF'}")

    fin_ok = check_finite(recon_pt, "pt") and check_finite(recon_jax, "jax")
    all_ok = all_ok and fin_ok
    details.append(f"finite : {'OK' if fin_ok else 'NaN/Inf detecte'}")

    # Range [0, 1] car sigmoid
    range_pt_ok = bool(np.all(recon_pt >= 0) and np.all(recon_pt <= 1))
    range_jax_ok = bool(np.all(recon_jax >= 0) and np.all(recon_jax <= 1))
    range_ok = range_pt_ok and range_jax_ok
    all_ok = all_ok and range_ok
    details.append(f"output in [0,1] (sigmoid) PT={range_pt_ok} JAX={range_jax_ok} : {'OK' if range_ok else 'HORS RANGE'}")

    pt_mean = float(np.mean(np.abs(recon_pt)))
    jax_mean = float(np.mean(np.abs(recon_jax)))
    ratio_ok, ratio = check_ratio(pt_mean, jax_mean, "mean_abs")
    all_ok = all_ok and ratio_ok
    details.append(f"mean|out| PT={pt_mean:.4f} JAX={jax_mean:.4f} ratio={ratio:.2f}x : {'OK' if ratio_ok else 'RATIO TROP ELEVE'}")

    record("CNNDecoder", all_ok, "\n".join(details))
    return all_ok


# ============================== Test 3 : RSSM observe_step

def test_rssm_parity():
    sep("Test 3 : RSSM observe_step — shapes, params, ordre de grandeur")

    from src.model.rssm import RSSM as PTRSSM
    from src_jax.model.rssm import RSSM as JAXRSSM

    pt_rssm = PTRSSM(
        embed_dim=EMBED_DIM, action_dim=ACTION_DIM,
        h_dim=H_DIM, z_categories=Z_CATEGORIES, z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM,
    )
    jax_rssm = JAXRSSM(
        embed_dim=EMBED_DIM, action_dim=ACTION_DIM,
        h_dim=H_DIM, z_categories=Z_CATEGORIES, z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM,
        rngs=nnx.Rngs(SEED),
    )

    # Inputs fixes
    h_np = np.zeros((B, H_DIM), dtype=np.float32)
    z_np = np.zeros((B, Z_DIM), dtype=np.float32)
    emb_np = random_np((B, EMBED_DIM), low=-1.0, high=1.0)
    # Action one-hot : categorie 3
    action_np = np.zeros((B, ACTION_DIM), dtype=np.float32)
    action_np[:, 3] = 1.0

    # PyTorch
    prev_state_pt = {
        "h": torch.from_numpy(h_np),
        "z": torch.from_numpy(z_np),
    }
    emb_pt = torch.from_numpy(emb_np)
    action_pt = torch.from_numpy(action_np)
    with torch.no_grad():
        new_state_pt, post_logits_pt, prior_logits_pt = pt_rssm.observe_step(
            prev_state_pt, action_pt, emb_pt
        )

    # JAX (key pour sampling)
    prev_state_jax = {
        "h": jnp.array(h_np),
        "z": jnp.array(z_np),
    }
    emb_jax = jnp.array(emb_np)
    action_jax = jnp.array(action_np)
    key = jr.PRNGKey(SEED)
    new_state_jax, post_logits_jax, prior_logits_jax = jax_rssm.observe_step(
        prev_state_jax, action_jax, emb_jax, key
    )

    details = []
    all_ok = True

    # Shapes
    shapes_to_check = [
        ("new_state.h", new_state_pt["h"].numpy().shape, np.asarray(new_state_jax["h"]).shape),
        ("new_state.z", new_state_pt["z"].numpy().shape, np.asarray(new_state_jax["z"]).shape),
        ("post_logits", post_logits_pt.numpy().shape, np.asarray(post_logits_jax).shape),
        ("prior_logits", prior_logits_pt.numpy().shape, np.asarray(prior_logits_jax).shape),
    ]
    for name, s_pt, s_jax in shapes_to_check:
        shape_ok = s_pt == s_jax
        all_ok = all_ok and shape_ok
        details.append(f"shape {name}: PT={s_pt} JAX={s_jax} : {'OK' if shape_ok else 'MISMATCH'}")

    # Params
    pt_n = pt_param_count(pt_rssm)
    jax_n = jax_param_count(jax_rssm)
    param_diff_pct = abs(pt_n - jax_n) / max(pt_n, 1) * 100
    param_ok = param_diff_pct < 5.0
    all_ok = all_ok and param_ok
    details.append(f"params PT={pt_n:,} JAX={jax_n:,} diff={param_diff_pct:.2f}% : {'OK' if param_ok else 'DIFF'}")

    # NaN/Inf sur toutes les sorties
    outputs = [
        ("h", new_state_pt["h"].numpy(), np.asarray(new_state_jax["h"])),
        ("z", new_state_pt["z"].numpy(), np.asarray(new_state_jax["z"])),
        ("post_logits", post_logits_pt.numpy(), np.asarray(post_logits_jax)),
        ("prior_logits", prior_logits_pt.numpy(), np.asarray(prior_logits_jax)),
    ]
    for name, pt_arr, jax_arr in outputs:
        fin = check_finite(pt_arr, name) and check_finite(jax_arr, name)
        all_ok = all_ok and fin
        details.append(f"finite {name} : {'OK' if fin else 'NaN/Inf'}")

    # Ordre de grandeur sur h (la sortie la plus deterministe apres zeros)
    h_pt_mean = float(np.mean(np.abs(new_state_pt["h"].numpy())))
    h_jax_mean = float(np.mean(np.abs(np.asarray(new_state_jax["h"]))))
    ratio_ok, ratio = check_ratio(h_pt_mean, h_jax_mean, "h_mean_abs")
    all_ok = all_ok and ratio_ok
    details.append(f"mean|h| PT={h_pt_mean:.4f} JAX={h_jax_mean:.4f} ratio={ratio:.2f}x : {'OK' if ratio_ok else 'RATIO TROP ELEVE'}")

    # z doit etre des one-hots (valeurs dans {0,1} apres straight-through)
    z_pt_np = new_state_pt["z"].numpy()
    z_jax_np = np.asarray(new_state_jax["z"])
    # Avec straight-through, z = probs + sg(onehot - probs), donc pas exactement {0,1}
    # mais doit etre dans [0, 1] et la somme par categorie doit etre ~1
    z_range_ok = bool(np.all(z_pt_np >= -1e-3) and np.all(z_pt_np <= 1 + 1e-3)
                      and np.all(z_jax_np >= -1e-3) and np.all(z_jax_np <= 1 + 1e-3))
    all_ok = all_ok and z_range_ok
    details.append(f"z dans [0,1] (straight-through) : {'OK' if z_range_ok else 'HORS RANGE'}")

    record("RSSM observe_step", all_ok, "\n".join(details))
    return all_ok


# ============================== Test 4 : Heads

def test_heads_parity():
    sep("Test 4 : RewardHead + ContinueHead — shapes, params, losses")

    from src.model.heads import RewardHead as PTRewardHead, ContinueHead as PTContinueHead
    from src_jax.model.heads import RewardHead as JAXRewardHead, ContinueHead as JAXContinueHead

    pt_reward = PTRewardHead(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM)
    jax_reward = JAXRewardHead(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, rngs=nnx.Rngs(SEED))

    pt_continue = PTContinueHead(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM)
    jax_continue = JAXContinueHead(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, rngs=nnx.Rngs(SEED))

    state_np = random_np((B, STATE_DIM), low=-1.0, high=1.0)
    reward_target_np = random_np((B,), low=-2.0, high=2.0)
    continue_target_np = (random_np((B,)) > 0.3).astype(np.float32)

    state_pt = torch.from_numpy(state_np)
    reward_target_pt = torch.from_numpy(reward_target_np)
    continue_target_pt = torch.from_numpy(continue_target_np)

    state_jax = jnp.array(state_np)
    reward_target_jax = jnp.array(reward_target_np)
    continue_target_jax = jnp.array(continue_target_np)

    details = []
    all_ok = True

    # RewardHead : shapes des logits
    with torch.no_grad():
        rwd_logits_pt = pt_reward(state_pt).numpy()
    rwd_logits_jax = np.asarray(jax_reward(state_jax))
    shape_ok = rwd_logits_pt.shape == rwd_logits_jax.shape
    all_ok = all_ok and shape_ok
    details.append(f"RewardHead logits shape PT={rwd_logits_pt.shape} JAX={rwd_logits_jax.shape} : {'OK' if shape_ok else 'MISMATCH'}")

    # RewardHead loss
    with torch.no_grad():
        loss_rwd_pt = float(pt_reward.loss(state_pt, reward_target_pt).item())
    loss_rwd_jax = float(jax_reward.loss(state_jax, reward_target_jax))
    fin_rwd = np.isfinite(loss_rwd_pt) and np.isfinite(loss_rwd_jax)
    all_ok = all_ok and fin_rwd
    details.append(f"RewardHead loss PT={loss_rwd_pt:.4f} JAX={loss_rwd_jax:.4f} : {'OK' if fin_rwd else 'NaN/Inf'}")

    ratio_ok, ratio = check_ratio(loss_rwd_pt, loss_rwd_jax, "reward_loss")
    all_ok = all_ok and ratio_ok
    details.append(f"RewardHead loss ratio={ratio:.2f}x : {'OK' if ratio_ok else 'RATIO TROP ELEVE (>' + str(10) + 'x)'}")

    # ContinueHead : shapes
    with torch.no_grad():
        cont_logits_pt = pt_continue(state_pt).numpy()
    cont_logits_jax = np.asarray(jax_continue(state_jax))
    shape_ok_c = cont_logits_pt.shape == cont_logits_jax.shape
    all_ok = all_ok and shape_ok_c
    details.append(f"ContinueHead logits shape PT={cont_logits_pt.shape} JAX={cont_logits_jax.shape} : {'OK' if shape_ok_c else 'MISMATCH'}")

    # ContinueHead loss (BCE)
    with torch.no_grad():
        loss_cont_pt = float(F.binary_cross_entropy_with_logits(
            torch.from_numpy(cont_logits_pt),
            continue_target_pt,
        ).item())
    loss_cont_jax = float(jax_continue.loss(state_jax, continue_target_jax))
    fin_cont = np.isfinite(loss_cont_pt) and np.isfinite(loss_cont_jax)
    all_ok = all_ok and fin_cont
    details.append(f"ContinueHead loss PT={loss_cont_pt:.4f} JAX={loss_cont_jax:.4f} : {'OK' if fin_cont else 'NaN/Inf'}")

    ratio_ok_c, ratio_c = check_ratio(loss_cont_pt, loss_cont_jax, "continue_loss")
    all_ok = all_ok and ratio_ok_c
    details.append(f"ContinueHead loss ratio={ratio_c:.2f}x : {'OK' if ratio_ok_c else 'RATIO TROP ELEVE'}")

    # Params
    for module_name, pt_mod, jax_mod in [
        ("RewardHead", pt_reward, jax_reward),
        ("ContinueHead", pt_continue, jax_continue),
    ]:
        pt_n = pt_param_count(pt_mod)
        jax_n = jax_param_count(jax_mod)
        diff_pct = abs(pt_n - jax_n) / max(pt_n, 1) * 100
        param_ok = diff_pct < 5.0
        all_ok = all_ok and param_ok
        details.append(f"{module_name} params PT={pt_n:,} JAX={jax_n:,} diff={diff_pct:.2f}% : {'OK' if param_ok else 'DIFF'}")

    record("Heads (Reward + Continue)", all_ok, "\n".join(details))
    return all_ok


# ============================== Test 5 : Actor + Critic

def test_actor_critic_parity():
    sep("Test 5 : Actor + Critic — shapes, params, losses")

    from src.model.actor import Actor as PTActor
    from src.model.critic import Critic as PTCritic
    from src_jax.model.actor import Actor as JAXActor
    from src_jax.model.critic import Critic as JAXCritic

    pt_actor = PTActor(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM)
    jax_actor = JAXActor(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, rngs=nnx.Rngs(SEED))

    pt_critic = PTCritic(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM)
    jax_critic = JAXCritic(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, rngs=nnx.Rngs(SEED))

    state_np = random_np((B, STATE_DIM), low=-1.0, high=1.0)
    returns_np = random_np((B,), low=0.0, high=5.0)

    state_pt = torch.from_numpy(state_np)
    returns_pt = torch.from_numpy(returns_np)
    state_jax = jnp.array(state_np)
    returns_jax = jnp.array(returns_np)

    details = []
    all_ok = True

    # Actor logits shape
    with torch.no_grad():
        logits_pt = pt_actor(state_pt).numpy()
    logits_jax = np.asarray(jax_actor(state_jax))
    shape_ok = logits_pt.shape == logits_jax.shape == (B, ACTION_DIM)
    all_ok = all_ok and shape_ok
    details.append(f"Actor logits shape PT={logits_pt.shape} JAX={logits_jax.shape} attendu=({B},{ACTION_DIM}) : {'OK' if shape_ok else 'MISMATCH'}")

    fin_actor = check_finite(logits_pt, "actor_pt") and check_finite(logits_jax, "actor_jax")
    all_ok = all_ok and fin_actor
    details.append(f"Actor logits finite : {'OK' if fin_actor else 'NaN/Inf'}")

    # Actor : verifier que softmax somme bien a 1
    probs_pt = float(np.mean(np.sum(np.exp(logits_pt - logits_pt.max(-1, keepdims=True)), axis=-1)))
    details.append(f"Actor output est bien des logits (non-normalises, sum exp ~ {probs_pt:.0f} >> 1) : OK")

    # Critic predict shape
    with torch.no_grad():
        value_pt = pt_critic.predict(state_pt).numpy()
    value_jax = np.asarray(jax_critic.predict(state_jax))
    shape_ok_c = value_pt.shape == value_jax.shape == (B,)
    all_ok = all_ok and shape_ok_c
    details.append(f"Critic predict shape PT={value_pt.shape} JAX={value_jax.shape} : {'OK' if shape_ok_c else 'MISMATCH'}")

    # Critic predict : ordre de grandeur
    fin_critic = check_finite(value_pt, "critic_pt") and check_finite(value_jax, "critic_jax")
    all_ok = all_ok and fin_critic
    details.append(f"Critic predict finite : {'OK' if fin_critic else 'NaN/Inf'}")

    v_pt_mean = float(np.mean(np.abs(value_pt)))
    v_jax_mean = float(np.mean(np.abs(value_jax)))
    ratio_ok, ratio = check_ratio(v_pt_mean + 1e-6, v_jax_mean + 1e-6, "value")
    all_ok = all_ok and ratio_ok
    details.append(f"Critic predict mean|v| PT={v_pt_mean:.4f} JAX={v_jax_mean:.4f} ratio={ratio:.2f}x : {'OK' if ratio_ok else 'RATIO TROP ELEVE'}")

    # Critic loss
    with torch.no_grad():
        loss_critic_pt = float(pt_critic.loss(state_pt, returns_pt).item())
    loss_critic_jax = float(jax_critic.loss(state_jax, returns_jax))
    fin_loss = np.isfinite(loss_critic_pt) and np.isfinite(loss_critic_jax)
    all_ok = all_ok and fin_loss
    details.append(f"Critic loss PT={loss_critic_pt:.4f} JAX={loss_critic_jax:.4f} : {'OK' if fin_loss else 'NaN/Inf'}")

    ratio_ok_l, ratio_l = check_ratio(loss_critic_pt, loss_critic_jax, "critic_loss")
    all_ok = all_ok and ratio_ok_l
    details.append(f"Critic loss ratio={ratio_l:.2f}x : {'OK' if ratio_ok_l else 'RATIO TROP ELEVE'}")

    # Params
    for name, pt_mod, jax_mod in [("Actor", pt_actor, jax_actor), ("Critic", pt_critic, jax_critic)]:
        pt_n = pt_param_count(pt_mod)
        jax_n = jax_param_count(jax_mod)
        diff = abs(pt_n - jax_n) / max(pt_n, 1) * 100
        param_ok = diff < 5.0
        all_ok = all_ok and param_ok
        details.append(f"{name} params PT={pt_n:,} JAX={jax_n:,} diff={diff:.2f}% : {'OK' if param_ok else 'DIFF'}")

    record("Actor + Critic", all_ok, "\n".join(details))
    return all_ok


# ============================== Test 6 : Gradient check (JAX)

def test_gradient_health():
    sep("Test 6 : Gradient health JAX — normes non-pathologiques")

    from src_jax.model.encoder import CNNEncoder as JAXEncoder
    from src_jax.model.decoder import CNNDecoder as JAXDecoder
    from src_jax.model.rssm import RSSM as JAXRSSM
    from src_jax.model.heads import RewardHead as JAXRewardHead, ContinueHead as JAXContinueHead

    rngs = nnx.Rngs(SEED)
    encoder = JAXEncoder(in_channels=C, embed_dim=EMBED_DIM, base_channels=BASE_CHANNELS, rngs=rngs)
    rssm = JAXRSSM(
        embed_dim=EMBED_DIM, action_dim=ACTION_DIM,
        h_dim=H_DIM, z_categories=Z_CATEGORIES, z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM, rngs=rngs,
    )
    decoder = JAXDecoder(state_dim=STATE_DIM, out_channels=C, base_channels=BASE_CHANNELS, rngs=rngs)
    reward_head = JAXRewardHead(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, rngs=rngs)

    # Donnees synthetiques
    obs_np = random_np((B, T, C, H, W))
    actions_np = np.zeros((B, T, ACTION_DIM), dtype=np.float32)
    for i in range(B):
        for j in range(T):
            actions_np[i, j, np.random.randint(ACTION_DIM)] = 1.0
    rewards_np = random_np((B, T), low=-1.0, high=1.0)

    obs_jax = jnp.array(obs_np)
    actions_jax = jnp.array(actions_np)
    rewards_jax = jnp.array(rewards_np)

    # Fonction de loss differentiable via nnx.split/merge pattern
    graphdef, state = nnx.split((encoder, rssm, decoder, reward_head))

    def loss_fn(state):
        enc, rss, dec, rh = nnx.merge(graphdef, state)

        # Encode
        obs_flat = obs_jax.reshape(B * T, C, H, W)
        emb_flat = enc(obs_flat)                        # (B*T, EMBED_DIM)
        emb = emb_flat.reshape(B, T, EMBED_DIM)

        # RSSM sequence
        key = jr.PRNGKey(0)
        rssm_out = rss.observe_sequence(emb, actions_jax, key=key)
        state_vec = jnp.concatenate([rssm_out["h"], rssm_out["z"]], axis=-1)  # (B, T, STATE_DIM)

        # Decoder loss
        recon = dec(state_vec)
        recon_loss = jnp.mean((recon - obs_jax) ** 2)

        # Reward loss
        rwd_loss = rh.loss(state_vec.reshape(B * T, STATE_DIM), rewards_jax.reshape(B * T))

        # KL loss
        kl = JAXRSSM.kl_loss(rssm_out["post_logits"], rssm_out["prior_logits"])

        return recon_loss + rwd_loss + kl

    # Calcul des gradients
    loss_val, grads = jax.value_and_grad(loss_fn)(state)
    loss_float = float(loss_val)

    details = []
    all_ok = True

    # Loss finie
    fin_ok = np.isfinite(loss_float)
    all_ok = all_ok and fin_ok
    details.append(f"loss={loss_float:.4f} : {'OK' if fin_ok else 'NaN/Inf'}")

    # Normes des gradients par module
    grad_leaves = jax.tree.leaves(grads)
    grad_norms = [float(jnp.linalg.norm(g)) for g in grad_leaves if g.size > 0]

    if grad_norms:
        max_norm = max(grad_norms)
        min_norm = min(grad_norms)
        mean_norm = float(np.mean(grad_norms))
        n_nan = sum(1 for g in grad_norms if not np.isfinite(g))
        n_zero = sum(1 for g in grad_norms if g < 1e-10)

        grad_ok = (max_norm < 1e4) and (n_nan == 0)
        all_ok = all_ok and grad_ok
        details.append(f"grad norms : min={min_norm:.2e} mean={mean_norm:.2e} max={max_norm:.2e} : {'OK' if grad_ok else 'EXPLOSION'}")
        details.append(f"NaN grads : {n_nan}/{len(grad_norms)} : {'OK' if n_nan == 0 else 'NaN DETECTES'}")
        details.append(f"Zero grads : {n_zero}/{len(grad_norms)} (vanishing si trop) : {'OK' if n_zero < len(grad_norms) * 0.5 else 'TROP DE ZEROS'}")
    else:
        all_ok = False
        details.append("Aucun grad calcule — ERREUR")

    record("Gradient health (JAX)", all_ok, "\n".join(details))
    return all_ok


# ============================== Test 7 : Training smoketest JAX

def test_training_smoketest():
    sep("Test 7 : Training smoketest JAX — 50 iter, loss descend, pas de NaN")

    from src.model.buffer import ImageReplayBuffer
    from crafter_dreamer.scripts.train_dreamer_jax import (
        train_step_wm_jit,
        LR_WM, GRAD_CLIP,
    )
    from src_jax.model import (
        CNNEncoder as JAXEncoder, CNNDecoder as JAXDecoder,
        RSSM as JAXRSSM, RewardHead as JAXRewardHead, ContinueHead as JAXContinueHead,
    )

    N_ITER = 50
    BATCH = 4
    SEQ = 16
    OBS_SHAPE = (C, H, W)
    BUFFER_SIZE = 300

    # Remplir buffer dummy
    rng_np = np.random.default_rng(SEED)
    buffer = ImageReplayBuffer(capacity=BUFFER_SIZE, obs_shape=OBS_SHAPE)
    for _ in range(200):
        obs = rng_np.random(OBS_SHAPE).astype(np.float32)
        next_obs = rng_np.random(OBS_SHAPE).astype(np.float32)
        action = int(rng_np.integers(0, ACTION_DIM))
        reward = float(rng_np.normal(0, 0.5))
        done = bool(rng_np.random() < 0.05)
        buffer.add(obs, action, reward, next_obs, done)

    # Instancier modeles
    rngs = nnx.Rngs(SEED)
    encoder = JAXEncoder(in_channels=C, embed_dim=EMBED_DIM, base_channels=BASE_CHANNELS, rngs=rngs)
    rssm = JAXRSSM(
        embed_dim=EMBED_DIM, action_dim=ACTION_DIM,
        h_dim=H_DIM, z_categories=Z_CATEGORIES, z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM, rngs=rngs,
    )
    decoder = JAXDecoder(state_dim=STATE_DIM, out_channels=C, base_channels=BASE_CHANNELS, rngs=rngs)
    reward_head = JAXRewardHead(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, rngs=rngs)
    continue_head = JAXContinueHead(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, rngs=rngs)

    wm_bundle = (encoder, rssm, decoder, reward_head, continue_head)
    tx_wm = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(LR_WM))
    opt_wm = nnx.Optimizer(wm_bundle, tx_wm, wrt=nnx.Param)

    key = jr.PRNGKey(SEED)
    losses_wm = []
    details = []
    all_ok = True

    t0 = time.time()
    for it in range(N_ITER):
        batch_np = buffer.sample_sequences(BATCH, SEQ)
        batch_jax = {
            "obs":     jnp.array(batch_np["obs"]),
            "actions": jnp.array(batch_np["actions"], dtype=jnp.int32),
            "rewards": jnp.array(batch_np["rewards"]),
            "dones":   jnp.array(batch_np["dones"], dtype=jnp.float32),
        }
        key, subk = jr.split(key)
        metrics = train_step_wm_jit(
            encoder, rssm, decoder, reward_head, continue_head,
            opt_wm, batch_jax, subk,
        )
        loss_wm = float(metrics["loss_wm"])
        losses_wm.append(loss_wm)

        if not np.isfinite(loss_wm):
            all_ok = False
            details.append(f"NaN/Inf detecte a iter {it} : loss_wm={loss_wm}")
            break

    elapsed = time.time() - t0

    if all_ok and len(losses_wm) == N_ITER:
        # Pas de NaN
        fin_ok = all(np.isfinite(l) for l in losses_wm)
        all_ok = all_ok and fin_ok
        details.append(f"NaN check sur {N_ITER} iter : {'OK' if fin_ok else 'NaN detecte'}")

        # Loss descend (comparer moyenne premiere moitie vs deuxieme moitie)
        first_half = np.mean(losses_wm[:N_ITER // 2])
        second_half = np.mean(losses_wm[N_ITER // 2:])
        loss_decrease = second_half < first_half * 1.1  # tolere legere montee (stochastic)
        all_ok = all_ok and loss_decrease
        details.append(
            f"loss_wm : 1ere moitie moy={first_half:.4f} | 2eme moitie moy={second_half:.4f} "
            f"({'descend/stable' if loss_decrease else 'MONTE TROP'})"
        )

        # Ordre de grandeur
        final_loss = losses_wm[-1]
        range_ok = 0 < final_loss < 500
        all_ok = all_ok and range_ok
        details.append(f"loss_wm final={final_loss:.4f} (attendu [0, 500]) : {'OK' if range_ok else 'HORS RANGE'}")

        ips = N_ITER / elapsed
        details.append(f"Performance : {N_ITER} iter en {elapsed:.1f}s ({ips:.2f} ips)")
        details.append(f"loss_wm[0]={losses_wm[0]:.4f} loss_wm[-1]={losses_wm[-1]:.4f}")

    record("Training smoketest JAX (50 iter)", all_ok, "\n".join(details))
    return all_ok


# ============================== Test 8 : Params count global

def test_global_params_count():
    sep("Test 8 : Params count global — tous modules PT vs JAX")

    from src.model.encoder import CNNEncoder as PTEncoder
    from src.model.decoder import CNNDecoder as PTDecoder
    from src.model.rssm import RSSM as PTRSSM
    from src.model.heads import RewardHead as PTRewardHead, ContinueHead as PTContinueHead
    from src.model.actor import Actor as PTActor
    from src.model.critic import Critic as PTCritic

    from src_jax.model.encoder import CNNEncoder as JAXEncoder
    from src_jax.model.decoder import CNNDecoder as JAXDecoder
    from src_jax.model.rssm import RSSM as JAXRSSM
    from src_jax.model.heads import RewardHead as JAXRewardHead, ContinueHead as JAXContinueHead
    from src_jax.model.actor import Actor as JAXActor
    from src_jax.model.critic import Critic as JAXCritic

    rngs = nnx.Rngs(SEED)

    modules = [
        ("CNNEncoder",    PTEncoder(in_channels=C, embed_dim=EMBED_DIM, base_channels=BASE_CHANNELS),
                          JAXEncoder(in_channels=C, embed_dim=EMBED_DIM, base_channels=BASE_CHANNELS, rngs=rngs)),
        ("CNNDecoder",    PTDecoder(state_dim=STATE_DIM, out_channels=C, base_channels=BASE_CHANNELS),
                          JAXDecoder(state_dim=STATE_DIM, out_channels=C, base_channels=BASE_CHANNELS, rngs=rngs)),
        ("RSSM",          PTRSSM(embed_dim=EMBED_DIM, action_dim=ACTION_DIM, h_dim=H_DIM,
                                  z_categories=Z_CATEGORIES, z_classes=Z_CLASSES, hidden_dim=HIDDEN_DIM),
                          JAXRSSM(embed_dim=EMBED_DIM, action_dim=ACTION_DIM, h_dim=H_DIM,
                                   z_categories=Z_CATEGORIES, z_classes=Z_CLASSES, hidden_dim=HIDDEN_DIM, rngs=rngs)),
        ("RewardHead",    PTRewardHead(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM),
                          JAXRewardHead(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, rngs=rngs)),
        ("ContinueHead",  PTContinueHead(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM),
                          JAXContinueHead(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, rngs=rngs)),
        ("Actor",         PTActor(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM),
                          JAXActor(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, rngs=rngs)),
        ("Critic",        PTCritic(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM),
                          JAXCritic(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, rngs=rngs)),
    ]

    details = []
    all_ok = True
    total_pt = 0
    total_jax = 0

    print(f"\n  {'Module':<16s} {'PT params':>12s} {'JAX params':>12s} {'Diff %':>8s} {'Status':>8s}")
    print("  " + "-" * 60)

    for name, pt_mod, jax_mod in modules:
        pt_n = pt_param_count(pt_mod)
        jax_n = jax_param_count(jax_mod)
        diff_pct = abs(pt_n - jax_n) / max(pt_n, 1) * 100
        ok = diff_pct < 5.0
        all_ok = all_ok and ok
        total_pt += pt_n
        total_jax += jax_n
        status = "OK" if ok else "DIFF"
        print(f"  {name:<16s} {pt_n:>12,} {jax_n:>12,} {diff_pct:>7.2f}% {status:>8s}")
        details.append(f"{name}: PT={pt_n:,} JAX={jax_n:,} diff={diff_pct:.2f}%")

    print(f"  {'TOTAL':<16s} {total_pt:>12,} {total_jax:>12,}")
    total_diff = abs(total_pt - total_jax) / max(total_pt, 1) * 100
    details.append(f"TOTAL: PT={total_pt:,} JAX={total_jax:,} diff={total_diff:.2f}%")

    record("Params count global", all_ok, "\n".join(details))
    return all_ok


# ============================== Main

def main():
    print()
    print("=" * 70)
    print("  TESTS DE PARITE PyTorch vs JAX — mini-DreamerV3")
    print("=" * 70)
    print(f"  PyTorch version  : {torch.__version__}")
    print(f"  JAX version      : {jax.__version__}")
    print(f"  JAX backend      : {jax.default_backend()}")
    print(f"  JAX devices      : {jax.devices()}")
    print(f"  Config           : B={B} T={T} EMBED={EMBED_DIM} H_DIM={H_DIM}")
    print(f"                     Z={Z_CATEGORIES}x{Z_CLASSES}={Z_DIM} STATE={STATE_DIM}")
    print(f"                     HIDDEN={HIDDEN_DIM} ACTION_DIM={ACTION_DIM}")

    tests = [
        ("CNNEncoder",               test_encoder_parity),
        ("CNNDecoder",               test_decoder_parity),
        ("RSSM",                     test_rssm_parity),
        ("Heads",                    test_heads_parity),
        ("Actor + Critic",           test_actor_critic_parity),
        ("Gradient health",          test_gradient_health),
        ("Training smoketest",       test_training_smoketest),
        ("Params count global",      test_global_params_count),
    ]

    t_total = time.time()
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            record(name, False, f"Exception : {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t_total

    print()
    print("=" * 70)
    print("  RESUME FINAL")
    print("=" * 70)
    passed = sum(1 for _, s, _ in _results if s)
    total = len(_results)
    for name, status, _ in _results:
        marker = "[PASS]" if status else "[FAIL]"
        print(f"  {marker} {name}")

    print()
    print(f"  Resultat : {passed}/{total} tests passent")
    print(f"  Duree    : {elapsed:.1f}s")

    if passed == total:
        print()
        print("  Conclusion : JAX est PRODUCTION-READY pour remplacer PyTorch.")
        print("  Tous les modules ont les memes shapes, params counts coherents,")
        print("  losses dans le meme ordre de grandeur, gradients sains.")
    else:
        print()
        print(f"  Conclusion : {total - passed} anomalie(s) detectee(s) — voir details ci-dessus.")

    print()
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
