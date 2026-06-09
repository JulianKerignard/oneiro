"""
Test de parité numérique RIGOUREUX entre PyTorch et JAX.

Charge weights PyTorch → copie dans JAX → compare outputs avec mêmes inputs.

Si les outputs divergent : on a un BUG d'implémentation quelque part.

Conventions à mapper (sources fréquentes de bugs) :
    - PyTorch nn.Linear(in, out).weight : shape (out, in)
      Flax nnx.Linear(in, out).kernel    : shape (in, out)              → TRANSPOSE
    - PyTorch nn.Conv2d(in, out, k).weight : shape (out, in, kH, kW)
      Flax nnx.Conv(in, out, k).kernel     : shape (kH, kW, in, out)    → TRANSPOSE
    - PyTorch nn.LayerNorm.weight/bias     : shape (D,)
      Flax nnx.LayerNorm.scale/bias        : shape (D,)                 → identique
    - PyTorch nn.GRUCell.weight_ih/hh      : shape (3*h, in)/(3*h, h), bias_ih/hh : (3*h,)
      Flax nnx.GRUCell.dense_i/dense_h.kernel : (in, 3*h)/(h, 3*h),     → TRANSPOSE
                                              .bias : (3*h,)            → identique
      Gate order identique : (r, z, n).

Usage :
    .venv/bin/python crafter_dreamer/scripts/test_numerical_parity.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import nnx

# PyTorch modules
from src.model.encoder import CNNEncoder as PT_CNNEncoder
from src.model.rssm import RSSM as PT_RSSM
from src.model.actor import Actor as PT_Actor
from src.model.critic import Critic as PT_Critic

# JAX modules
from src_jax.model.encoder import CNNEncoder as JX_CNNEncoder
from src_jax.model.rssm import RSSM as JX_RSSM
from src_jax.model.actor import Actor as JX_Actor
from src_jax.model.critic import Critic as JX_Critic


# Hyperparams (mêmes que train_dreamer_jax.py Crafter Palier 2)
EMBED_DIM = 192
H_DIM = 384
Z_CATEGORIES = 24
Z_CLASSES = 24
HIDDEN_DIM = 768
ACTION_DIM = 17

# Seuils de parité
TOL = 1e-4   # max abs diff tolérée

# ============================================================== helpers


def _to_jax(arr_pt: torch.Tensor) -> jnp.ndarray:
    """Tensor PyTorch (CPU) → jnp.ndarray (float32)."""
    return jnp.asarray(arr_pt.detach().cpu().numpy(), dtype=jnp.float32)


def _set_linear(jx_linear: nnx.Linear, pt_weight: torch.Tensor, pt_bias: torch.Tensor):
    """PyTorch nn.Linear weight (out, in) → Flax nnx.Linear kernel (in, out)."""
    jx_linear.kernel.value = _to_jax(pt_weight.T)
    jx_linear.bias.value = _to_jax(pt_bias)


def _set_conv(jx_conv: nnx.Conv, pt_weight: torch.Tensor, pt_bias: torch.Tensor):
    """PyTorch Conv2d weight (out, in, kH, kW) → Flax Conv kernel (kH, kW, in, out)."""
    # (out, in, kH, kW) → (kH, kW, in, out)
    w = pt_weight.permute(2, 3, 1, 0).contiguous()
    jx_conv.kernel.value = _to_jax(w)
    jx_conv.bias.value = _to_jax(pt_bias)


def _set_layernorm(jx_ln: nnx.LayerNorm, pt_weight: torch.Tensor, pt_bias: torch.Tensor):
    """PyTorch LayerNorm (weight, bias) → Flax LayerNorm (scale, bias)."""
    jx_ln.scale.value = _to_jax(pt_weight)
    jx_ln.bias.value = _to_jax(pt_bias)


def _set_grucell(jx_gru, pt_gru: torch.nn.GRUCell):
    """PyTorch GRUCell → CustomGRUCell (src_jax/model/rssm.py).

    PyTorch GRUCell :
        weight_ih (3*h, in), weight_hh (3*h, h)   — gate order (r, z, n)
        bias_ih (3*h),       bias_hh (3*h)        — DEUX biases séparés

    CustomGRUCell (matche PyTorch exactement) :
        dense_i.kernel (in, 3*h), dense_i.bias (3*h)  ← bias_ih
        dense_h.kernel (h,  3*h), dense_h.bias (3*h)  ← bias_hh  (USE_BIAS=TRUE)
        Gate order (r, z, n) — identique à PyTorch.
        Formule n : tanh(W_in x + b_in + r * (W_hn h + b_hn)) — identique à PyTorch.
    """
    sd = pt_gru.state_dict()
    # weight_ih : (3*h, in) → transpose → (in, 3*h)
    jx_gru.dense_i.kernel.value = _to_jax(sd["weight_ih"].T)
    # weight_hh : (3*h, h) → transpose → (h, 3*h)
    jx_gru.dense_h.kernel.value = _to_jax(sd["weight_hh"].T)
    # Bias SEPARES : on copie chacun dans son dense respectif (plus de merge).
    jx_gru.dense_i.bias.value = _to_jax(sd["bias_ih"])
    jx_gru.dense_h.bias.value = _to_jax(sd["bias_hh"])


def _check_layernorm_eps(pt_ln: torch.nn.LayerNorm, jx_ln: nnx.LayerNorm, name: str):
    """Vérifie que les epsilon LayerNorm sont alignés (sinon mini-diff)."""
    pt_eps = pt_ln.eps
    # Flax NNX LayerNorm stocke epsilon dans .epsilon
    jx_eps = jx_ln.epsilon if hasattr(jx_ln, "epsilon") else None
    if jx_eps is not None and abs(pt_eps - jx_eps) > 1e-10:
        print(f"  [warn] {name}: LayerNorm eps PT={pt_eps} vs JX={jx_eps}")


def _compare(name: str, y_pt: np.ndarray, y_jax: np.ndarray) -> bool:
    abs_diff = np.abs(y_pt - y_jax)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    ok = max_diff < TOL
    marker = "OK" if ok else "FAIL"
    print(f"  [{marker}] {name:20s} | shape PT={y_pt.shape} JX={y_jax.shape}"
          f" | max={max_diff:.3e} mean={mean_diff:.3e}")
    return ok


# ============================================================== Encoder


def test_encoder_parity() -> bool:
    print("\n=== Encoder (CNNEncoder)")
    torch.manual_seed(42)
    pt = PT_CNNEncoder(in_channels=3, embed_dim=EMBED_DIM, base_channels=32,
                       input_resolution=64)
    pt.eval()

    jx = JX_CNNEncoder(in_channels=3, embed_dim=EMBED_DIM, base_channels=32,
                       input_resolution=64, rngs=nnx.Rngs(0))

    # Copy weights
    _set_conv(jx.conv1, pt.conv[0].weight, pt.conv[0].bias)
    _set_conv(jx.conv2, pt.conv[2].weight, pt.conv[2].bias)
    _set_conv(jx.conv3, pt.conv[4].weight, pt.conv[4].bias)
    _set_conv(jx.conv4, pt.conv[6].weight, pt.conv[6].bias)
    _set_linear(jx.proj_linear, pt.proj[0].weight, pt.proj[0].bias)
    _set_layernorm(jx.proj_norm, pt.proj[1].weight, pt.proj[1].bias)

    _check_layernorm_eps(pt.proj[1], jx.proj_norm, "proj_norm")

    # Same input
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((4, 3, 64, 64)).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_jax = jnp.asarray(x_np)

    with torch.no_grad():
        y_pt = pt(x_pt).numpy()
    y_jax = np.asarray(jx(x_jax))

    # Test intermédiaire : sortie de conv1 seule (avant ELU)
    with torch.no_grad():
        # conv1 only
        y1_pt = pt.conv[0](x_pt).numpy()  # NCHW
    y1_jax = np.asarray(jx.conv1(jnp.transpose(x_jax, (0, 2, 3, 1))))
    # Transposer JAX NHWC → NCHW pour compare
    y1_jax_nchw = np.transpose(y1_jax, (0, 3, 1, 2))
    _compare("conv1 only", y1_pt, y1_jax_nchw)

    # ⚠️ BUG SUSPECTED : ordre du flatten
    # PyTorch (NCHW) reshape(B, -1) → ordre C-H-W
    # JAX (NHWC)     reshape(B, -1) → ordre H-W-C
    # Les éléments aplatis sont dans un ordre DIFFÉRENT.
    # Donc proj_linear voit des features dans un ordre différent → divergence.
    import torch.nn.functional as F
    with torch.no_grad():
        y4_pt = F.elu(pt.conv[6](F.elu(pt.conv[4](F.elu(pt.conv[2](F.elu(pt.conv[0](x_pt))))))))
        flat_pt = y4_pt.reshape(y4_pt.shape[0], -1).numpy()
    y1j = jax.nn.elu(jx.conv1(jnp.transpose(x_jax, (0, 2, 3, 1))))
    y2j = jax.nn.elu(jx.conv2(y1j))
    y3j = jax.nn.elu(jx.conv3(y2j))
    y4j = jax.nn.elu(jx.conv4(y3j))
    flat_jx_raw = np.asarray(y4j.reshape(y4j.shape[0], -1))
    # Si on transpose NHWC → NCHW AVANT flatten, on devrait recoller PyTorch :
    flat_jx_nchw = np.asarray(jnp.transpose(y4j, (0, 3, 1, 2)).reshape(y4j.shape[0], -1))
    _compare("flatten raw  ", flat_pt, flat_jx_raw)
    _compare("flatten NCHW ", flat_pt, flat_jx_nchw)

    return _compare("encoder full", y_pt, y_jax)


# ============================================================== RSSM


def test_rssm_parity() -> bool:
    print("\n=== RSSM (observe_step)")
    torch.manual_seed(43)
    pt = PT_RSSM(
        embed_dim=EMBED_DIM,
        action_dim=ACTION_DIM,
        h_dim=H_DIM,
        z_categories=Z_CATEGORIES,
        z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM,
    )
    pt.eval()

    jx = JX_RSSM(
        embed_dim=EMBED_DIM,
        action_dim=ACTION_DIM,
        h_dim=H_DIM,
        z_categories=Z_CATEGORIES,
        z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM,
        rngs=nnx.Rngs(0),
    )

    # ---- Copy weights ----
    # pre_gru = Sequential[Linear, LayerNorm, ELU]
    _set_linear(jx.pre_gru_linear, pt.pre_gru[0].weight, pt.pre_gru[0].bias)
    _set_layernorm(jx.pre_gru_norm, pt.pre_gru[1].weight, pt.pre_gru[1].bias)

    # GRU
    _set_grucell(jx.gru, pt.gru)

    # prior_net = Sequential[Linear, LayerNorm, ELU, Linear]
    _set_linear(jx.prior_linear1, pt.prior_net[0].weight, pt.prior_net[0].bias)
    _set_layernorm(jx.prior_norm, pt.prior_net[1].weight, pt.prior_net[1].bias)
    _set_linear(jx.prior_linear2, pt.prior_net[3].weight, pt.prior_net[3].bias)

    # posterior_net = Sequential[Linear, LayerNorm, ELU, Linear]
    _set_linear(jx.post_linear1, pt.posterior_net[0].weight, pt.posterior_net[0].bias)
    _set_layernorm(jx.post_norm, pt.posterior_net[1].weight, pt.posterior_net[1].bias)
    _set_linear(jx.post_linear2, pt.posterior_net[3].weight, pt.posterior_net[3].bias)

    # ---- Same inputs ----
    B = 4
    rng = np.random.default_rng(1)
    h_np = rng.standard_normal((B, H_DIM)).astype(np.float32)
    z_np = rng.standard_normal((B, Z_CATEGORIES * Z_CLASSES)).astype(np.float32)
    a_np = rng.standard_normal((B, ACTION_DIM)).astype(np.float32)
    emb_np = rng.standard_normal((B, EMBED_DIM)).astype(np.float32)

    prev_state_pt = {"h": torch.from_numpy(h_np), "z": torch.from_numpy(z_np)}
    a_pt = torch.from_numpy(a_np)
    emb_pt = torch.from_numpy(emb_np)

    prev_state_jx = {"h": jnp.asarray(h_np), "z": jnp.asarray(z_np)}
    a_jx = jnp.asarray(a_np)
    emb_jx = jnp.asarray(emb_np)

    # ---- Intermédiaire : pre_gru + GRU → h ----
    with torch.no_grad():
        gru_in_raw_pt = torch.cat([prev_state_pt["z"], a_pt], dim=-1)
        gru_in_pt = pt.pre_gru(gru_in_raw_pt)
        h_pt = pt.gru(gru_in_pt, prev_state_pt["h"]).numpy()

    gru_in_raw_jx = jnp.concatenate([prev_state_jx["z"], a_jx], axis=-1)
    gru_in_jx = jx._pre_gru(gru_in_raw_jx)
    h_jx, _ = jx.gru(prev_state_jx["h"], gru_in_jx)
    h_jx_np = np.asarray(h_jx)

    _compare("pre_gru output", np.asarray(gru_in_pt), np.asarray(gru_in_jx))
    ok_gru = _compare("GRU new_h",       h_pt, h_jx_np)

    # ---- Intermédiaire : prior_net(h) ----
    with torch.no_grad():
        prior_pt = pt.prior_net(torch.from_numpy(h_pt)).numpy()
    prior_jx = np.asarray(jx._prior_net(jnp.asarray(h_pt)))
    _compare("prior_net logits", prior_pt, prior_jx)

    # ---- Intermédiaire : posterior_net((h, embedding)) ----
    with torch.no_grad():
        post_in = torch.cat([torch.from_numpy(h_pt), emb_pt], dim=-1)
        post_pt = pt.posterior_net(post_in).numpy()
    post_in_jx = jnp.concatenate([jnp.asarray(h_pt), emb_jx], axis=-1)
    post_jx = np.asarray(jx._posterior_net(post_in_jx))
    ok_post = _compare("posterior_net logits", post_pt, post_jx)

    return ok_gru and ok_post


# ============================================================== Actor


def test_actor_parity() -> bool:
    print("\n=== Actor")
    state_dim = H_DIM + Z_CATEGORIES * Z_CLASSES
    torch.manual_seed(44)
    pt = PT_Actor(state_dim=state_dim, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM)
    pt.eval()

    jx = JX_Actor(state_dim=state_dim, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM,
                  rngs=nnx.Rngs(0))

    # Copy weights : net = Seq[Linear, LayerNorm, ELU, Linear, LayerNorm, ELU, Linear]
    _set_linear(jx.linear1, pt.net[0].weight, pt.net[0].bias)
    _set_layernorm(jx.norm1, pt.net[1].weight, pt.net[1].bias)
    _set_linear(jx.linear2, pt.net[3].weight, pt.net[3].bias)
    _set_layernorm(jx.norm2, pt.net[4].weight, pt.net[4].bias)
    _set_linear(jx.out,     pt.net[6].weight, pt.net[6].bias)

    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((8, state_dim)).astype(np.float32)
    with torch.no_grad():
        y_pt = pt(torch.from_numpy(x_np)).numpy()
    y_jax = np.asarray(jx(jnp.asarray(x_np)))
    return _compare("actor logits", y_pt, y_jax)


# ============================================================== Critic


def test_critic_parity() -> bool:
    print("\n=== Critic")
    state_dim = H_DIM + Z_CATEGORIES * Z_CLASSES
    torch.manual_seed(45)
    pt = PT_Critic(state_dim=state_dim, hidden_dim=HIDDEN_DIM)
    pt.eval()

    jx = JX_Critic(state_dim=state_dim, hidden_dim=HIDDEN_DIM, rngs=nnx.Rngs(0))

    # net = Seq[Linear, LayerNorm, ELU, Linear, LayerNorm, ELU, Linear]
    _set_linear(jx.linear1, pt.net[0].weight, pt.net[0].bias)
    _set_layernorm(jx.norm1, pt.net[1].weight, pt.net[1].bias)
    _set_linear(jx.linear2, pt.net[3].weight, pt.net[3].bias)
    _set_layernorm(jx.norm2, pt.net[4].weight, pt.net[4].bias)
    _set_linear(jx.out,     pt.net[6].weight, pt.net[6].bias)

    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((8, state_dim)).astype(np.float32)
    with torch.no_grad():
        logits_pt = pt(torch.from_numpy(x_np)).numpy()
        val_pt = pt.predict(torch.from_numpy(x_np)).numpy()
    logits_jx = np.asarray(jx(jnp.asarray(x_np)))
    val_jx = np.asarray(jx.predict(jnp.asarray(x_np)))

    ok1 = _compare("critic logits", logits_pt, logits_jx)
    ok2 = _compare("critic predict", val_pt, val_jx)
    return ok1 and ok2


# ============================================================== main


def main():
    print("=" * 64)
    print("TEST DE PARITE NUMERIQUE PyTorch vs JAX")
    print(f"  Tolerance max_abs_diff < {TOL:.0e}")
    print(f"  Hyperparams : embed={EMBED_DIM} h={H_DIM} z={Z_CATEGORIES}x{Z_CLASSES}"
          f" hidden={HIDDEN_DIM} action={ACTION_DIM}")
    print("=" * 64)

    results = []
    for name, test_fn in [
        ("Encoder", test_encoder_parity),
        ("RSSM",    test_rssm_parity),
        ("Actor",   test_actor_parity),
        ("Critic",  test_critic_parity),
    ]:
        try:
            ok = test_fn()
        except Exception as e:
            print(f"\n[EXCEPTION] {name}: {e}")
            import traceback
            traceback.print_exc()
            ok = False
        results.append((name, ok))

    print("\n" + "=" * 64)
    print("RESUME")
    print("=" * 64)
    for name, ok in results:
        marker = "OK  " if ok else "FAIL"
        print(f"  [{marker}] {name}")
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n  {n_ok}/{len(results)} modules en parite numerique.")

    print("\n" + "=" * 64)
    print("BUGS IDENTIFIES")
    print("=" * 64)
    print("""
  1. ENCODER — Flatten order incompatible (BUG MAJEUR)
     - PyTorch : conv4 output (B, C=256, H=4, W=4) -> reshape(B, -1) = ordre C-H-W
     - JAX     : conv4 output (B, H=4, W=4, C=256) -> reshape(B, -1) = ordre H-W-C
     - Le LinearProj voit donc des features dans un ordre DIFFERENT.
     - FIX : transposer NHWC -> NCHW AVANT flatten dans src_jax/model/encoder.py
       x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW pour matcher PyTorch
       x = x.reshape(x.shape[0], -1)

  2. RSSM GRU — bias_hh manquant (BUG STRUCTUREL)
     - PyTorch nn.GRUCell : bias_ih + bias_hh (deux biases additionnes dans le gate)
     - Flax NNX GRUCell   : dense_h.use_bias=False -> SEUL bias_ih existe
     - Resultat : Flax calcule un GRU MATHEMATIQUEMENT different.
     - FIX : reimplementer une GRUCell custom (avec deux biases) OU s'assurer que
       l'init PyTorch pose bias_hh=0 (mais ce n'est pas le default).

  3. LayerNorm epsilon (mineur)
     - PyTorch : 1e-5
     - Flax    : 1e-6
     - Impact petit mais cumulable. Aligner via nnx.LayerNorm(..., epsilon=1e-5).
""")


if __name__ == "__main__":
    main()
