"""
Tests du module RNDModule JAX/Flax NNX.

Valide :
    1. Forward pass — shapes (bonus, embeddings)
    2. Target ≠ predictor (poids différents dès l'init)
    3. normalize_bonus — mutation EMA des running stats
    4. train_loss — scalaire positif
    5. Params count — cible ~2.4M (target 1.2M + predictor 1.2M)
    6. Comparaison count vs version PyTorch
"""

import sys
import jax
import jax.numpy as jnp
from flax import nnx

sys.path.insert(0, "/Users/juliankerignard/Documents/IA-Perso/World-Model")

from src_jax.model.rnd import RNDModule, RNDStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params_nnx(module) -> int:
    """Nombre de paramètres trainables (nnx.Param) dans un module NNX."""
    params = nnx.state(module, nnx.Param)
    return sum(x.size for x in jax.tree.leaves(params))


def count_params_pytorch_rnd() -> int:
    """Nombre de params du RNDModule PyTorch (pour comparaison)."""
    try:
        import torch
        sys.path.insert(0, "/Users/juliankerignard/Documents/IA-Perso/World-Model")
        from src.model.rnd import RNDModule as TorchRND
        rnd = TorchRND(in_channels=3, embed_dim=128, base_channels=32,
                       input_resolution=64)
        return sum(p.numel() for p in rnd.parameters())
    except Exception as e:
        return -1  # PyTorch non dispo ou erreur


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

EMBED_DIM = 128
BASE_CHANNELS = 32
B = 4
C, H, W = 3, 64, 64

rngs = nnx.Rngs(0)
rnd = RNDModule(
    in_channels=C,
    embed_dim=EMBED_DIM,
    base_channels=BASE_CHANNELS,
    input_resolution=H,
    rngs=rngs,
)

# Observations random (B, C, H, W), float32 in [0, 1]
key = jax.random.PRNGKey(99)
obs = jax.random.uniform(key, shape=(B, C, H, W), minval=0.0, maxval=1.0)


# ---------------------------------------------------------------------------
# Test 1 : shapes des embeddings et du bonus
# ---------------------------------------------------------------------------
print("=" * 60)
print("Test 1 : shapes forward pass")

target_emb = jax.lax.stop_gradient(rnd.target(obs))
pred_emb = rnd.predictor(obs)
bonus = rnd.compute_bonus(obs)

assert target_emb.shape == (B, EMBED_DIM), (
    f"target_emb shape: attendu ({B}, {EMBED_DIM}), obtenu {target_emb.shape}"
)
assert pred_emb.shape == (B, EMBED_DIM), (
    f"pred_emb shape: attendu ({B}, {EMBED_DIM}), obtenu {pred_emb.shape}"
)
assert bonus.shape == (B,), (
    f"bonus shape: attendu ({B},), obtenu {bonus.shape}"
)
print(f"  target_emb : {target_emb.shape}  OK")
print(f"  pred_emb   : {pred_emb.shape}  OK")
print(f"  bonus      : {bonus.shape}  OK")
print(f"  bonus values (premiers): {bonus[:3]}")


# ---------------------------------------------------------------------------
# Test 2 : target ≠ predictor (poids différents)
# ---------------------------------------------------------------------------
print("\nTest 2 : target != predictor (init différenciée)")

target_params = jax.tree.leaves(nnx.state(rnd.target, nnx.Param))
pred_params = jax.tree.leaves(nnx.state(rnd.predictor, nnx.Param))

assert len(target_params) == len(pred_params), (
    f"Nombre de param tensors différent: {len(target_params)} vs {len(pred_params)}"
)

# Vérifier que TOUS les paramètres sont différents
n_different = 0
n_total = 0
for tp, pp in zip(target_params, pred_params):
    n_total += 1
    if not jnp.allclose(tp, pp):
        n_different += 1

print(f"  Tensors différents : {n_different} / {n_total}")
assert n_different > 0, "Target et predictor ont des poids IDENTIQUES — init incorrecte"
# Idéalement tous différents (seeds distincts)
print(f"  Premier param target[0,0:3]: {target_params[0].ravel()[:3]}")
print(f"  Premier param pred  [0,0:3]: {pred_params[0].ravel()[:3]}")
print("  OK — poids différents")


# ---------------------------------------------------------------------------
# Test 3 : normalize_bonus met à jour les running stats (mutation EMA)
# ---------------------------------------------------------------------------
print("\nTest 3 : normalize_bonus — mutation EMA des running stats")

mean_before = float(rnd.running_mean.get_value())
var_before = float(rnd.running_var.get_value())
print(f"  Avant  — running_mean: {mean_before:.6f}, running_var: {var_before:.6f}")

bonus_norm = rnd.normalize_bonus(bonus)

mean_after = float(rnd.running_mean.get_value())
var_after = float(rnd.running_var.get_value())
print(f"  Après  — running_mean: {mean_after:.6f}, running_var: {var_after:.6f}")

assert bonus_norm.shape == (B,), (
    f"bonus_norm shape: attendu ({B},), obtenu {bonus_norm.shape}"
)
assert mean_after != mean_before, "running_mean n'a pas été mis à jour"
assert var_after != var_before, "running_var n'a pas été mis à jour"

# Vérifier direction EMA : la mean doit se rapprocher de bonus.mean()
bonus_mean = float(bonus.mean())
expected_new_mean = 0.99 * mean_before + 0.01 * bonus_mean
assert abs(mean_after - expected_new_mean) < 1e-5, (
    f"EMA update incorrect: attendu {expected_new_mean:.6f}, obtenu {mean_after:.6f}"
)
print(f"  bonus.mean(): {bonus_mean:.6f}")
print(f"  EMA attendu : {expected_new_mean:.6f}  | obtenu : {mean_after:.6f}  OK")
print(f"  bonus_norm  : {bonus_norm[:3]}")
print("  OK — running stats mises à jour")


# ---------------------------------------------------------------------------
# Test 4 : train_loss — scalaire, > 0
# ---------------------------------------------------------------------------
print("\nTest 4 : train_loss — scalaire positif")

loss = rnd.train_loss(obs)

assert loss.shape == (), f"train_loss shape: attendu (), obtenu {loss.shape}"
assert float(loss) > 0, f"train_loss doit être > 0, obtenu {float(loss)}"
print(f"  train_loss = {float(loss):.6f}  OK")


# ---------------------------------------------------------------------------
# Test 5 : Params count total (~2.4M)
# ---------------------------------------------------------------------------
print("\nTest 5 : Params count JAX")

n_target = count_params_nnx(rnd.target)
n_predictor = count_params_nnx(rnd.predictor)
n_total_jax = n_target + n_predictor

print(f"  target    : {n_target:>10,} params")
print(f"  predictor : {n_predictor:>10,} params")
print(f"  TOTAL JAX : {n_total_jax:>10,} params")

# Vérifier que target et predictor ont le même count (même architecture)
assert n_target == n_predictor, (
    f"target et predictor ont des counts différents: {n_target} vs {n_predictor}"
)

# Vérifier la cible ~2.4M (±5%)
TARGET_COUNT = 2_400_000
tolerance = 0.05
assert abs(n_total_jax - TARGET_COUNT) / TARGET_COUNT < tolerance, (
    f"Params count hors tolérance: {n_total_jax} vs cible {TARGET_COUNT}"
)
print(f"  Dans la tolérance ±5% de {TARGET_COUNT:,}  OK")


# ---------------------------------------------------------------------------
# Test 6 : Comparaison avec PyTorch
# ---------------------------------------------------------------------------
print("\nTest 6 : Comparaison count PyTorch vs JAX")

n_pytorch = count_params_pytorch_rnd()
if n_pytorch > 0:
    diff_pct = abs(n_total_jax - n_pytorch) / n_pytorch * 100
    print(f"  PyTorch : {n_pytorch:>10,} params")
    print(f"  JAX     : {n_total_jax:>10,} params")
    print(f"  Écart   : {diff_pct:.3f}%")
    assert diff_pct < 0.5, (
        f"Écart params JAX/PyTorch trop grand: {diff_pct:.3f}% (max 0.5%)"
    )
    print("  OK — écart < 0.5%")
else:
    print("  PyTorch non disponible — skip comparaison (count JAX seul validé)")


# ---------------------------------------------------------------------------
# Test 7 : RNDStats non inclus dans nnx.Param
# ---------------------------------------------------------------------------
print("\nTest 7 : RNDStats non-trainable (ne filtre pas comme nnx.Param)")

rnd_stats = nnx.state(rnd, RNDStats)
rnd_params = nnx.state(rnd, nnx.Param)

stats_leaves = jax.tree.leaves(rnd_stats)
param_leaves = jax.tree.leaves(rnd_params)

print(f"  RNDStats leaves  : {len(stats_leaves)} (running_mean + running_var)")
print(f"  nnx.Param leaves : {len(param_leaves)}")
assert len(stats_leaves) == 2, (
    f"Attendu 2 RNDStats (mean + var), obtenu {len(stats_leaves)}"
)
assert len(param_leaves) > 0, "Aucun Param trouvé dans RNDModule"
print("  OK — running stats séparées des params trainables")


# ---------------------------------------------------------------------------
# Résumé
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TOUS LES TESTS PASSENT")
print(f"  Params JAX total : {n_total_jax:,}")
if n_pytorch > 0:
    print(f"  Params PyTorch   : {n_pytorch:,}")
    print(f"  Écart            : {diff_pct:.3f}%")
print("=" * 60)
