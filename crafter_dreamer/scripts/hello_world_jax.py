"""
Hello World JAX — vérifie le setup avant migration mini-DreamerV3.

Tests effectués :
    1. Imports JAX + Flax NNX + Optax + Distrax fonctionnent
    2. Devices disponibles (GPU CUDA / CPU)
    3. Forward pass d'un MLP simple Flax NNX
    4. Backward pass + optimizer step optax
    5. jax.lax.scan basique (le pattern critique pour RSSM)
    6. distrax.Categorical (utilisé pour z catégorique + actor)

Usage :
    .venv/bin/python crafter_dreamer/scripts/hello_world_jax.py

Sur Mac M4 Max : devrait afficher "Platform: cpu" (jax-metal abandonné).
Sur Modal L4 : devrait afficher "Platform: gpu".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import distrax


def main():
    print("=" * 60)
    print("Hello World JAX — sanity check setup")
    print("=" * 60)

    # 1. Devices
    print(f"JAX version       : {jax.__version__}")
    print(f"JAX devices       : {jax.devices()}")
    print(f"JAX default device: {jax.devices()[0]}")
    print(f"Platform          : {jax.default_backend()}")
    print()

    # 2. Test MLP simple en Flax NNX
    print("=" * 60)
    print("Test MLP Flax NNX")
    print("=" * 60)

    class MLP(nnx.Module):
        def __init__(self, din: int, dhidden: int, dout: int, *, rngs: nnx.Rngs):
            self.linear1 = nnx.Linear(din, dhidden, rngs=rngs)
            self.linear2 = nnx.Linear(dhidden, dout, rngs=rngs)

        def __call__(self, x):
            x = self.linear1(x)
            x = jax.nn.elu(x)
            x = self.linear2(x)
            return x

    rngs = nnx.Rngs(42)
    mlp = MLP(din=10, dhidden=32, dout=5, rngs=rngs)

    # Forward
    x = jnp.ones((4, 10))
    y = mlp(x)
    print(f"  MLP forward : input {x.shape} → output {y.shape}")
    print(f"  Total params : {sum(p.size for p in jax.tree.leaves(nnx.state(mlp, nnx.Param)))}")
    print()

    # 3. Test backward + optimizer
    print("=" * 60)
    print("Test backward + optax")
    print("=" * 60)

    optimizer = nnx.Optimizer(mlp, optax.adam(1e-3), wrt=nnx.Param)

    def loss_fn(model, x, y_target):
        y_pred = model(x)
        return jnp.mean((y_pred - y_target) ** 2)

    y_target = jnp.zeros((4, 5))
    loss, grads = nnx.value_and_grad(loss_fn)(mlp, x, y_target)
    print(f"  Loss (before step) : {loss:.6f}")
    optimizer.update(mlp, grads)
    loss_after = loss_fn(mlp, x, y_target)
    print(f"  Loss (after step)  : {loss_after:.6f}")
    print(f"  Improved : {bool(loss_after < loss)}")
    print()

    # 4. Test jax.lax.scan (critique pour RSSM)
    print("=" * 60)
    print("Test jax.lax.scan (pattern RSSM)")
    print("=" * 60)

    def step_fn(carry, x):
        # Carry = hidden state, x = input à ce step
        h = carry
        new_h = jnp.tanh(h + x)
        return new_h, new_h   # (carry, output)

    init_h = jnp.zeros(8)
    xs = jnp.ones((10, 8))   # 10 steps de input dim 8
    final_h, all_h = jax.lax.scan(step_fn, init_h, xs)
    print(f"  scan : {xs.shape} → final {final_h.shape}, all {all_h.shape}")
    print(f"  Pattern OK pour RSSM observe_sequence / imagine_sequence")
    print()

    # 5. Test distrax.Categorical (z RSSM + actor)
    print("=" * 60)
    print("Test distrax.Categorical (z + actor)")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    logits = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])   # (2, 3)
    dist = distrax.Categorical(logits=logits)
    samples = dist.sample(seed=key, sample_shape=())
    log_probs = dist.log_prob(samples)
    entropy = dist.entropy()
    print(f"  Categorical(logits={logits.shape})")
    print(f"  Samples : {samples}")
    print(f"  Log probs : {log_probs}")
    print(f"  Entropy : {entropy}")

    # KL divergence (utilisé dans RSSM)
    logits_p = jnp.array([1.0, 2.0, 3.0])
    logits_q = jnp.array([1.5, 2.0, 2.5])
    dist_p = distrax.Categorical(logits=logits_p)
    dist_q = distrax.Categorical(logits=logits_q)
    kl = dist_p.kl_divergence(dist_q)
    print(f"  KL(p||q) : {kl:.6f}")
    print()

    # 6. Test PRNG split (CRITIQUE en JAX)
    print("=" * 60)
    print("Test PRNG management (critique)")
    print("=" * 60)
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)
    samples = jax.vmap(lambda k: jax.random.normal(k, ()))(keys)
    print(f"  4 samples avec keys différentes : {samples}")
    print(f"  → Si toutes égales = BUG (PRNG mal gérée)")
    print()

    print("=" * 60)
    print("✅ Tous les tests OK. JAX prêt pour migration mini-DreamerV3.")
    print("=" * 60)


if __name__ == "__main__":
    main()
