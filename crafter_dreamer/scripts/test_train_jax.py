"""
Test rapide de train_dreamer_jax.py.

Vérifie :
  - Les modules JAX se chargent sans erreur.
  - compute_lambda_returns produit les bonnes shapes.
  - imagine_trajectory produit les bonnes shapes.
  - train_step_wm + train_step_ac tournent sur 10 iter avec un buffer dummy.
  - Pas de NaN/Inf dans les losses, valeurs raisonnables.
  - Le slow critic se met à jour via Polyak.

Lance : .venv/bin/python crafter_dreamer/scripts/test_train_jax.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
import optax

from src.model.buffer import ImageReplayBuffer
from src_jax.model import (
    CNNEncoder, CNNDecoder, RSSM, RewardHead, ContinueHead,
    Actor, Critic,
)
from crafter_dreamer.scripts.train_dreamer_jax import (
    compute_lambda_returns,
    imagine_trajectory,
    train_step_wm_jit,
    train_step_ac_jit,
    EMBED_DIM, H_DIM, Z_CATEGORIES, Z_CLASSES, HIDDEN_DIM,
    LR_WM, LR_AC, GRAD_CLIP, ENTROPY_COEF,
    SEQ_LEN, IMAGINATION_HORIZON,
)


# ============================== Config test (mini)

OBS_SHAPE = (3, 64, 64)
ACTION_DIM = 17
BUFFER_CAPACITY = 200
N_TRANSITIONS_INIT = 100
BATCH_SIZE_TEST = 4
SEQ_LEN_TEST = 16
TEST_HORIZON = 8
N_ITER_TEST = 10


# ============================== Helpers test


def check(condition, msg, fatal=True):
    status = "[OK  ]" if condition else "[FAIL]"
    print(f"  {status} {msg}")
    if not condition and fatal:
        raise AssertionError(msg)


def sep(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def fill_dummy_buffer(buffer, n, rng):
    """Remplit le buffer avec n transitions random."""
    for _ in range(n):
        obs = rng.random(OBS_SHAPE, dtype=np.float32)
        next_obs = rng.random(OBS_SHAPE, dtype=np.float32)
        action = rng.integers(0, ACTION_DIM)
        reward = float(rng.normal(0, 0.5))
        done = bool(rng.random() < 0.05)
        buffer.add(obs, action, reward, next_obs, done)


# ============================== Tests


def test_lambda_returns():
    sep("1. compute_lambda_returns")
    B, T = 4, 8
    rewards = jnp.ones((B, T)) * 0.5
    values_next = jnp.zeros((B, T))
    continues = jnp.ones((B, T))

    returns = compute_lambda_returns(rewards, values_next, continues, gamma=0.99, lambda_=0.95)
    check(returns.shape == (B, T), f"shape (B, T) = {returns.shape}")
    check(jnp.all(jnp.isfinite(returns)), "pas de NaN/Inf")
    # Avec r=0.5, gamma=0.99, lambda=0.95, V=0 : les returns sont décroissants vers la fin
    # Le dernier R = r_{T-1} + gamma * c * lambda * R_T = 0.5 + 0.99*1*0.95*0 = 0.5
    # (puisque la formule prend R_next = values_next du dernier step, ici 0)
    # Wait, on lit le code : R_final = values_next_T[-1] = 0
    # Then for t=T-1 : R = r[T-1] + gamma * c[T-1] * ((1-lambda) * v_next[T-1] + lambda * R_final)
    #                = 0.5 + 0.99 * 1 * (0.05 * 0 + 0.95 * 0) = 0.5
    last_return = float(returns[0, -1])
    check(abs(last_return - 0.5) < 1e-5, f"R_{{T-1}} = {last_return:.4f} (attendu ~0.5)")

    # Avec continues=0, le return doit être juste rewards[t]
    continues_zero = jnp.zeros((B, T))
    returns_zero = compute_lambda_returns(rewards, values_next, continues_zero, gamma=0.99, lambda_=0.95)
    check(
        jnp.allclose(returns_zero, rewards),
        f"avec continues=0, returns == rewards (max diff: {float(jnp.max(jnp.abs(returns_zero - rewards))):.2e})",
    )


def test_imagine_trajectory():
    sep("2. imagine_trajectory")
    rngs = nnx.Rngs(0)
    rssm = RSSM(
        embed_dim=EMBED_DIM, action_dim=ACTION_DIM,
        h_dim=H_DIM, z_categories=Z_CATEGORIES, z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM, rngs=rngs,
    )
    actor = Actor(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, rngs=rngs)
    reward_head = RewardHead(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, rngs=rngs)
    continue_head = ContinueHead(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, rngs=rngs)

    BT = 8
    initial_state = rssm.init_state(BT)
    key = jr.PRNGKey(42)

    traj = imagine_trajectory(
        initial_state, actor, rssm, reward_head, continue_head,
        TEST_HORIZON, ACTION_DIM, key,
    )

    check(traj["states"].shape == (BT, TEST_HORIZON, rssm.state_dim),
          f"states shape {traj['states'].shape}")
    check(traj["actions"].shape == (BT, TEST_HORIZON),
          f"actions shape {traj['actions'].shape}")
    check(traj["rewards"].shape == (BT, TEST_HORIZON),
          f"rewards shape {traj['rewards'].shape}")
    check(traj["continues"].shape == (BT, TEST_HORIZON),
          f"continues shape {traj['continues'].shape}")
    check(traj["log_probs"].shape == (BT, TEST_HORIZON),
          f"log_probs shape {traj['log_probs'].shape}")
    check(traj["entropies"].shape == (BT, TEST_HORIZON),
          f"entropies shape {traj['entropies'].shape}")
    check(traj["last_state"].shape == (BT, rssm.state_dim),
          f"last_state shape {traj['last_state'].shape}")
    check(jnp.all(jnp.isfinite(traj["rewards"])), "rewards finis")
    check(jnp.all((traj["continues"] >= 0) & (traj["continues"] <= 1)),
          "continues dans [0, 1] (post-sigmoid)")


def test_train_step():
    sep("3. train_step_wm + train_step_ac (10 iter)")

    rng = np.random.default_rng(42)

    # ----- Setup models
    rngs = nnx.Rngs(0)
    encoder = CNNEncoder(in_channels=3, embed_dim=EMBED_DIM, base_channels=32, rngs=rngs)
    rssm = RSSM(
        embed_dim=EMBED_DIM, action_dim=ACTION_DIM,
        h_dim=H_DIM, z_categories=Z_CATEGORIES, z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM, rngs=rngs,
    )
    decoder = CNNDecoder(state_dim=rssm.state_dim, out_channels=3, base_channels=32, rngs=rngs)
    reward_head = RewardHead(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, rngs=rngs)
    continue_head = ContinueHead(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, rngs=rngs)
    actor = Actor(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM, rngs=rngs)
    critic = Critic(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, rngs=rngs)

    slow_critic = Critic(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM,
                         rngs=nnx.Rngs(100))
    nnx.update(slow_critic, nnx.state(critic, nnx.Param))

    # Optimizers
    wm_bundle = (encoder, rssm, decoder, reward_head, continue_head)
    tx_wm = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(LR_WM))
    opt_wm = nnx.Optimizer(wm_bundle, tx_wm, wrt=nnx.Param)

    ac_bundle = (actor, critic)
    tx_ac = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(LR_AC))
    opt_ac = nnx.Optimizer(ac_bundle, tx_ac, wrt=nnx.Param)

    # ----- Buffer dummy
    buffer = ImageReplayBuffer(capacity=BUFFER_CAPACITY, obs_shape=OBS_SHAPE)
    fill_dummy_buffer(buffer, N_TRANSITIONS_INIT, rng)
    check(len(buffer) == N_TRANSITIONS_INIT, f"buffer rempli : {len(buffer)} transitions")

    # ----- Save initial slow critic param (pour vérifier Polyak)
    slow_kernel_initial = np.array(slow_critic.linear1.kernel[...])

    # ----- 10 iter de train
    return_ema_std = jnp.array(1.0, dtype=jnp.float32)
    key = jr.PRNGKey(123)

    print()
    print(f"  {'iter':>4s} | {'loss_wm':>8s} {'recon':>8s} {'kl':>8s} {'reward':>8s} {'cont':>8s} | "
          f"{'actor':>8s} {'critic':>8s} {'H':>6s} | {'ips':>6s}")
    print("  " + "-" * 90)

    t_start = time.time()
    losses_history = []

    for it in range(N_ITER_TEST):
        # Train WM
        batch_np = buffer.sample_sequences(BATCH_SIZE_TEST, SEQ_LEN_TEST)
        batch_jax = {
            "obs": jnp.array(batch_np["obs"]),
            "actions": jnp.array(batch_np["actions"], dtype=jnp.int32),
            "rewards": jnp.array(batch_np["rewards"]),
            "dones": jnp.array(batch_np["dones"], dtype=jnp.float32),
        }
        key, subk = jr.split(key)
        wm_metrics = train_step_wm_jit(
            encoder, rssm, decoder, reward_head, continue_head,
            opt_wm, batch_jax, subk,
        )

        # Train AC
        batch_np = buffer.sample_sequences(BATCH_SIZE_TEST, SEQ_LEN_TEST)
        batch_jax = {
            "obs": jnp.array(batch_np["obs"]),
            "actions": jnp.array(batch_np["actions"], dtype=jnp.int32),
            "dones": jnp.array(batch_np["dones"], dtype=jnp.float32),
        }
        key, subk = jr.split(key)
        ac_metrics, return_ema_std = train_step_ac_jit(
            encoder, rssm, reward_head, continue_head,
            actor, critic, slow_critic,
            opt_ac, batch_jax, return_ema_std, subk,
            ENTROPY_COEF,
        )

        all_metrics = {**wm_metrics, **ac_metrics}
        # Force tous les arrays à devenir Python floats (synchro implicite)
        vals = {k: float(v) for k, v in all_metrics.items()}
        losses_history.append(vals)

        # Vérifs anti-NaN
        for name, v in vals.items():
            if not np.isfinite(v):
                raise AssertionError(f"NaN/Inf détecté à iter {it} : {name} = {v}")

        ips = (it + 1) / (time.time() - t_start)
        print(
            f"  {it+1:4d} | "
            f"{vals['loss_wm']:8.3f} {vals['loss_recon']:8.4f} {vals['loss_kl']:8.4f} "
            f"{vals['loss_reward']:8.4f} {vals['loss_continue']:8.4f} | "
            f"{vals['loss_actor']:8.3f} {vals['loss_critic']:8.3f} {vals['entropy']:6.2f} | "
            f"{ips:6.2f}"
        )

    # ----- Assertions globales
    print()
    print("  Vérifications post-training :")
    first = losses_history[0]
    last = losses_history[-1]

    # Reconstruction MSE doit baisser (training basic functional check)
    check(
        last["loss_recon"] < first["loss_recon"] * 1.5,
        f"loss_recon ne pas exploser : {first['loss_recon']:.4f} → {last['loss_recon']:.4f}",
        fatal=False,
    )

    # Slow critic doit avoir changé (Polyak update fonctionnel)
    slow_kernel_after = np.array(slow_critic.linear1.kernel[...])
    diff = np.abs(slow_kernel_after - slow_kernel_initial).mean()
    check(
        diff > 1e-8,
        f"slow_critic mis à jour via Polyak (mean diff = {diff:.2e})",
    )

    # Ordre de grandeur des losses
    check(0 < last["loss_recon"] < 10.0, f"loss_recon raisonnable : {last['loss_recon']:.4f}")
    check(0 < last["loss_kl"] < 100.0, f"loss_kl raisonnable : {last['loss_kl']:.4f}")
    check(-50 < last["loss_actor"] < 50, f"loss_actor raisonnable : {last['loss_actor']:.4f}")
    check(0 < last["loss_critic"] < 100.0, f"loss_critic raisonnable : {last['loss_critic']:.4f}")
    check(0 < last["entropy"] < 5.0, f"entropy positive et raisonnable : {last['entropy']:.3f}")

    total_time = time.time() - t_start
    ips_final = N_ITER_TEST / total_time
    print(f"\n  Performance : {N_ITER_TEST} iter en {total_time:.2f}s ({ips_final:.2f} ips)")

    return ips_final


def main():
    print()
    print("=" * 60)
    print("  Tests train_dreamer_jax")
    print("=" * 60)
    print(f"  JAX version  : {jax.__version__}")
    print(f"  Backend      : {jax.default_backend()}")
    print(f"  Devices      : {jax.devices()}")

    test_lambda_returns()
    test_imagine_trajectory()
    ips = test_train_step()

    sep("Résumé")
    print(f"  Tous les tests passent.")
    print(f"  Premières mesures ips : {ips:.2f} iter/s sur CPU local.")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
