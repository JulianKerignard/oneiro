"""
Micro-benchmark isolé des sous-systèmes du training JAX.

But : profiler en local CPU rapidement, sans payer le coût d'un full pipeline.
Mesure chaque composant en isolation après un warmup (JIT compile),
puis chronomètre N répétitions pour une moyenne stable.

Sections benchmarkées (en ordre d'intérêt) :
  1. env.step    : Crafter Python pur, séquentiel sur N envs
  2. act_fn      : jit observe_step + sample action (collecte)
  3. transfer    : jnp.array <-> np.array round-trip
  4. sample_batch: buffer.sample_sequences + jnp.array
  5. train_wm    : un step de WM jit
  6. train_ac    : un step d'AC jit

Le ratio relatif de ces mesures sur CPU est représentatif (à un facteur près)
de ce qu'on observerait sur Modal L4 GPU pour les sections CPU-bound
(env.step, transfer). Les sections GPU-bound (train_wm/ac) seront plus rapides
sur GPU L4 que sur Mac CPU, mais la **collecte env reste séquentielle**
et donc le ratio collecte/total grandit sur GPU.

Usage :
    .venv/bin/python crafter_dreamer/scripts/microbench_jax.py
    .venv/bin/python crafter_dreamer/scripts/microbench_jax.py --n_envs 4 --reps 5
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
import optax

from crafter_dreamer.env import CrafterEnv
from src_jax.buffer import ImageReplayBuffer
from src_jax.model import (
    CNNEncoder, CNNDecoder, RSSM, RewardHead, ContinueHead,
    Actor, Critic,
)
from crafter_dreamer.scripts.train_dreamer_jax import (
    train_step_wm_jit, train_step_ac_jit, make_act_fn,
    SEQ_LEN, EMBED_DIM, H_DIM, Z_CATEGORIES, Z_CLASSES, HIDDEN_DIM,
    LR_WM, LR_AC, GRAD_CLIP,
)


def time_block(label, fn, reps=5, warmup=1):
    """Run fn() warmup+reps fois, retourne (mean_ms, std_ms)."""
    # Warmup (JIT compile + first call cost)
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        # Block JAX async
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        elif isinstance(out, (list, tuple)):
            for x in out:
                if hasattr(x, "block_until_ready"):
                    x.block_until_ready()
        elif isinstance(out, dict):
            for x in out.values():
                if hasattr(x, "block_until_ready"):
                    x.block_until_ready()
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
    times = np.array(times)
    print(f"  {label:<22} : {times.mean():>8.2f} ms  (std {times.std():>5.2f} ms, "
          f"min {times.min():>6.2f}, max {times.max():>6.2f}, reps={reps})")
    return times.mean(), times.std()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"JAX backend : {jax.default_backend()}")
    print(f"Devices     : {jax.devices()}")
    print(f"n_envs      : {args.n_envs}")
    print(f"batch_size  : {args.batch_size}")
    print(f"reps        : {args.reps}")
    print()

    seed = args.seed
    np.random.seed(seed)

    # Setup envs
    envs = [CrafterEnv(seed=seed + i) for i in range(args.n_envs)]
    obs_list = [env.reset() for env in envs]
    obs_shape = obs_list[0].shape
    action_dim = envs[0].action_dim

    buffer = ImageReplayBuffer(capacity=5000, obs_shape=obs_shape)
    # Remplir avec assez de transitions pour permettre les samples seq
    for _ in range(SEQ_LEN + 50):
        for i, env in enumerate(envs):
            a = np.random.randint(0, action_dim)
            next_obs, r, done, _ = env.step(a)
            buffer.add(obs_list[i], a, r, next_obs, done)
            obs_list[i] = next_obs if not done else env.reset()
    print(f"Buffer rempli avec {len(buffer)} transitions")
    print()

    # Models
    rngs = nnx.Rngs(seed)
    encoder = CNNEncoder(in_channels=3, embed_dim=EMBED_DIM, base_channels=32, rngs=rngs)
    rssm = RSSM(
        embed_dim=EMBED_DIM, action_dim=action_dim,
        h_dim=H_DIM, z_categories=Z_CATEGORIES, z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM, rngs=rngs,
    )
    decoder = CNNDecoder(state_dim=rssm.state_dim, out_channels=3, base_channels=32, rngs=rngs)
    reward_head = RewardHead(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, rngs=rngs)
    continue_head = ContinueHead(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, rngs=rngs)
    actor = Actor(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, action_dim=action_dim, rngs=rngs)
    critic = Critic(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, rngs=rngs)
    slow_critic = Critic(state_dim=rssm.state_dim, hidden_dim=HIDDEN_DIM, rngs=nnx.Rngs(seed + 100))
    nnx.update(slow_critic, nnx.state(critic, nnx.Param))

    wm_bundle = (encoder, rssm, decoder, reward_head, continue_head)
    opt_wm = nnx.Optimizer(wm_bundle, optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP), optax.adam(LR_WM)
    ), wrt=nnx.Param)
    ac_bundle = (actor, critic)
    opt_ac = nnx.Optimizer(ac_bundle, optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP), optax.adam(LR_AC)
    ), wrt=nnx.Param)

    return_ema_std = jnp.array(1.0, dtype=jnp.float32)
    act_fn = make_act_fn()

    # Persistent state across measurements
    rssm_state = rssm.init_state(args.n_envs)
    prev_actions_oh = jnp.zeros((args.n_envs, action_dim))

    print("=" * 60)
    print("MICRO-BENCHMARK")
    print("=" * 60)

    # ============ 1. env.step (Python pur, séquentiel sur N envs)
    def bench_env_step():
        # collect 1 step sur chaque env
        for i, env in enumerate(envs):
            a = np.random.randint(0, action_dim)
            next_obs, r, done, _ = env.step(a)
            if done:
                obs_list[i] = env.reset()
            else:
                obs_list[i] = next_obs
        return None
    t_env_step, _ = time_block(
        f"env.step x{args.n_envs} (seq)", bench_env_step,
        reps=args.reps * 4, warmup=2,  # plus de reps, l'env est rapide
    )

    # ============ 2. act_fn (jit) — un sample d'action
    main_key = jr.PRNGKey(seed)
    def bench_act_fn():
        nonlocal main_key, rssm_state, prev_actions_oh
        obs_batch = jnp.array(np.stack(obs_list))
        main_key, subk = jr.split(main_key)
        new_state, actions_int = act_fn(
            encoder, rssm, actor,
            rssm_state, prev_actions_oh,
            obs_batch, subk, None,
        )
        return new_state, actions_int
    t_act_fn, _ = time_block(
        "act_fn (jit)", bench_act_fn, reps=args.reps, warmup=2,
    )

    # ============ 3. transfer : np -> jnp -> np round-trip pour state RSSM
    def bench_transfer():
        h = np.zeros((args.n_envs, H_DIM), dtype=np.float32)
        z = np.zeros((args.n_envs, Z_CATEGORIES * Z_CLASSES), dtype=np.float32)
        h_j = jnp.array(h)
        z_j = jnp.array(z)
        h_back = np.array(h_j)
        z_back = np.array(z_j)
        return None
    t_transfer, _ = time_block(
        "transfer (np<->jnp)", bench_transfer, reps=args.reps * 4, warmup=2,
    )

    # ============ 4. sample_batch + transfer to jnp
    def bench_sample_batch():
        batch_np = buffer.sample_sequences(args.batch_size, SEQ_LEN)
        batch_jax = {
            "obs": jnp.array(batch_np["obs"]),
            "actions": jnp.array(batch_np["actions"], dtype=jnp.int32),
            "rewards": jnp.array(batch_np["rewards"]),
            "dones": jnp.array(batch_np["dones"], dtype=jnp.float32),
        }
        return batch_jax["obs"]
    t_sample, _ = time_block(
        "sample_batch", bench_sample_batch, reps=args.reps, warmup=2,
    )

    # ============ 5. train_step_wm (jit) — c'est le gros morceau
    def bench_train_wm():
        nonlocal main_key
        batch_np = buffer.sample_sequences(args.batch_size, SEQ_LEN)
        batch_jax = {
            "obs": jnp.array(batch_np["obs"]),
            "actions": jnp.array(batch_np["actions"], dtype=jnp.int32),
            "rewards": jnp.array(batch_np["rewards"]),
            "dones": jnp.array(batch_np["dones"], dtype=jnp.float32),
        }
        main_key, subk = jr.split(main_key)
        wm_metrics = train_step_wm_jit(
            encoder, rssm, decoder, reward_head, continue_head,
            opt_wm, batch_jax, subk,
        )
        return wm_metrics
    t_train_wm, _ = time_block(
        "train_step_wm (jit)", bench_train_wm, reps=args.reps, warmup=1,
    )

    # ============ 6. train_step_ac (jit)
    def bench_train_ac():
        nonlocal main_key, return_ema_std
        batch_np = buffer.sample_sequences(args.batch_size, SEQ_LEN)
        batch_jax = {
            "obs": jnp.array(batch_np["obs"]),
            "actions": jnp.array(batch_np["actions"], dtype=jnp.int32),
            "dones": jnp.array(batch_np["dones"], dtype=jnp.float32),
        }
        main_key, subk = jr.split(main_key)
        ac_metrics, return_ema_std = train_step_ac_jit(
            encoder, rssm, reward_head, continue_head,
            actor, critic, slow_critic,
            opt_ac, batch_jax, return_ema_std, subk,
            0.005,
        )
        return ac_metrics
    t_train_ac, _ = time_block(
        "train_step_ac (jit)", bench_train_ac, reps=args.reps, warmup=1,
    )

    # ============ Synthese
    # Coût d'1 train iter typique = collect_per_iter * (act_fn + env_step + 2*transfer) + train_wm + train_ac
    # COLLECT_PER_ITER=10 dans train_dreamer_jax.py, mais avec n_envs=4 → collect_per_iter = 10/4 = 2 (au moins 1)
    collect_per_iter = max(1, 10 // args.n_envs)
    coll_per_iter_ms = collect_per_iter * (t_act_fn + t_env_step + 2 * t_transfer)
    total_iter_ms = coll_per_iter_ms + t_sample + t_train_wm + t_sample + t_train_ac

    print()
    print("=" * 60)
    print("SYNTHESE PAR TRAIN_ITER (estimation)")
    print("=" * 60)
    print(f"  collect_per_iter         = {collect_per_iter}")
    print(f"  Collect (x{collect_per_iter}/iter)  : {coll_per_iter_ms:>8.2f} ms  "
          f"({coll_per_iter_ms / total_iter_ms * 100:>5.1f}%)")
    print(f"     act_fn                : {t_act_fn * collect_per_iter:>8.2f} ms")
    print(f"     env.step              : {t_env_step * collect_per_iter:>8.2f} ms")
    print(f"     transfer              : {2 * t_transfer * collect_per_iter:>8.2f} ms")
    print(f"  Sample batch (x2)        : {2 * t_sample:>8.2f} ms  "
          f"({2 * t_sample / total_iter_ms * 100:>5.1f}%)")
    print(f"  Train WM (jit)           : {t_train_wm:>8.2f} ms  "
          f"({t_train_wm / total_iter_ms * 100:>5.1f}%)")
    print(f"  Train AC (jit)           : {t_train_ac:>8.2f} ms  "
          f"({t_train_ac / total_iter_ms * 100:>5.1f}%)")
    print(f"  TOTAL / train_iter       : {total_iter_ms:>8.2f} ms  "
          f"→ {1000.0 / total_iter_ms:.2f} ips")
    print()


if __name__ == "__main__":
    main()
