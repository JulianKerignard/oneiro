"""
Credit-assignment probe — does the agent rank the REWARDING action first?

The metric the run logs (`rew@ach`, `scale`, `H`, achievements) cannot tell whether the
policy actually credits the action that earns a +1. This probe measures it directly, on a
checkpoint, in ~2 min on CPU.

Method
------
Roll out the policy in the real Crafter env. At every step, fork the env and try action
`do` (index 5) to label the state:
    positive : `do` increments `wood`   -> the +1 for collect_wood is one press away
    control  : `do` yields nothing      -> same action, no payoff
Then rank all 17 actions within each state by
    (a) the actor logits                     — pure policy, no world model involved
    (b) r(s) + gamma * V(s')                 — one-step proxy for the advantage
and report the rank of `do` in each group.

Reading the result
------------------
    rank 1  in the positive group  -> credit assignment works
    rank ~9 in both groups         -> uninformative (chance)
    positive rank WORSE than control -> INVERTED credit: the policy avoids the payoff

Measured on v50 @35k (2026-07-30): positive 17/17 by actor, control 2/17 — a -15 rank
inversion. See docs/HYPOTHESES.md H_317.

Caveats
-------
- (b) is a ONE-STEP lookahead, not the horizon-16 lambda-return the actor is trained on.
  It is a proxy. (a) has no such caveat: it is the policy itself.
- The same PRNG key is reused across the 17 actions so the only difference is the action
  (the categorical z sample is stochastic; a per-action key would swamp the signal).

Usage
-----
    JAX_PLATFORMS=cpu .venv/bin/python experiments/credit_assignment_probe.py \
        --checkpoint kaggle_outputs/v50-ckpt/checkpoints/dreamer_crafter_jax_v50-wmfix-t1_iter035000.npz
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
from flax import nnx

from crafter_dreamer.env import CrafterEnv
from src_jax.model import CNNEncoder, RSSM, Actor, Critic, RewardHead

DO_ACTION = 5          # index of "do" in ACTION_NAMES
GAMMA = 0.997
N_KEYS = 4             # z samples averaged per action


def load_modules(path: Path, *, embed_dim, h_dim, z_cat, z_cls, hidden, cnn_depth,
                 ac_hidden, ac_layers):
    """Rebuild the modules and fill them from a save_checkpoint() .npz."""
    ckpt = np.load(path)
    state_dim = h_dim + z_cat * z_cls
    rngs = nnx.Rngs(0)
    mods = {
        "encoder": CNNEncoder(embed_dim=embed_dim, base_channels=cnn_depth, rngs=rngs),
        "rssm": RSSM(embed_dim=embed_dim, action_dim=17, h_dim=h_dim, z_categories=z_cat,
                     z_classes=z_cls, hidden_dim=hidden, rngs=rngs),
        "actor": Actor(state_dim=state_dim, hidden_dim=ac_hidden, action_dim=17,
                       num_layers=ac_layers, rngs=rngs),
        "critic": Critic(state_dim=state_dim, hidden_dim=ac_hidden,
                         num_layers=ac_layers, rngs=rngs),
        "reward_head": RewardHead(state_dim=state_dim, hidden_dim=hidden, rngs=rngs),
    }
    for prefix, module in mods.items():
        flat = dict(nnx.to_flat_state(nnx.state(module, nnx.Param)))
        missing = 0
        for path_parts, var in flat.items():
            key = prefix + "." + ".".join(str(p) for p in path_parts)
            if key in ckpt.files:
                var.value = jnp.asarray(ckpt[key])
            else:
                missing += 1
        if missing:
            print(f"  [warn] {prefix}: {missing} clés absentes du checkpoint")
        nnx.update(module, nnx.from_flat_state(flat))
    return mods, state_dim


def action_scores(mods, state_vec, h_dim):
    """r(s) + gamma * V(s') for the 17 actions, same z key across actions."""
    rssm, critic, reward_head = mods["rssm"], mods["critic"], mods["reward_head"]
    base = float(reward_head.predict(state_vec)[0])
    scores = np.zeros(17)
    for ki in range(N_KEYS):
        key = jax.random.PRNGKey(777 + ki)
        for a in range(17):
            action_oh = jax.nn.one_hot(jnp.array([a]), 17)
            new_state, _ = rssm.imagine_step(
                {"h": state_vec[:, :h_dim], "z": state_vec[:, h_dim:]}, action_oh, key,
            )
            nsv = jnp.concatenate([new_state["h"], new_state["z"]], axis=-1)
            scores[a] += base + GAMMA * float(critic.predict(nsv)[0])
    return scores / N_KEYS


def collect_states(mods, h_dim, n_target, max_episodes, seed0):
    """Roll out on-policy, labelling each state by whether `do` would yield wood."""
    encoder, rssm, actor = mods["encoder"], mods["rssm"], mods["actor"]
    positive, control = [], []
    for ep in range(max_episodes):
        env = CrafterEnv(seed=seed0 + ep)
        obs = env.reset()
        state = rssm.init_state(1)
        prev_action = jnp.zeros((1, 17))
        for t in range(200):
            emb = encoder(jnp.asarray(obs)[None])
            state, _, _ = rssm.observe_step(state, prev_action, emb, jax.random.PRNGKey(t))
            state_vec = jnp.concatenate([state["h"], state["z"]], axis=-1)

            # Fork the env to label the state without consuming the real step.
            wood_before = env._env._player.inventory.get("wood", 0)
            snapshot = copy.deepcopy(env._env)
            env._env.step(DO_ACTION)
            gained = env._env._player.inventory.get("wood", 0) - wood_before
            env._env = snapshot

            (positive if gained > 0 else control).append(state_vec)

            action = int(jax.random.categorical(
                jax.random.PRNGKey(1000 + t), actor(state_vec)[0]))
            obs, _, done, _ = env.step(action)
            prev_action = jax.nn.one_hot(jnp.array([action]), 17)
            if done:
                break
        if len(positive) >= n_target:
            break
    return positive, control[:len(positive)]


def rank_of_do(mods, group, h_dim):
    by_value, by_policy = [], []
    for state_vec in group:
        scores = action_scores(mods, state_vec, h_dim)
        by_value.append(int(np.where(np.argsort(scores)[::-1] == DO_ACTION)[0][0]) + 1)
        logits = np.asarray(mods["actor"](state_vec)[0])
        by_policy.append(int(np.where(np.argsort(logits)[::-1] == DO_ACTION)[0][0]) + 1)
    return np.array(by_value), np.array(by_policy)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--n_states", type=int, default=80)
    p.add_argument("--max_episodes", type=int, default=70)
    p.add_argument("--seed0", type=int, default=70_000)
    # Architecture must match the checkpoint (defaults = the ~65M config).
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--h_dim", type=int, default=2048)
    p.add_argument("--z_categories", type=int, default=32)
    p.add_argument("--z_classes", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--cnn_depth", type=int, default=32)
    p.add_argument("--ac_hidden_dim", type=int, default=1024)
    p.add_argument("--ac_num_layers", type=int, default=3)
    args = p.parse_args()

    print(f"Checkpoint : {args.checkpoint.name}")
    mods, _ = load_modules(
        args.checkpoint, embed_dim=args.embed_dim, h_dim=args.h_dim,
        z_cat=args.z_categories, z_cls=args.z_classes, hidden=args.hidden_dim,
        cnn_depth=args.cnn_depth, ac_hidden=args.ac_hidden_dim, ac_layers=args.ac_num_layers,
    )

    positive, control = collect_states(
        mods, args.h_dim, args.n_states, args.max_episodes, args.seed0)
    print(f"États : {len(positive)} positifs (do → bois) | {len(control)} contrôle\n")
    if not positive:
        print("Aucun état positif trouvé — augmenter --max_episodes.")
        return

    results = {}
    for label, group in (("[+] do RAPPORTE du bois", positive),
                         ("[-] contrôle : do ne rapporte rien", control)):
        rv, rp = rank_of_do(mods, group, args.h_dim)
        results[label] = (rv, rp)
        print(f"  {label}")
        print(f"    rang de 'do' par r+γV : médiane={np.median(rv):>4.1f}/17"
              f"   #1 dans {100 * np.mean(rv == 1):>4.0f}% des cas")
        print(f"    rang de 'do' par π    : médiane={np.median(rp):>4.1f}/17"
              f"   #1 dans {100 * np.mean(rp == 1):>4.0f}% des cas")

    (pv, pp), (cv, cp) = results["[+] do RAPPORTE du bois"], results["[-] contrôle : do ne rapporte rien"]
    gap_v, gap_p = np.median(cv) - np.median(pv), np.median(cp) - np.median(pp)
    print(f"\n  DISCRIMINATION (positif = le système préfère l'action payante)")
    print(f"    écart de rang par r+γV : {gap_v:+.1f}")
    print(f"    écart de rang par π    : {gap_p:+.1f}")
    print(f"    hasard = 9.0/17 dans les deux groupes, écart 0.0")

    # Le verdict porte d'abord sur le rang ABSOLU dans le groupe positif : la question
    # est « la politique choisit-elle l'action qui paie ? ». L'écart au contrôle est
    # secondaire — `do` sert aussi à récolter des saplings, donc le contrôle peut
    # légitimement être bon lui aussi, ce qui comprime l'écart sans rien dire de mauvais.
    rank_pos, top1_pos = float(np.median(pp)), float(np.mean(pp == 1))
    if rank_pos <= 3.0 and top1_pos >= 0.5:
        verdict = f"CRÉDIT SAIN — 'do' classée {rank_pos:.0f}e/17, #1 dans {100*top1_pos:.0f}% des cas"
    elif gap_p < -2:
        verdict = "CRÉDIT INVERSÉ — la politique évite l'action qui paie"
    elif rank_pos >= 7.0:
        verdict = "crédit non informatif (proche du hasard)"
    else:
        verdict = f"crédit partiel — 'do' classée {rank_pos:.0f}e/17, #1 dans {100*top1_pos:.0f}%"
    print(f"\n  VERDICT : {verdict}")


if __name__ == "__main__":
    main()
