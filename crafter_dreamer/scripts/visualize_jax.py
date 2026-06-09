"""
Visualize a trained Crafter Dreamer JAX agent playing.

Charge un checkpoint .npz produit par train_dreamer_jax.py, reconstruit
encoder + RSSM + actor, joue 1-N episodes dans Crafter, et sauvegarde
les frames RGB sous forme de GIF ou MP4.

Usage :
    .venv/bin/python crafter_dreamer/scripts/visualize_jax.py \\
        --checkpoint ./modal_outputs/checkpoints/dreamer_crafter_jax_xxx_iter010000.npz \\
        --output ./videos/episode_iter10000.gif \\
        --n_episodes 3 \\
        --fps 10 \\
        --deterministic

Notes :
    - MP4 requiert imageio_ffmpeg (pip install imageio[ffmpeg]).
      Si non disponible, utiliser un output .gif (fonctionne avec Pillow).
    - Le premier appel JIT prend ~10-30s (compile), les suivants sont rapides.
"""

import sys
import argparse
import json
from pathlib import Path

# Project root sur le sys.path pour permettre les imports relatifs au repo
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from crafter_dreamer.env import CrafterEnv
from src_jax.model import CNNEncoder, RSSM, Actor


# ============================== Constantes (doivent matcher train_dreamer_jax.py)
EMBED_DIM = 192
H_DIM = 384
Z_CATEGORIES = 24
Z_CLASSES = 24
HIDDEN_DIM = 768
ACTION_DIM = 17   # Crafter
BASE_CHANNELS = 32
INPUT_RES = 64


# ============================== Helpers de (de)serialisation des states

def _variable_to_numpy(v):
    """Extrait le contenu d'une nnx.Variable ou d'un array en numpy."""
    try:
        return np.array(v[...])
    except (TypeError, IndexError):
        pass
    if hasattr(v, "get_value"):
        return np.array(v.get_value())
    return np.array(v)


def flatten_state(state, prefix=""):
    """Flatten un pytree de state nnx.Param en {dotted_path: array}.

    Reimplementation identique a train_dreamer_jax.flatten_state.
    """
    out = {}
    if hasattr(state, "items"):
        items = state.items()
    elif isinstance(state, (list, tuple)):
        items = enumerate(state)
    else:
        return {prefix.rstrip("."): _variable_to_numpy(state)}

    for k, v in items:
        new_prefix = f"{prefix}{k}."
        if isinstance(v, (jnp.ndarray, np.ndarray)):
            out[new_prefix.rstrip(".")] = np.array(v)
        elif hasattr(v, "items") or isinstance(v, (list, tuple)):
            out.update(flatten_state(v, new_prefix))
        else:
            out[new_prefix.rstrip(".")] = _variable_to_numpy(v)
    return out


def _set_at_path(state, path_parts, value):
    """Affecte recursivement une valeur a l'interieur d'un state nnx.

    state peut etre un dict, un nnx.State, ou contenir des Variables.
    On descend selon les keys de path_parts, puis on remplace la feuille
    via Variable.value (ou par affectation directe pour les dicts).
    """
    if len(path_parts) == 1:
        key = path_parts[0]
        # Conversion eventuelle d'index numerique
        try:
            key_int = int(key)
            container_key = key_int if (isinstance(state, (list, tuple))
                                        or _has_int_key(state, key_int)) else key
        except ValueError:
            container_key = key

        leaf = state[container_key]
        # Si feuille = Variable : remplace .value, sinon ecrase directement
        if hasattr(leaf, "value"):
            leaf.value = jnp.asarray(value)
        else:
            state[container_key] = jnp.asarray(value)
        return

    key = path_parts[0]
    try:
        key_int = int(key)
        container_key = key_int if (isinstance(state, (list, tuple))
                                    or _has_int_key(state, key_int)) else key
    except ValueError:
        container_key = key

    _set_at_path(state[container_key], path_parts[1:], value)


def _has_int_key(container, k):
    """True si container[k] (k int) marche sans throw."""
    try:
        _ = container[k]
        return True
    except (KeyError, TypeError, IndexError):
        return False


def load_module_params(module, ckpt, prefix):
    """Charge dans `module` (instance nnx.Module) les params du checkpoint
    correspondant a la clef prefixe `prefix.` du npz.

    Approche : on recupere le state nnx, on remplace chaque feuille par sa
    version chargee, puis nnx.update.
    """
    state = nnx.state(module, nnx.Param)

    # Recupere les keys attendues pour ce module (sans prefix)
    flat_expected = flatten_state(state)
    expected_keys = set(flat_expected.keys())

    # Sous-ensemble du ckpt pour ce module
    loaded = {}
    missing = []
    for ek in expected_keys:
        full_key = f"{prefix}.{ek}"
        if full_key in ckpt.files:
            loaded[ek] = ckpt[full_key]
        else:
            missing.append(full_key)

    if missing:
        print(f"  WARN [{prefix}]: {len(missing)} missing keys (e.g. {missing[:3]})")

    # Affecte les feuilles dans le state
    for key, value in loaded.items():
        path = key.split(".")
        _set_at_path(state, path, value)

    nnx.update(module, state)


# ============================== Build + load des modules

def build_modules(seed: int = 0):
    """Instancie encoder + RSSM + actor avec les hyperparams attendus."""
    rngs = nnx.Rngs(seed)
    encoder = CNNEncoder(
        in_channels=3, embed_dim=EMBED_DIM,
        base_channels=BASE_CHANNELS, input_resolution=INPUT_RES,
        rngs=rngs,
    )
    rssm = RSSM(
        embed_dim=EMBED_DIM, action_dim=ACTION_DIM,
        h_dim=H_DIM, z_categories=Z_CATEGORIES, z_classes=Z_CLASSES,
        hidden_dim=HIDDEN_DIM, rngs=rngs,
    )
    state_dim = rssm.state_dim  # H_DIM + Z_CATEGORIES * Z_CLASSES
    actor = Actor(
        state_dim=state_dim, action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM, rngs=rngs,
    )
    return encoder, rssm, actor


def load_checkpoint(checkpoint_path: Path, seed: int = 0):
    """Charge un checkpoint .npz et retourne (encoder, rssm, actor, meta)."""
    ckpt = np.load(checkpoint_path, allow_pickle=False)
    print(f"Loaded {len(ckpt.files)} keys from {checkpoint_path.name}")

    # Metadata side file (optionnel : utile pour logging)
    meta = None
    meta_path = checkpoint_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  meta : iter={meta.get('iter', '?')}")
        except Exception as e:
            print(f"  meta : failed to read ({e})")

    encoder, rssm, actor = build_modules(seed=seed)

    print("Loading params...")
    load_module_params(encoder, ckpt, prefix="encoder")
    load_module_params(rssm, ckpt, prefix="rssm")
    load_module_params(actor, ckpt, prefix="actor")

    return encoder, rssm, actor, meta


# ============================== Forward pass : obs -> action

def make_act_fn(encoder, rssm, actor):
    """Construit une fonction JIT compilee qui calcule (new_state, action)
    a partir d'une obs courante, d'un state RSSM et d'une action precedente.

    Closure sur les modules pour pouvoir bencher en jit sans recompile.
    """

    @nnx.jit
    def act(encoder, rssm, actor, obs_bchw, state, prev_action_onehot, key, deterministic):
        # Encode (batch dim = 1)
        embedding = encoder(obs_bchw)  # (1, embed_dim)

        # RSSM observe (utilise le posterior car on a l'obs)
        new_state, _post_logits, _prior_logits = rssm.observe_step(
            state, prev_action_onehot, embedding, key
        )

        # State complet h+z pour l'actor
        state_vec = RSSM.get_state_vec(new_state)  # (1, h_dim + z_dim)
        logits = actor(state_vec)                  # (1, action_dim)

        # Choix de l'action
        det_action = jnp.argmax(logits, axis=-1)
        stoch_action = jr.categorical(key, logits)
        action = jnp.where(deterministic, det_action, stoch_action)  # (1,)

        return new_state, action

    def call(obs_chw, state, prev_action_idx, key, deterministic):
        """Wrapper Python avec batch dim explicite + conversion en jnp."""
        obs_bchw = jnp.asarray(obs_chw, dtype=jnp.float32)[None, ...]
        prev_action_onehot = jax.nn.one_hot(
            jnp.asarray([prev_action_idx]), ACTION_DIM
        )
        new_state, action = act(
            encoder, rssm, actor,
            obs_bchw, state, prev_action_onehot, key,
            jnp.asarray(deterministic),
        )
        return new_state, int(action[0])

    return call


# ============================== Play loop

def play_episode(act_fn, rssm, env, key, deterministic=False, max_steps=500,
                 verbose=True, render_size=(512, 512)):
    """Joue un episode complet. Retourne (frames_uint8_hwc, achievements, total_reward, steps).

    render_size : (W, H) — résolution du GIF/MP4 final. Default 512×512 (vs 64×64 native).
    """
    obs = env.reset()  # (3, 64, 64) float32 [0, 1]

    # State RSSM initial
    state = rssm.init_state(batch_size=1)

    # Action precedente = 0 (noop) au reset
    prev_action_idx = 0

    frames = []
    achievements_seen = []
    total_reward = 0.0
    steps_done = 0

    # Accès à l'env crafter sous-jacent (CrafterEnv wrappe ._env)
    inner_env = env._env if hasattr(env, "_env") else env

    for t in range(max_steps):
        # Frame courante AVANT l'action (résolution custom via crafter render(size))
        frame = inner_env.render(render_size)
        frames.append(frame)

        # Forward
        key, subkey = jr.split(key)
        state, action_idx = act_fn(
            obs, state, prev_action_idx, subkey, deterministic,
        )

        # Step env
        obs, reward, done, info = env.step(action_idx)
        total_reward += reward
        steps_done = t + 1

        # Track achievements
        if info and "achievements" in info:
            for ach, unlocked in info["achievements"].items():
                if unlocked and ach not in achievements_seen:
                    achievements_seen.append(ach)
                    if verbose:
                        print(f"    step {t:4d} | unlocked : {ach}")

        prev_action_idx = action_idx

        if done:
            # Capture une frame finale
            frames.append(inner_env.render(render_size))
            break

    return frames, achievements_seen, total_reward, steps_done


# ============================== Save video

def save_video(frames, output_path: Path, fps: int = 10):
    """Sauvegarde frames (list of uint8 HWC) en GIF ou MP4."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        print("ERROR : no frames to save")
        return

    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        # GIF via Pillow (toujours dispo, pas de ffmpeg requis)
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("PIL/Pillow not installed (needed for GIF)")

        pil_frames = [Image.fromarray(f) for f in frames]
        duration_ms = int(1000 / max(fps, 1))
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        print(f"GIF saved : {output_path} ({len(frames)} frames @ {fps} fps)")

    elif suffix == ".mp4":
        try:
            import imageio
        except ImportError:
            raise RuntimeError("imageio not installed. Try : pip install imageio")
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "MP4 output requires imageio_ffmpeg. Install with :\n"
                "  pip install imageio[ffmpeg]\n"
                "Or use a .gif output instead."
            )
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"MP4 saved : {output_path} ({len(frames)} frames @ {fps} fps)")

    else:
        raise ValueError(
            f"Unsupported output format : {suffix}. Use .gif or .mp4"
        )


# ============================== Main

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained Crafter Dreamer JAX agent.",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .npz checkpoint")
    parser.add_argument("--output", type=str, default="videos/episode.gif",
                        help="Output path (.gif or .mp4)")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true",
                        help="Use argmax instead of categorical sample")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for env + sampling")
    parser.add_argument("--render_size", type=int, default=512,
                        help="Resolution rendered (square, default 512). 64=native, 256/512/1024 upscale")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR : checkpoint not found : {checkpoint_path}")
        sys.exit(1)

    output_path = Path(args.output)

    # ----- Load
    print(f"=== Loading checkpoint")
    encoder, rssm, actor, meta = load_checkpoint(checkpoint_path, seed=args.seed)
    act_fn = make_act_fn(encoder, rssm, actor)

    # ----- Env
    env = CrafterEnv(seed=args.seed)

    # ----- Play
    all_frames = []
    per_episode_stats = []
    key = jr.PRNGKey(args.seed)

    for ep in range(args.n_episodes):
        print(f"\n=== Episode {ep + 1}/{args.n_episodes}"
              f"   (deterministic={args.deterministic})")
        key, subkey = jr.split(key)
        frames, achievements, reward, steps = play_episode(
            act_fn, rssm, env, subkey,
            deterministic=args.deterministic,
            max_steps=args.max_steps,
            verbose=True,
            render_size=(args.render_size, args.render_size),
        )
        print(f"  -> steps={steps}, reward={reward:.2f}, "
              f"achievements={len(achievements)}")
        if achievements:
            print(f"     unlocked : {achievements}")
        all_frames.extend(frames)
        per_episode_stats.append({
            "episode": ep + 1,
            "steps": steps,
            "reward": reward,
            "achievements": achievements,
        })

    # ----- Save
    print(f"\n=== Saving video")
    save_video(all_frames, output_path, fps=args.fps)

    # ----- Summary
    print(f"\n=== Summary")
    print(f"  Episodes      : {args.n_episodes}")
    print(f"  Total frames  : {len(all_frames)}")
    mean_reward = float(np.mean([s["reward"] for s in per_episode_stats]))
    mean_ach = float(np.mean([len(s["achievements"]) for s in per_episode_stats]))
    print(f"  Mean reward   : {mean_reward:.2f}")
    print(f"  Mean unlocks  : {mean_ach:.2f}")
    if meta is not None:
        print(f"  From iter     : {meta.get('iter', '?')}")


if __name__ == "__main__":
    main()
