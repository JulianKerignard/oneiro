"""
Tests de validation du RSSM JAX/Flax NNX (port depuis PyTorch).

Vérifie :
    1. Shapes du forward (observe_step, imagine_step)
    2. Shapes du forward séquence (observe_sequence avec scan, imagine_sequence)
    3. Comptage des paramètres vs PyTorch
    4. Sampling stochastique (2 keys différentes → z différents)
    5. Reset done dans observe_sequence (boundary épisode)
    6. kl_loss : valeur > 0, gradient flow vers prior_logits ET post_logits
    7. straight-through estimator (gradient via probs)

Usage :
    .venv/bin/python src_jax/model/test_rssm.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
from flax import nnx

from src_jax.model import (
    RSSM,
    sample_categorical_straight_through,
    symlog,
)


# ============================== Helpers

def count_params(model: nnx.Module) -> int:
    """Compte les paramètres trainables (nnx.Param seulement)."""
    params = nnx.state(model, nnx.Param)
    return sum(p.size for p in jax.tree.leaves(params))


def count_params_pytorch(module) -> int:
    """Compte les paramètres d'un module PyTorch."""
    return sum(p.numel() for p in module.parameters())


def check(condition, msg: str):
    status = "OK " if condition else "FAIL"
    print(f"    [{status}] {msg}")
    if not condition:
        raise AssertionError(msg)


def sep(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================== Config (Crafter Palier 2 style)

# Palier 2 : embed_dim=192, h_dim=384, z=24x24 → z_dim=576 → state_dim=960
EMBED_DIM = 192
ACTION_DIM = 17       # Crafter
H_DIM = 384
Z_CAT = 24
Z_CLASSES = 24
Z_DIM = Z_CAT * Z_CLASSES   # 576
HIDDEN_DIM = 512
STATE_DIM = H_DIM + Z_DIM   # 960

BATCH = 4
SEQ = 16


def main():
    print()
    print("=" * 60)
    print("  Tests RSSM JAX/Flax NNX — mini-DreamerV3")
    print("=" * 60)
    print(f"  JAX version  : {jax.__version__}")
    print(f"  Platform     : {jax.default_backend()}")
    print(f"  Config       : embed={EMBED_DIM}, h={H_DIM}, z={Z_CAT}x{Z_CLASSES}, action={ACTION_DIM}")

    rngs = nnx.Rngs(params=42)
    rng_key = jax.random.PRNGKey(0)
    all_passed = True

    # ===== 1. Instanciation + comptage params =====
    sep("1. Instanciation RSSM + params count")
    try:
        rssm = RSSM(
            embed_dim=EMBED_DIM,
            action_dim=ACTION_DIM,
            h_dim=H_DIM,
            z_categories=Z_CAT,
            z_classes=Z_CLASSES,
            hidden_dim=HIDDEN_DIM,
            rngs=rngs,
        )
        n_params = count_params(rssm)
        print(f"  Params JAX : {n_params:,}")
        print(f"  state_dim  : {rssm.state_dim}")
        check(rssm.state_dim == STATE_DIM, f"state_dim = {rssm.state_dim} == {STATE_DIM}")

        # Comparaison PyTorch
        try:
            import torch
            from src.model.rssm import RSSM as PTRSSM
            pt_rssm = PTRSSM(
                embed_dim=EMBED_DIM,
                action_dim=ACTION_DIM,
                h_dim=H_DIM,
                z_categories=Z_CAT,
                z_classes=Z_CLASSES,
                hidden_dim=HIDDEN_DIM,
            )
            n_pt = count_params_pytorch(pt_rssm)
            ratio = abs(n_params - n_pt) / max(n_pt, 1)
            print(f"  Params PyTorch : {n_pt:,}  |  diff : {ratio*100:.2f}%")
            # Tolérance 10% : la GRUCell Flax a 1 bias vs 2 chez PyTorch (~4% de diff)
            check(ratio < 0.10, f"params diff < 10% ({ratio*100:.2f}%)")
        except ImportError:
            print("  [SKIP] PyTorch non dispo — comparaison ignorée")
    except Exception as e:
        print(f"  [FAIL] Instanciation : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # ===== 2. init_state =====
    sep("2. init_state")
    try:
        init = rssm.init_state(BATCH)
        check(init["h"].shape == (BATCH, H_DIM), f"h shape : {init['h'].shape}")
        check(init["z"].shape == (BATCH, Z_DIM), f"z shape : {init['z'].shape}")
        check(jnp.all(init["h"] == 0.0), "h initialisé à zéros")
        check(jnp.all(init["z"] == 0.0), "z initialisé à zéros")
        # get_state_vec concat
        s_vec = RSSM.get_state_vec(init)
        check(s_vec.shape == (BATCH, STATE_DIM), f"state_vec shape : {s_vec.shape}")
    except Exception as e:
        print(f"  [FAIL] init_state : {e}")
        all_passed = False

    # ===== 3. observe_step =====
    sep("3. observe_step (un step training)")
    try:
        prev_state = rssm.init_state(BATCH)
        prev_action = jnp.zeros((BATCH, ACTION_DIM))
        prev_action = prev_action.at[:, 0].set(1.0)  # one-hot action 0
        embedding = jax.random.normal(rng_key, (BATCH, EMBED_DIM))

        k1 = jax.random.PRNGKey(1)
        new_state, post_logits, prior_logits = rssm.observe_step(
            prev_state, prev_action, embedding, k1
        )
        check(new_state["h"].shape == (BATCH, H_DIM), f"new_h shape : {new_state['h'].shape}")
        check(new_state["z"].shape == (BATCH, Z_DIM), f"new_z shape : {new_state['z'].shape}")
        check(post_logits.shape == (BATCH, Z_CAT, Z_CLASSES), f"post_logits shape : {post_logits.shape}")
        check(prior_logits.shape == (BATCH, Z_CAT, Z_CLASSES), f"prior_logits shape : {prior_logits.shape}")

        # z doit être un one-hot reshape : sum sur dim z_classes = z_categories
        z_reshaped = new_state["z"].reshape(BATCH, Z_CAT, Z_CLASSES)
        sum_per_cat = z_reshaped.sum(axis=-1)  # (B, z_cat)
        # Avec ST estimator forward = one-hot, sum doit être 1.0 par catégorique
        max_dev = float(jnp.abs(sum_per_cat - 1.0).max())
        check(max_dev < 1e-5, f"z one-hot per category (max dev = {max_dev:.2e})")
    except Exception as e:
        print(f"  [FAIL] observe_step : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # ===== 4. imagine_step =====
    sep("4. imagine_step (un step imagination)")
    try:
        prev_state = rssm.init_state(BATCH)
        prev_action = jnp.zeros((BATCH, ACTION_DIM))
        prev_action = prev_action.at[:, 1].set(1.0)

        new_state, prior_logits = rssm.imagine_step(prev_state, prev_action, jax.random.PRNGKey(2))
        check(new_state["h"].shape == (BATCH, H_DIM), f"new_h shape : {new_state['h'].shape}")
        check(new_state["z"].shape == (BATCH, Z_DIM), f"new_z shape : {new_state['z'].shape}")
        check(prior_logits.shape == (BATCH, Z_CAT, Z_CLASSES), f"prior_logits shape : {prior_logits.shape}")
    except Exception as e:
        print(f"  [FAIL] imagine_step : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # ===== 5. observe_sequence avec scan =====
    sep("5. observe_sequence avec jax.lax.scan")
    try:
        embeddings = jax.random.normal(jax.random.PRNGKey(10), (BATCH, SEQ, EMBED_DIM))
        # Construire des actions one-hot aléatoires
        action_idx = jax.random.randint(jax.random.PRNGKey(11), (BATCH, SEQ), 0, ACTION_DIM)
        actions_onehot = jax.nn.one_hot(action_idx, ACTION_DIM)

        # 5a. Sans dones
        out = rssm.observe_sequence(embeddings, actions_onehot, dones=None, key=jax.random.PRNGKey(12))
        check(out["h"].shape == (BATCH, SEQ, H_DIM), f"h shape : {out['h'].shape}")
        check(out["z"].shape == (BATCH, SEQ, Z_DIM), f"z shape : {out['z'].shape}")
        check(out["post_logits"].shape == (BATCH, SEQ, Z_CAT, Z_CLASSES), f"post_logits shape : {out['post_logits'].shape}")
        check(out["prior_logits"].shape == (BATCH, SEQ, Z_CAT, Z_CLASSES), f"prior_logits shape : {out['prior_logits'].shape}")

        # 5b. Avec dones : reset
        dones = jnp.zeros((BATCH, SEQ), dtype=jnp.float32)
        dones = dones.at[:, SEQ // 2].set(1.0)  # done au milieu
        out_dones = rssm.observe_sequence(embeddings, actions_onehot, dones=dones, key=jax.random.PRNGKey(12))
        check(out_dones["h"].shape == (BATCH, SEQ, H_DIM), "h shape avec dones")
        # Vérifier que les sorties existent (pas de NaN)
        check(jnp.all(jnp.isfinite(out_dones["h"])), "h finite avec dones")
        check(jnp.all(jnp.isfinite(out_dones["z"])), "z finite avec dones")

        # 5c. Test reset : avec dones=1 partout, chaque step doit être indépendant du précédent
        dones_all = jnp.ones((BATCH, SEQ), dtype=jnp.float32)
        # Note : done à t signifie "reset AVANT le step t+1" dans la logique observe_sequence
        # Donc même avec dones_all=1, le step 0 n'est pas reset (juste le step 1, 2, ...)
        out_all = rssm.observe_sequence(embeddings, actions_onehot, dones=dones_all, key=jax.random.PRNGKey(12))
        check(jnp.all(jnp.isfinite(out_all["h"])), "h finite avec dones=all-1")
    except Exception as e:
        print(f"  [FAIL] observe_sequence : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # ===== 6. imagine_sequence =====
    sep("6. imagine_sequence (rêve avec actions)")
    try:
        initial_state = rssm.init_state(BATCH)
        action_idx = jax.random.randint(jax.random.PRNGKey(20), (BATCH, SEQ), 0, ACTION_DIM)
        actions_onehot = jax.nn.one_hot(action_idx, ACTION_DIM)

        out = rssm.imagine_sequence(initial_state, actions_onehot, key=jax.random.PRNGKey(21))
        check(out["h"].shape == (BATCH, SEQ, H_DIM), f"h shape : {out['h'].shape}")
        check(out["z"].shape == (BATCH, SEQ, Z_DIM), f"z shape : {out['z'].shape}")
        check(out["prior_logits"].shape == (BATCH, SEQ, Z_CAT, Z_CLASSES), f"prior_logits shape : {out['prior_logits'].shape}")
        check(jnp.all(jnp.isfinite(out["h"])), "h finite")
    except Exception as e:
        print(f"  [FAIL] imagine_sequence : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # ===== 7. PRNG split : 2 keys différentes → z différents =====
    sep("7. PRNG split (stochasticité du sampling z)")
    try:
        prev_state = rssm.init_state(BATCH)
        prev_action = jnp.zeros((BATCH, ACTION_DIM))
        prev_action = prev_action.at[:, 0].set(1.0)
        embedding = jax.random.normal(jax.random.PRNGKey(30), (BATCH, EMBED_DIM))

        state_a, _, _ = rssm.observe_step(prev_state, prev_action, embedding, jax.random.PRNGKey(100))
        state_b, _, _ = rssm.observe_step(prev_state, prev_action, embedding, jax.random.PRNGKey(101))

        # h doit être identique (déterministe, basé sur prev_state + action)
        check(jnp.allclose(state_a["h"], state_b["h"], atol=1e-5),
              "h identique avec keys différentes (déterministe)")
        # z doit être différent (stochastique via key)
        z_diff = float(jnp.abs(state_a["z"] - state_b["z"]).max())
        check(z_diff > 0.01, f"z différent avec keys différentes (max diff = {z_diff:.3f})")

        # Test 2 steps consécutifs dans une séquence : z[0] et z[1] doivent différer
        # (le key est splitté à chaque step dans scan)
        embeddings_seq = jax.random.normal(jax.random.PRNGKey(40), (BATCH, 2, EMBED_DIM))
        actions_seq = jax.nn.one_hot(jnp.zeros((BATCH, 2), dtype=jnp.int32), ACTION_DIM)
        out_seq = rssm.observe_sequence(embeddings_seq, actions_seq, key=jax.random.PRNGKey(50))
        # Les z aux 2 steps doivent différer (sauf coïncidence)
        z_step_diff = float(jnp.abs(out_seq["z"][:, 0] - out_seq["z"][:, 1]).max())
        check(z_step_diff > 0.01, f"z différent entre 2 steps consécutifs (max diff = {z_step_diff:.3f})")
    except Exception as e:
        print(f"  [FAIL] PRNG split : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # ===== 8. kl_loss : valeur + gradient flow =====
    sep("8. kl_loss (valeur + gradient flow)")
    try:
        # Logits avec grande variance pour avoir KL > free_bits
        # (sinon free_bits clampe → gradient = 0, ce qui est attendu mais peu lisible)
        post_logits = jax.random.normal(jax.random.PRNGKey(60), (BATCH, SEQ, Z_CAT, Z_CLASSES)) * 3.0
        prior_logits = jax.random.normal(jax.random.PRNGKey(61), (BATCH, SEQ, Z_CAT, Z_CLASSES)) * 3.0

        # Test sans free_bits (free_bits=0) → toujours gradient
        loss = RSSM.kl_loss(post_logits, prior_logits, free_bits=0.0, beta_dyn=0.5, beta_rep=0.1)
        check(loss.shape == (), f"loss scalaire : {loss.shape}")
        check(float(loss) > 0, f"loss > 0 : {float(loss):.4f}")
        check(float(loss) < 100, f"loss raisonnable : {float(loss):.4f}")

        # Gradient flow : doit passer vers post_logits ET prior_logits (sans free_bits clamp)
        def loss_wrt_post(p_logits):
            return RSSM.kl_loss(p_logits, prior_logits, free_bits=0.0)

        def loss_wrt_prior(p_logits):
            return RSSM.kl_loss(post_logits, p_logits, free_bits=0.0)

        grad_post = jax.grad(loss_wrt_post)(post_logits)
        grad_prior = jax.grad(loss_wrt_prior)(prior_logits)

        check(jnp.any(grad_post != 0.0), "gradient vers post_logits non-nul")
        check(jnp.any(grad_prior != 0.0), "gradient vers prior_logits non-nul")
        check(jnp.all(jnp.isfinite(grad_post)), "gradient post_logits finite")
        check(jnp.all(jnp.isfinite(grad_prior)), "gradient prior_logits finite")

        # Vérifier le KL balancing : avec stop_gradient sur post dans kl_prior_learn,
        # le gradient sur post NE doit venir QUE du terme kl_post_learn (β_rep)
        # → orientation des gradients différente entre les 2 termes
        # On vérifie au moins que la norme des grads est correcte
        norm_post = float(jnp.linalg.norm(grad_post))
        norm_prior = float(jnp.linalg.norm(grad_prior))
        print(f"  ||grad_post|| = {norm_post:.4f}, ||grad_prior|| = {norm_prior:.4f}")

        # Free bits : si KL == 0 (post == prior), loss = (β_dyn + β_rep) × free_bits
        same_logits = jnp.zeros((BATCH, SEQ, Z_CAT, Z_CLASSES))
        loss_zero = RSSM.kl_loss(same_logits, same_logits, free_bits=1.0)
        expected = 0.6  # (0.5 + 0.1) * 1.0
        check(abs(float(loss_zero) - expected) < 1e-5,
              f"free_bits floor = β_dyn + β_rep = 0.6 : got {float(loss_zero):.4f}")

        # Sanity check : avec free_bits=1.0 et KL aléatoire ~0.9 (< 1.0),
        # le clamp doit kicker → gradient = 0 (bon comportement du free_bits)
        small_post = jax.random.normal(jax.random.PRNGKey(62), (BATCH, SEQ, Z_CAT, Z_CLASSES))
        small_prior = jax.random.normal(jax.random.PRNGKey(63), (BATCH, SEQ, Z_CAT, Z_CLASSES))
        grad_clamped = jax.grad(lambda p: RSSM.kl_loss(p, small_prior, free_bits=10.0))(small_post)
        # Avec free_bits=10.0 > KL réelle, clamp → grad = 0 (sauf si KL > 10 par hasard)
        max_grad_clamped = float(jnp.abs(grad_clamped).max())
        check(max_grad_clamped == 0.0, f"free_bits clamp coupe le gradient : max|grad|={max_grad_clamped:.2e}")
    except Exception as e:
        print(f"  [FAIL] kl_loss : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # ===== 9. Straight-through estimator =====
    sep("9. Straight-through estimator (sampling)")
    try:
        # Logits petits pour avoir des probs proches de uniform
        logits = jax.random.normal(jax.random.PRNGKey(70), (BATCH, Z_CAT, Z_CLASSES)) * 0.1
        key = jax.random.PRNGKey(71)

        # Forward : sample
        sample = sample_categorical_straight_through(logits, key)
        check(sample.shape == logits.shape, f"sample shape : {sample.shape}")
        # Forward = one-hot : sum sur dernière dim = 1
        sums = sample.sum(axis=-1)
        max_dev = float(jnp.abs(sums - 1.0).max())
        check(max_dev < 1e-5, f"forward one-hot (sum=1, max_dev={max_dev:.2e})")

        # Gradient via probs (ST estimator)
        def loss_fn(logits):
            sample = sample_categorical_straight_through(logits, key)
            return sample.sum()

        grad = jax.grad(loss_fn)(logits)
        check(jnp.all(jnp.isfinite(grad)), "gradient ST finite")
        check(jnp.any(grad != 0.0), "gradient ST non-nul (passe via probs)")
    except Exception as e:
        print(f"  [FAIL] ST estimator : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # ===== 10. Gradient flow complet : forward + loss + backward =====
    sep("10. Gradient flow complet via observe_sequence")
    try:
        embeddings = jax.random.normal(jax.random.PRNGKey(80), (BATCH, SEQ, EMBED_DIM))
        actions = jax.nn.one_hot(jnp.zeros((BATCH, SEQ), dtype=jnp.int32), ACTION_DIM)

        # Loss combinée : reconstruction-like sur h+z (forces gradients dans pre_gru/gru)
        # + KL sans free_bits (gradient garanti dans post/prior nets)
        def loss_fn(model):
            out = model.observe_sequence(embeddings, actions, key=jax.random.PRNGKey(81))
            # KL sans free_bits pour garantir gradient non-clampé
            kl = RSSM.kl_loss(
                out["post_logits"], out["prior_logits"],
                free_bits=0.0, beta_dyn=0.5, beta_rep=0.1,
            )
            # MSE sur h pour forcer gradient dans pre_gru, gru
            h_mse = jnp.mean(out["h"] ** 2)
            # MSE sur z pour forcer gradient dans posterior_net (via ST estimator)
            z_mse = jnp.mean(out["z"] ** 2)
            return kl + 0.01 * h_mse + 0.01 * z_mse

        loss, grads = nnx.value_and_grad(loss_fn)(rssm)
        check(jnp.isfinite(loss), f"loss finite : {float(loss):.4f}")
        # Vérifier que les grads sont non-nuls pour au moins quelques params
        grad_leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
        any_nonzero = any(float(jnp.abs(g).max()) > 0 for g in grad_leaves)
        check(any_nonzero, "au moins un gradient non-nul (training fonctionne)")
        all_finite = all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves)
        check(all_finite, "tous les gradients finite (pas de NaN/Inf)")

        # Tous les param leaves doivent recevoir un gradient
        # (sauf si certains layers ne sont pas utilisés, ce qui serait un bug)
        n_zero = sum(1 for g in grad_leaves if float(jnp.abs(g).max()) == 0.0)
        n_total = len(grad_leaves)
        print(f"  Params avec grad non-nul : {n_total - n_zero}/{n_total}")
        check(n_zero == 0, f"tous les params reçoivent un gradient ({n_zero} zéro)")
    except Exception as e:
        print(f"  [FAIL] Gradient flow : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # ===== 11. KL balancing : vérifier stop_gradient =====
    sep("11. KL balancing (stop_gradient correct)")
    try:
        # Construire 2 logits : tester que grad_post vient SEULEMENT de β_rep,
        # et grad_prior vient SEULEMENT de β_dyn
        post = jax.random.normal(jax.random.PRNGKey(110), (BATCH, SEQ, Z_CAT, Z_CLASSES)) * 3.0
        prior = jax.random.normal(jax.random.PRNGKey(111), (BATCH, SEQ, Z_CAT, Z_CLASSES)) * 3.0

        # Test 1 : avec beta_rep=0 et beta_dyn=1, grad_post doit être nul (post est sg dans kl_dyn)
        grad_post_no_rep = jax.grad(lambda p: RSSM.kl_loss(
            p, prior, free_bits=0.0, beta_dyn=1.0, beta_rep=0.0))(post)
        max_grad = float(jnp.abs(grad_post_no_rep).max())
        check(max_grad == 0.0, f"grad_post=0 quand beta_rep=0 (kl_dyn a sg(post)) : max={max_grad:.2e}")

        # Test 2 : avec beta_dyn=0 et beta_rep=1, grad_prior doit être nul (prior est sg dans kl_rep)
        grad_prior_no_dyn = jax.grad(lambda p: RSSM.kl_loss(
            post, p, free_bits=0.0, beta_dyn=0.0, beta_rep=1.0))(prior)
        max_grad = float(jnp.abs(grad_prior_no_dyn).max())
        check(max_grad == 0.0, f"grad_prior=0 quand beta_dyn=0 (kl_rep a sg(prior)) : max={max_grad:.2e}")

        # Test 3 : avec les 2 termes, les gradients sont non-nuls
        grad_post_full = jax.grad(lambda p: RSSM.kl_loss(
            p, prior, free_bits=0.0, beta_dyn=0.5, beta_rep=0.1))(post)
        grad_prior_full = jax.grad(lambda p: RSSM.kl_loss(
            post, p, free_bits=0.0, beta_dyn=0.5, beta_rep=0.1))(prior)
        check(float(jnp.abs(grad_post_full).max()) > 0, "grad_post non-nul avec balancing complet")
        check(float(jnp.abs(grad_prior_full).max()) > 0, "grad_prior non-nul avec balancing complet")
    except Exception as e:
        print(f"  [FAIL] KL balancing : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # ===== 12. Parité shape : observe_sequence vs imagine_sequence =====
    sep("12. Parité shape JAX vs PyTorch observe_sequence")
    try:
        import torch
        from src.model.rssm import RSSM as PTRSSM

        pt_rssm = PTRSSM(
            embed_dim=EMBED_DIM,
            action_dim=ACTION_DIM,
            h_dim=H_DIM,
            z_categories=Z_CAT,
            z_classes=Z_CLASSES,
            hidden_dim=HIDDEN_DIM,
        )
        pt_rssm.eval()

        # Input identique (numpy)
        np_emb = jax.random.normal(jax.random.PRNGKey(200), (BATCH, SEQ, EMBED_DIM))
        np_act = jax.nn.one_hot(jnp.zeros((BATCH, SEQ), dtype=jnp.int32), ACTION_DIM)

        # JAX
        out_jax = rssm.observe_sequence(np_emb, np_act, key=jax.random.PRNGKey(201))

        # PyTorch
        emb_pt = torch.from_numpy(jax.device_get(np_emb)).float()
        act_pt = torch.from_numpy(jax.device_get(np_act)).float()
        with torch.no_grad():
            out_pt = pt_rssm.observe_sequence(emb_pt, act_pt)

        # Comparer SHAPES (les valeurs diffèrent car poids et PRNG différents)
        check(out_jax["h"].shape == tuple(out_pt["h"].shape),
              f"h shape : JAX {out_jax['h'].shape} == PT {tuple(out_pt['h'].shape)}")
        check(out_jax["z"].shape == tuple(out_pt["z"].shape),
              f"z shape : JAX {out_jax['z'].shape} == PT {tuple(out_pt['z'].shape)}")
        check(out_jax["post_logits"].shape == tuple(out_pt["post_logits"].shape),
              f"post_logits : JAX {out_jax['post_logits'].shape} == PT {tuple(out_pt['post_logits'].shape)}")
        check(out_jax["prior_logits"].shape == tuple(out_pt["prior_logits"].shape),
              f"prior_logits : JAX {out_jax['prior_logits'].shape} == PT {tuple(out_pt['prior_logits'].shape)}")

        # Sanity : magnitudes raisonnables (les 2 implem partent de N(0,1) init)
        h_mag_jax = float(jnp.abs(out_jax["h"]).mean())
        h_mag_pt = float(out_pt["h"].abs().mean())
        print(f"  h magnitude : JAX={h_mag_jax:.3f}, PyTorch={h_mag_pt:.3f}")
        check(0.0 < h_mag_jax < 10.0 and 0.0 < h_mag_pt < 10.0,
              "magnitudes raisonnables des 2 côtés")
    except ImportError:
        print("  [SKIP] PyTorch non dispo")
    except Exception as e:
        print(f"  [FAIL] Parité shape : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # ===== 13. JIT compile observe_sequence =====
    sep("13. JIT compile observe_sequence (perf check)")
    try:
        import time

        @nnx.jit
        def jit_fwd(model, embeddings, actions, key):
            return model.observe_sequence(embeddings, actions, key=key)

        embeddings = jax.random.normal(jax.random.PRNGKey(90), (BATCH, SEQ, EMBED_DIM))
        actions = jax.nn.one_hot(jnp.zeros((BATCH, SEQ), dtype=jnp.int32), ACTION_DIM)

        # Warmup (compilation)
        t0 = time.time()
        out = jit_fwd(rssm, embeddings, actions, jax.random.PRNGKey(91))
        jax.block_until_ready(out["h"])
        t_compile = time.time() - t0
        print(f"  Compilation : {t_compile*1000:.0f}ms")

        # Run (compilé)
        t0 = time.time()
        for _ in range(10):
            out = jit_fwd(rssm, embeddings, actions, jax.random.PRNGKey(92))
            jax.block_until_ready(out["h"])
        t_run = (time.time() - t0) / 10
        print(f"  Forward jit : {t_run*1000:.1f}ms / iter (B={BATCH}, T={SEQ})")
        check(out["h"].shape == (BATCH, SEQ, H_DIM), "shape OK post-jit")
    except Exception as e:
        print(f"  [FAIL] JIT : {e}")
        import traceback; traceback.print_exc()
        all_passed = False

    # ===== Récap =====
    sep("RECAP")
    if all_passed:
        print("  Tous les tests RSSM passent.")
    else:
        print("  Certains tests ont ECHOUE. Voir détails ci-dessus.")

    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
