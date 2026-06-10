# Benchmark des runs DreamerV3 JAX sur Crafter

> **Objectif** : tracker l'impact mesuré de chaque changement.
> Permet d'identifier rapidement ce qui aide vs ce qui casse.

## Cible de référence

- Paper DreamerV3-XL (200M params) : **score 14.5 / ~11.7 achievements**
- Paper DreamerV3-S (12M params) : **~7-8 achievements** (estimation Fig 5)
- Notre archi : **15.26M params** (entre S et M, plus proche S)
- Cible réaliste pour 15M : **6-8 achievements**
- Baseline Rainbow (DQN) : 4.3 score / ~3-4 ach
- Baseline random : ~2 ach

## Hyperparams qu'on a testés

| Hyperparam | Paper officiel | Notre v13 (full paper) | **Notre v17 (post-audit)** |
|---|---|---|---|
| LR_WM | 4e-5 | 1e-4 | 1e-4 |
| LR_AC | 3e-5 | 1e-4 | 1e-4 (H_309 ouvert : 3e-5 ?) |
| entropy_coef | **3e-4 fixe** | 3e-4 (mais AVANT fix stop_gradient) | **3e-4 fixe, adaptive OFF** ✓ |
| adaptive_alpha | n'existe pas (aucune ref) | OFF | **OFF par défaut** ✓ |
| GRAD_CLIP | AGC 0.3 | 1.0 | 1.0 |
| BATCH_SIZE | 16 | 16 | 16 (wrapper aussi, était 32) |
| SEQ_LEN | 64 | 64 | 64 |
| Free bits | 1 nat / step (KL sommée) | par catégorie = 24 nats ✗ | **par step (sum puis clamp)** ✓ |
| Recon loss | sum(pixels).mean() | sum().mean() ✓ | sum().mean() ✓ |
| Return scale | max(1, p95-p5) sans cap | clip(., 1, 5) ✗ | **max(1, range)** ✓ |
| Buffer | séquences mono-trajectoire | **interleavé 16 envs** ✗✗ | **per-env** ✓ |
| Bootstrap returns | fast critic | slow critic ✗ | slow (H_309 ouvert) |
| GAMMA | 0.997 | 0.99 | 0.99 (H_310 ouvert) |
| train_ratio | 512 | 32 (commentaire mentait) | 32 (H_308 ouvert, flags forwardés) |
| n_envs | 1 | 16 | 16 |
| Actor unimix / zero-init / SiLU / slowreg | OUI | OUI ✓ | OUI ✓ |

## Chronologie des runs (DreamerV3 JAX sur Crafter, archi 15M)

### v1-v3 : baseline initial (sans fixes)
- v3 = meilleur baseline : **2.70 ach @ iter 2000** (peak)
- Hyperparams : LR 3e-4 partout, entropy_coef 0.005, batch 32, seq 50, n_envs 8

### v4 : buffer 500k FIFO simple
- **Pas de gain net** : variance possible, similaire v3
- Insight : pour 30k iter, 50k FIFO suffit
- Verdict : NEUTRE

### v5 : replay 4× (WM_TRAIN_PER_ITER=4)
- Peak 0.20 ach @ iter 1000 (PIRE que baseline)
- Insight : train_ratio bas (~40) = overfitting batches récents
- Paper Crafter utilise train_ratio 512 (1 train per 512 env_steps)
- Verdict : RÉGRESSION

### v6 : config "paper" partial (batch 16, seq 64, collect 32)
- Peak 1.10 ach
- Verdict : RÉGRESSION vs v3 (LR_AC encore 3e-5, mauvais)

### v7 : LR_AC 3e-4 (fix LR)
- iter 2000 : 1.50 ach (mieux que v6 mais moins que v3)
- Verdict : NEUTRE (mix fixes incomplets)

### v8 : LR fix + H_target schedule 1.5→0.8
- iter 1000 : 1.60 / iter 2000 : 0.40 / iter 3000 : 1.30
- Insight : H_target descend → encourage exploitation → mode collapse sleep
- Verdict : RÉGRESSION

### v9 : H_target constant 1.13 + anti-spam sleep
- Peak 2.10 @ iter 2000 (proche v3)
- iter 3000 régression
- Verdict : NEUTRE

### v10 : LR_ALPHA 1e-3 + spam softer
- iter 1000 : 0.00 (CASSÉ — oscillations α trop fortes)
- Insight : LR_ALPHA 1e-3 trop haut → adaptive_alpha instable
- Verdict : RÉGRESSION

### v11 : Fixes parité numérique (encoder NHWC + GRU bias_hh + LayerNorm eps)
- iter 1000 : 1.00 ach
- Modules JAX maintenant mathematically equivalents PyTorch
- Mais : modules en parité ≠ training convergent
- Verdict : NEUTRE sur score, CRUCIAL pour correctness

### v12 : 3 FIXES CRITIQUES
- Fix #1 : recon loss .mean() → .sum(pixels).mean() (× 12288 scale shift)
- Fix #2 : KL clamp PER-STEP (vs mean puis clamp)
- Fix #3 : LR 1e-4, GRAD_CLIP 1.0 (au lieu de 100.0)
- Résultats :
  - iter 750 : 1.40 / iter 3000 : 1.30 / iter 3250 : 1.60 (peak)
  - **Recon descend de 598 → 55** ← le WM apprend ENFIN
  - Mais EVAL oscille, actor toujours random (H stuck ~2.7)
- Insight : Le **WM est réparé** (recon converge comme paper), mais **l'actor ne suit pas**
- Verdict : PROGRÈS MAJEUR (sur WM), mais incomplet (actor)

### v13 : 5 FIXES ACTOR (P4-P5)
- P4 : actor unimix 0.01
- P5a : zero init reward + critic heads
- P5b : entropy_coef = 3e-4 paper fixe, adaptive_alpha OFF, auto_explore OFF
- P5c : ELU → SiLU partout
- P5d : slowreg 1.0 sur critic loss
- Résultats (jusqu'à iter 1250) :
  - iter 750 : 0.30 / iter 1000 : 0.20 / iter 1250 : 0.80 (score = -15.56 !)
  - Score très négatif = sleep spam cumulé (anti-spam env amplifie)
- Insight : entropy_coef 3e-4 PAPER trop bas pour notre setup avec anti-spam env
  - Paper n'a pas anti-spam → entropy 3e-4 OK
  - Nous : anti-spam interfère, actor figé sur sleep
- Verdict : RÉGRESSION vs v12

## Param-by-param impact analysis

### LR (learning rate)
| Valeur testée | Effet observé |
|---|---|
| 3e-4 WM + 3e-4 AC | v3 baseline : 2.70 peak. WM stagne (recon 0.01) mais actor produit du signal. |
| 1e-4 + 1e-4 | v12+v13 : WM converge bien si recon-sum activée. Plus stable. |
| 1e-4 + 3e-5 | v6/v7 : actor trop lent, RÉGRESSION |
| Paper recommande 4e-5 | Non testé, plus conservateur |

→ **Sweet spot : 1e-4 partout** avec recon-sum fix.

### entropy_coef
| Valeur testée | Effet observé |
|---|---|
| 0.005 + adaptive_alpha + auto_explore | v3 : meilleur baseline. Trop fort mais système compense. |
| 3e-4 paper fixe | v13 : actor reste random, tombe dans sleep spam |
| 0.001 effectif (3e-4 × 5 boost auto_explore) | v3-v9 : OK |

→ **3e-4 paper TROP FAIBLE pour notre env modifié.** Avec anti-spam, besoin plus de pression exploration. Soit 0.005, soit réactiver adaptive_alpha.

### Recon loss formula (CRITIQUE)
| Valeur | Effet |
|---|---|
| `.mean()` global (B,T,C,H,W) | v1-v11 : recon 0.01, WM stagne. Decoder n'apprend pas. |
| `.sum(pixels).mean(B,T)` | v12+v13 : recon 600 → 55, WM apprend, paper-aligned |

→ **BUG TROUVÉ. Sum over pixels obligatoire.** Le plus gros gain de tout le projet.

### KL free_bits clamp position
| Valeur | Effet |
|---|---|
| Global (clamp after mean) | v1-v11 : posterior collapse partiel possible |
| Per-step (clamp before mean) | v12+v13 : aligné paper officiel |

→ **Per-step est paper-exact.**

### Buffer capacity
| Valeur | Effet |
|---|---|
| 50k | v3 : OK, forgetting attendu sur long runs |
| 500k | v4+ : OK, plus de marge anti-forgetting |
| 1M | Non testé |

→ **500k = safe choice.**

### GRAD_CLIP
| Valeur | Effet |
|---|---|
| 100.0 (global norm) | v1-v11 : clip jamais déclenché, gradients explosent |
| 1.0 (global norm) | v12+ : stable, mieux que paper AGC 0.3 |
| AGC 0.3 (per-param) | Non testé, paper-officiel |

→ **1.0 OK, AGC potentially better.**

### Actor unimix
| Valeur | Effet |
|---|---|
| Pas d'unimix | v1-v12 : actor peut devenir déterministe instable |
| 0.01 | v13 : actor a 1% uniforme, mais entropy_coef trop bas masque l'effet |

→ **À garder, mais effet visible seulement si entropy adapté.**

### Zero init heads (reward + critic)
| Valeur | Effet |
|---|---|
| Init standard | v1-v12 : critic bruit au début, instable |
| Zero init | v13 : critic.out.kernel ~5e-4 après 5 iter (parti de zéro) |

→ **À garder, paper-aligned.**

### Activation
| Valeur | Effet |
|---|---|
| ELU | v1-v12 |
| SiLU | v13 (paper) |

→ Effet difficile à isoler (v13 a 5 changements). À garder, paper-aligned.

### Slowreg critic
| Valeur | Effet |
|---|---|
| 0.0 (absent) | v1-v12 |
| 1.0 (paper) | v13 |

→ Effet difficile à isoler.

### Anti-spam env (sleep/noop penalty)
| Valeur | Effet |
|---|---|
| 0.0 (off) | Default Crafter, paper-compatible |
| 0.02 par step (Phase 12+) | v9-v13 : interfère avec entropy_coef paper |
| 0.05 par step | Phase 12 : trop agressif |

→ **À DÉSACTIVER si on veut matcher paper hyperparams.** Trop intrusif sur le signal.

## Décisions pour v14 (next run)

Hypothèse : **v13 a régressé car entropy_coef 3e-4 paper + anti-spam env = mauvais cocktail**.

Plan v14 :
- Garder TOUS les fixes architecturaux (P4, P5a, P5c, P5d)
- REVERT entropy_coef à 0.005 (notre baseline qui marchait)
- DÉSACTIVER anti-spam env (SPAM_PENALTY = 0.0)
- Permettre adaptive_alpha + auto_explore (filets de sécurité)

Variables changées d'un coup en v14 : 3 (entropy_coef, anti-spam, adaptive). Pas idéal mais pragmatique vu le temps.

Pour cleaner approach : 
- v14a : juste enlever anti-spam
- v14b : v14a + entropy_coef 0.005
- v14c : v14b + adaptive_alpha
- Mais 3 runs coûte ~$3 et 2h chacun = 6h total. Choix Julian.

## Insights généraux

1. **Le plus gros bug du projet** : recon loss mean() au lieu de sum(pixels).mean() — 12288× sous-évaluée. Sans ça, le WM ne s'entraînait jamais vraiment.

2. **L'audit paper-exact paie** : cloner danijar/dreamerv3 et diff ligne-à-ligne a révélé les vrais bugs. Notre PyTorch baseline avait les mêmes bugs (sum-vs-mean) donc on ne pouvait pas les attraper en comparant juste avec lui.

3. **Notre PyTorch n'était pas paper-exact non plus**. La parité numérique PyTorch ↔ JAX ne suffit PAS si PyTorch lui-même diverge du paper.

4. **Empiler les fixes sans isoler** = brouillard total. Pour v14+, idéalement isoler UN changement à la fois.

5. **L'anti-spam env est notre invention**, pas paper. Interfère avec les hyperparams paper. À DÉSACTIVER pour comparer fairment.

6. **Le model 15M devrait atteindre 5-7 achievements** d'après le paper. Notre meilleur reste 2.70 → on a un gap de 2-3×.

## Statut actuel

- Modules JAX : ✓ paper-exact (encoder, RSSM, decoder, heads, actor, critic)
- WM training : ✓ converge (recon descend de 600 → 55)
- Actor training : ✗ ne converge pas (H stuck haut, EVAL oscille 0-1.6)
- Cause restante probable : anti-spam env interfère, entropy_coef pas adapté

## Explications théoriques des 4 patterns "qui cassent" (Investigation Agent B)

### #1 — LR_ALPHA = 1e-3 cause oscillations (v10)

**Théorie** : SAC adaptive temperature est une optimisation duale couplée. α et π forment un système avec délai. Si α bouge plus vite que π ne peut s'adapter, on a une boucle PI mal tunée → oscillations entre policy uniforme (α=1.0) et déterministe (α=5e-5).

**Règle empirique** : `LR_ALPHA ≤ LR_AC` (souvent 1× à 3× plus bas).
- v10 avait LR_ALPHA=1e-3 et LR_AC=1e-4 → ratio 10× → instable.

**Diagnostic rapide** : tracker `std(log_alpha)` sur 100 iter. Si > 0.5 → oscillation.

### #2 — train_ratio trop bas (~40) écroule l'apprentissage (v5)

**Théorie** : train_ratio bas = buffer dominé par policy courante. Le WM **overfit la policy** au lieu d'apprendre la dynamique du monde. C'est la "Sutton's deadly triad" (off-policy + bootstrap + non-stationnarité).

**Règle** : `train_ratio = (batch_size × seq_len × wm_train_per_iter) / (n_envs × collect_per_iter)`
- Doit être > 100 pour model-based RL.
- Paper Crafter : 512.

### #3 — H_target descendant 1.5→0.8 cause mode collapse (v8)

**Théorie** : Crafter a 22 achievements en chaîne hiérarchique. Decay H_target force exploitation **avant** que l'agent ait découvert la reward landscape. → **Premature commitment** sur un optimum local (sleep spam).

**Règle** : Si environnement pas résolu, NE PAS baisser la pression d'exploration.
- Paper DreamerV3 utilise `entropy_coef = 3e-4` CONSTANT (pas de schedule).
- Pour Crafter, considérer que les 20-50k premiers iter sont "early training" peu importe le calendrier.

### #4 — entropy_coef 3e-4 paper + anti-spam env = cocktail létal (v13)

**Théorie** : `entropy_coef=3e-4` du paper assume une reward "propre" + return normalization paper. Notre penalty cumulative anti-spam injecte du **bruit** dans la distribution des returns. Les percentiles 5/95 (utilisés pour normaliser advantages) deviennent dominés par penalties, pas achievements → signal achievement noyé → agent reste random pour minimiser variance penalty.

**Règle** : Un changement env (reward shaping) + un hparam paper-style = JAMAIS en même temps.
- Soit anti-spam OFF + entropy paper.
- Soit anti-spam ON + entropy boosté (0.005+) pour compenser le bruit.

## Méta-leçons générales

1. **Coupling matters** : SAC adaptive (α↔π), train_ratio (collect↔train), reward shaping (penalty↔entropy) sont des systèmes couplés. Tuner un côté sans l'autre = oscillations ou collapse.

2. **Schedule = a priori fort** : tout schedule (LR decay, H_target decay, entropy decay) encode "je sais où on en est". Sur sparse-reward chain envs comme Crafter, cet a priori est presque toujours faux à court horizon.

3. **Le paper est un système** : les hyperparams paper sont co-tunés avec normalisations + archi + env exact. Picker une seule valeur paper en gardant le reste custom est plus risqué qu'utiliser un set custom cohérent.

## v15 — 🎯 BREAKTHROUGH : stop_gradient sur state_vec dans imagine

**Statut : SUCCÈS. Le bug final identifié et fixé.**

### Le bug

Dans `imagine_trajectory.step_fn`, le `state_vec = concat([h, z])` était passé à `actor.get_dist()` SANS stop_gradient. Le gradient policy remontait via la chaîne RSSM (GRU + sample STE × 16 imagination steps) → bruit énorme dans le PG signal → actor restait random.

Aligné avec `danijar/dreamerv3` `agent.py:196` qui fait `sg(imgfeat)` pour enrober TOUTE l'imagination.

### Fix (1 ligne)

```python
# train_dreamer_jax.py:356 (dans imagine_trajectory.step_fn)
state_vec = jnp.concatenate([state["h"], state["z"]], axis=-1)
state_vec_sg = jax.lax.stop_gradient(state_vec)  # NEW
dist = actor.get_dist(state_vec_sg)  # CHANGED: sg input
```

### Résultats v15 (4000 iter, ~$0.7)

| Iter | EVAL ach | H entropy | Notes |
|---|---|---|---|
| 500  | 0.00 | 2.83 | early, normal |
| 1000 | 0.00 | 2.81 | début apprentissage WM |
| 1500 | 0.00 | 2.80 | |
| 2000 | 0.00 | 2.80 | |
| 2500 | 0.00 | 2.26 | H descend pour la 1ère fois |
| 3000 | **0.50** | 2.37 | Progress +0.30 auto_explore |
| 3500 | **1.90** ⭐ | 1.34 | Progress +1.40 ! Actor exploite |
| 4000 | **2.00** ⭐ | 0.95 | Trajectoire monte encore |

### Comparaison historique

| Run | Best peak | Iter peak | H final |
|---|---|---|---|
| v3 (baseline random) | 2.70 | 2000 | ~2.7 |
| v12 (3 fixes WM) | 1.60 | 3250 | ~2.8 |
| v14 (tous fixes paper) | 0.80 | 3500 | ~2.8 → 1.8 |
| **v15 (+ stop_gradient)** | **2.00 ⭐** | **4000** | **0.95** (exploite !) |

À iter 4000, **v15 montait encore**. v3 baseline est dépassé en trajectoire (sans même avoir atteint l'asymptote).

### Diagnostic

C'était LE bug. Symptôme parfaitement aligné :
- ✓ WM convergeait similairement au paper (recon, KL ne dépendent pas du sample STE)
- ✗ Actor restait random (gradient PG noyé dans le bruit STE multi-step)

Le fix correspond exactement à `ac_grads: False` du config Crafter du paper.

### Action suivante : long run final 30k iter

```bash
.venv/bin/modal run --detach crafter_dreamer/scripts/modal_train_jax.py::main \
    --train-iter 30000 \
    --eval-interval 2000 \
    --batch-size 16 \
    --run-name v16_final_30k_paper_aligned \
    --no-use-rnd
```

Coût ~$7, durée ~5h. Cible : **5-8 ach** (paper-S level).

---

## v16 — Run 2000 iter, PLAT (et c'était mathématiquement prévu)

**Statut : ÉCHEC INFORMATIF. A permis de prouver la dynamique alpha.**

Config = v15 sauf 2 changements faits "pour la stabilité" : `LR_ALPHA 1e-3 → 1e-4`, `auto_explore ON → OFF`. Run de 2000 iter seulement.

### Résultats

| Iter | EVAL ach | H | α | Notes |
|---|---|---|---|---|
| 250 | 0.30 | 2.83 | 0.0049 | WM converge (rec 672→132) |
| 500 | 0.00 | 2.83 | 0.0048 | critic colle (val≈ret) |
| 750 | 0.30 | 2.83 | 0.0046 | H toujours au max |
| 850+ | — | 2.83 | 0.0046 | actor 100% random |

### Le diagnostic (audit 46 agents, vérifié sur 5 points de logs)

**Adam sur le scalaire `log_alpha` avec gradient constant = sign-step** : `d(log α)/dt = -LR_ALPHA` exactement, quelle que soit la magnitude du gradient (1.70 ici). Donc :

```
α(t) = α_init × exp(-LR_ALPHA × t)
décollage quand α croise ~3e-4 (le coef entropie FIXE du paper)
t_décollage = ln(α_init / 3e-4) / LR_ALPHA
```

| Run | LR_ALPHA | t_décollage prédit | t observé |
|---|---|---|---|
| v15 | 1e-3 | ~2 800 iter | **~3 000** ✓ |
| v16 | 1e-4 | ~28 000 iter | plat à 2000 (cohérent) |

Le modèle prédit α aux 5 points de logs v16 à la précision d'affichage (0.0050→0.0049→0.0048→0.0046→0.0046). **Le "breakthrough" v15 n'était PAS H\* ni auto_explore : c'était α redescendant exponentiellement jusqu'à la valeur paper 3e-4.** auto_explore a même retardé v15 (boost ×1.5/1000 iter perd contre décroissance ×0.37/1000).

**v16 ne réfute donc rien** : trop court (2000 < 3000) ET freiné 10× (LR_ALPHA).

---

## AUDIT EXHAUSTIF post-v16 (46 agents, 6 lentilles + vérif adversariale)

### Les 3 causes racines confirmées

**1. L'alpha adaptatif est un détour inutile (CRITICAL)**
Aucune référence (danijar, symoon11 17.65 ach, NM512, sheeprl) n'utilise d'alpha adaptatif. Toutes : `entropy_coef` FIXE 3e-4. Notre mécanisme = attendre 3000 iter que α s'érode jusqu'à... la valeur où le paper démarre. Le test v13 de 3e-4 datait d'avant le fix stop_gradient → non probant.
→ **FIX : `ENTROPY_COEF=3e-4` fixe, `adaptive_alpha` OFF par défaut.** Décollage prédit ~700-1500 iter (borné par convergence WM/critic, plus par l'érosion d'α).

**2. Le buffer entrelaçait les 16 envs (CRITICAL)**
`buffer.add()` appelé dans `for i in range(n_envs)` + stockage plat + sampling contigu → chaque "séquence de 64 steps" donnée au RSSM changeait d'env À CHAQUE STEP (env0_t, env1_t, ..., env15_t, env0_t+1...). Le prior apprenait une dynamique inter-envs fictive. La recon convergeait quand même (posterior = fonction de la frame courante), ce qui masquait le bug. Explique le plafond ~2 ach de v15 : l'imagination roulait sur un modèle de dynamique mort.
→ **FIX : buffer per-env `(n_envs, per_env_cap)`, séquences mono-env.** Testé : mono-env + timesteps consécutifs + wrap FIFO sain.

**3. Free bits par catégorie au lieu de par step (CRITICAL)**
`maximum(kl, 1.0)` appliqué sur shape (B,T,24) = clamp PAR CATÉGORIE → free bits effectif 24 nats/step au lieu de 1. Preuve dans les logs : `kl=0.62 ≈ 0.5×1.0 + 0.1×1.0` = le floor exact → gradient KL ≈ 0 → le prior n'était presque pas entraîné. Double peine avec le bug 2.
→ **FIX : somme sur les 24 catégories AVANT le clamp** (KL jointe par step, comme l'officiel). Testé : floor exact à KL=0, gradient vivant sinon.

### Findings secondaires (fixés dans le même bundle)

| Finding | Fix appliqué |
|---|---|
| Cap scale 5.0 absent du paper | `maximum(range, 1.0)` sans cap |
| INIT_ALPHA=0.005 = 17× le seuil | 3e-4 (si adaptive réactivé) |
| LR_ALPHA=1e-4 (le faux fix v16) | revert 1e-3 + commentaire sign-step |
| Bug latent `args.h_target` brut dans boucle AC extra | `get_h_target(it, args)` |
| Wrapper Modal : batch=32, adaptive non forwardé 2 sens, seq_len fantôme | batch=16, forwarding explicite, params fantômes retirés, + `entropy_coef`/`wm_train_per_iter`/`ac_train_per_iter` forwardés |
| EVAL argmax aveugle sur policy uniforme | + 5 épisodes mode sample (`sample=X.XX` dans les logs) |
| Commentaires mensongers (l.69-70 "512 env_steps", rssm "aligné officiel") | corrigés |

### Findings confirmés NON fixés (prochains tests, à isoler)

| Finding | Sévérité | Levier |
|---|---|---|
| train_ratio réel 32 vs paper 512 (16× sous) | high | `--wm_train_per_iter`/`--ac_train_per_iter` (maintenant forwardés) |
| Bootstrap lambda-returns par slow critic (paper = fast) | medium | 1 ligne dans train_step_ac (l.602) |
| GAMMA 0.99 vs 0.997 paper | medium | returns lointains 4× plus faibles à t=200 |
| LR_AC 1e-4 vs 3e-5 officiel | low | — |
| Archi déséquilibrée (deter 384 vs 2048, hidden 768 vs 256) | medium | gros refactor, dernier recours |

### Réfutés par l'audit (ne plus suspecter)

- Continue head sans zero-init ✗ — decoder sigmoid vs symlog ✗ — unimix ✗ — warmup ✗ — sampling uniforme ✗ — structure actor loss (weight cumprod, adv/scale) : **conforme officiel** ✓

### v17 — le run de validation du bundle

```bash
.venv/bin/modal run --detach crafter_dreamer/scripts/modal_train_jax.py::main \
    --train-iter 4000 \
    --eval-interval 500 \
    --run-name v17_bundle_entropy_buffer_freebits \
    --no-use-rnd
```

(Tous les nouveaux défauts s'appliquent : entropy fixe 3e-4, adaptive OFF, batch 16, buffer per-env, free bits par step.)

| Iter | Cible | Interprétation |
|---|---|---|
| 1000 | H < 2.5 | l'actor sort de l'uniforme SANS attendre l'érosion d'α |
| 2000 | ≥ 2 ach (ou sample ≥ 2) | bat le plafond v15 à mi-course |
| 4000 | ≥ 3-4 ach, montant | bundle validé → run 30k |

Si H reste à 2.83 à iter 1500 malgré entropy 3e-4 → le frein n'était pas (que) l'entropie → suspecter le signal advantage (critic/returns) en priorité.

---

## v17 — Bundle audit : H libéré, mais collapse prématuré sur bruit

**Statut : échec informatif — a révélé le déséquilibre policy/WM.**

Les 3 signatures attendues du bundle sont là dès iter 50 :
- **H libéré** : 2.33 → 0.80 → 2.46 → 1.41 (vs 2.83 soudé pendant 2500 iter avant)
- **KL vivante** : ~3 nats (vs 0.62 = floor exact avant le fix free bits)
- **EVAL sample** : 1.40 @ 500 quand l'argmax voit encore 0.00

Mais ensuite : H s'effondre progressivement (0.38-0.44 @ 1050-1200), `sample` BAISSE (1.40 → 1.20), r/step devient négatif, val≈ret (advantages ≈ 0), argmax reste à 0.00 ach.

**Diagnostic** : collapse prématuré. La policy, libérée dès iter 0, se commit sur du **bruit** avant que le reward head ait du signal (~300 événements de reward dans 32k env_steps, le head prédit la marginale). v15 marchait *par accident* : les 3000 iter de frein alpha laissaient le WM mûrir avant la libération de la policy. → C'est H_308 (train_ratio 32 vs 512).

---

## v18 — 🏆 RECORD : wm_train_per_iter×4 → 4.00 ach @ iter 2000

**Statut : SUCCÈS MAJEUR. H_308 validée. Best du projet : 4.00 (×2 le plafond v15, ≈ Rainbow 4.3), en 64k env_steps.**

Config = v17 + `--wm-train-per-iter 4` (ratio replay 32 → 128). Un seul changement.

### Trajectoire EVAL — la hiérarchie Crafter se construit

| Iter | argmax | sample | Unlocked | Notes |
|---|---|---|---|---|
| 500 | 0.70 | 0.80 | sapling 60%, cow 10% | verrou sapling (H 0.23, health warns) |
| 1000 | 2.30 ↑ | 2.60 | + wake_up 100%, wood 20%, drink | **déverrouillage AUTO** |
| 1500 | 3.50 ↑ | 4.00 | + place_plant 100%, zombie 40% | la chaîne s'étend |
| 2000 | **4.00** ⭐ | 3.00 | **7/22** : wood 80% | record projet |
| 2500 | 2.30 ↓ | 3.40 | place_plant et zombie perdus | oscillation |
| 3000 | 2.00 ↓ | 2.60 | sapling/wake_up/wood | déclin confirmé |
| 3500+ | — | — | — | crédits Modal épuisés |

### Ce que v18 prouve

1. **H_308 VALIDÉE** : le train_ratio était le déverrouilleur. Le WM ×4 donne au reward head le signal à temps ; le verrou initial (sapling, H=0.23 @ 200) se desserre TOUT SEUL (mécanisme sain : advantage appris → s'éteint → exploration suivante).
2. **`scale` décolle pour la 1ère fois** (1.00 → 2.29, p95=1.84) — la normalisation percentile travaille ; le fix « sans cap » arrive au bon moment.
3. WM excellent : rec ~25-35, KL qui monte en continu (2.8 → 8.7 : les données s'enrichissent plus vite que le prior n'apprend — non pathologique mais à surveiller).

### Le problème restant : instabilité post-pic

Après le pic 4.00 @ 2000, déclin (2.30, 2.00) avec perte des achievements profonds. `sample > argmax` systématiquement → le mode argmax glisse, la policy stochastique reste meilleure. val/ret oscillent (0.56/0.45, 0.20/0.18...). Pattern cohérent avec **H_309** : le slow critic (tau=0.98) retarde les values → advantages bruités quand les returns évoluent vite → la policy oscille entre comportements au lieu de les empiler.

### v19 — fix H_309 APPLIQUÉ (en attente de crédits Modal)

Bootstrap des lambda returns par le **fast critic** (1 ligne, `train_dreamer_jax.py` dans `ac_loss_fn`) ; le slow critic ne sert plus que de régularisateur slowreg — structure officielle exacte. Smoketest OK.

```bash
.venv/bin/modal run --detach crafter_dreamer/scripts/modal_train_jax.py::main \
    --train-iter 4000 \
    --eval-interval 500 \
    --wm-train-per-iter 4 \
    --run-name v19_fast_critic_bootstrap \
    --no-use-rnd
```

Hypothèse : mêmes pics, moins de creux. Gates : best ≥ 4.5, et surtout PAS de déclin durable post-pic (les EVAL 2500-4000 restent ≥ 3.5). Si validé → run long 30k avec cette config (cible paper-S 5-7 ach à ~1M env_steps).

---

## v19/v19b — Fast critic bootstrap (H_309) : oscillation amortie, pas éliminée

**Statut : MITIGÉ. Fix conservé (alignement officiel), mais ce n'était pas la cause racine de l'instabilité.**

Premier run piloté par le SDK Lightning (RTXP 6000 spot, `lightning_train_jax.py`). v19 initial tué à iter 300 par le health monitor (`H_collapse` ×5 consécutifs — seuils calibrés à l'ère adaptive_alpha, faux positif en régime entropy-fixe où le burn-in plonge normalement à H≈0.08). v19b relancé avec `--no_health_auto_stop` : $1.05, 24 min de training, **2.8 ips** (vs 1.1 sur L4 — RTXP + mp_collect = ×2.5).

### Trajectoire v19b (= v18 + fast critic, un seul changement)

| Iter | argmax | sample | Notes |
|---|---|---|---|
| 500 | 1.20 | 1.80 | burn-in H=0.08 puis déverrouillage |
| 1000 | 3.20 | 3.60 | EN AVANCE sur v18 (2.30) — cycle plus rapide |
| 1500 | 3.30 | 3.20 | pic local |
| 2000-3000 | 3.00 → 2.30 | 2.60 → 2.20 | creux (même pattern que v18) |
| 3500 | **3.40** | 3.60 | remontée — nouveau best |
| 4000 | **3.40** | **3.80** | **finit À son best, sample montant** |

### Lecture

- **vs v18** : pic absolu inférieur (3.40 vs 4.00) mais PROFIL meilleur — v18 déclinait à la coupure (2.00 @ 3000), v19b fait un cycle creux→récupération et finit stable-montant. (Caveat : EVAL à 10 épisodes, variance réelle ; v18 aurait peut-être aussi rebondi.)
- **Gates stricts ✗** (best < 4.5, creux < 3.5) → H_309 n'était PAS la cause racine de l'oscillation. Le fix reste (structure officielle + accélère le early : 3.20 @ 1000).
- **La vraie dynamique de l'oscillation** (visible colonne `ret`) : returns imaginés 0.68 → ~0.0 après le pic. Les achievements maîtrisés (one-shot par épisode) saturent → le buffer se remplit d'états post-achievement à reward ~0 → reward head prédit ~0 → advantages morts → seule l'entropie agit → H remonte (0.45→0.9) → dilution → puis ré-apprentissage. Cycle, pas effondrement.

### Suspect suivant : H_310 — GAMMA 0.99 → 0.997 (paper)

γ=0.99 = horizon de valeur ~100 steps. Les achievements suivants (wood→table→pickaxe) sont plus profonds dans l'épisode → hors de portée des returns une fois les quick-wins saturés → c'est exactement le « ret≈0 » observé. γ=0.997 (paper) = horizon ~330. Un seul changement de constante.

### v20 (à lancer)

```bash
# GAMMA=0.997 nécessite un commit (constante l.~84) puis :
source .env.lightning && python crafter_dreamer/scripts/lightning_train_jax.py launch \
    --run-name v20-gamma0997 --machine rtxp6000 --interruptible \
    --extra-args="--no_health_auto_stop"
```

Gates v20 : best ≥ 4.0, creux ≥ 3.0, fin ≥ best v19b (3.40) avec wood/table en progression.

### Leçons pipeline Lightning

- `job.logs` indisponible PENDANT le run (limitation SDK) → temps réel = dashboard web uniquement
- Health monitor : seuils à recalibrer (H<0.3 normal en burn-in entropy-fixe) — en attendant, `--no_health_auto_stop` systématique
- RTXP 6000 + `--mp_collect` : 2.5× le débit L4 pour 2.5× le prix spot → même coût/iter, moitié moins d'attente. Run 30k estimé ~3h, ~$4.5 spot.
- **Resume + persistance implémentés** (post-v20) : `--resume_from` côté script (soft resume : poids + iter, Adam/buffer repartent), `lightning_sync.py` up/down vers le storage teamspace (`oneiro/<run>/`), sync background toutes les 5 min dans les jobs (préemption spot = ≤5 min perdues), `--resume-from-job` et `pull` côté launcher. Le run 30k peut partir en spot sereinement.

---

## v20 — GAMMA 0.997 (H_310) : burn-in 4× plus lent, profil post-décollage prometteur

**Statut : gates ratés à 4000 iter, MAIS asymptote indéterminée → v20b 8000 iter.**

Un seul changement vs v19b : `GAMMA 0.99 → 0.997` (horizon de valeur ~100 → ~330 steps, aligné paper). ~$1, 25 min.

### Trajectoire

| Iter | argmax | sample | unlocked |
|---|---|---|---|
| 500-1500 | **0.00** | 0.0-0.4 | 0/22 — burn-in très lent |
| 2000 | 0.90 | 0.80 | wake_up 90% |
| 2500 | 2.20 | 1.80 | +sapling, cow |
| 3000 | 2.00 | 2.20 | — |
| 3500 | 2.60 | 2.60 | +place_plant, wood 30% |
| 4000 | 2.50 | 3.00 ↑ | **5/22, wood 40%** (meilleur taux wood du projet à 4k) |

### Le mécanisme du burn-in lent : « l'agent voit sa mort »

À iter 1000-1500 : H=0.08 et **ret = -0.19, -0.39 (négatifs)**. Avec l'horizon 330, les returns imaginés capturent les pénalités de santé jusqu'à la mort (inévitable pour une policy débutante) → le signal dominant des 2000 premières iter est « tout mène à la mort » → le PG optimise la survie passive avant de chercher les +1. À γ=0.99 cette mortalité lointaine était invisible → décollage rapide. Coût structurel du long horizon, pas un bug.

### Le profil post-décollage (la raison de continuer)

- pente +0.8 ach/1000 iter sur 2000-4000, sample au max du run à 4000
- les unlocked **s'empilent sans se perdre** (1→3→4→5) vs v19b qui perdait place_plant pendant que H remontait
- `scale` monte en continu jusqu'à 2.81 (vs plafond ~1.9 à γ=0.99) — le système valorise plus loin
- pas de signature du cycle H_311 sur la fenêtre observée

---

## v20b — γ=0.997 × 8000 iter : PERCÉE COUCHE 2 (place_table) — verdict gamma : RETENU

**Statut : TERMINÉ** (47.7 min, ~$2, 261k env_steps = 26% du benchmark). Config identique à v20.

| Iter | argmax | sample | Fait marquant |
|---|---|---|---|
| 500-1000 | 0.0-0.3 | 0.2-0.8 | burn-in γ=0.997 (attendu) |
| 1500 | 2.80 | 2.40 | 6/22 — répertoire le plus large du projet à 1500 |
| **2000** | **3.90** ★ | 3.40 | best argmax (zombie 50%) |
| 2500-4000 | 3.8 → 2.7 | 2.8-3.8 | vague descendante, plancher ~2.7 |
| 4500-5500 | 2.5-3.0 | 2.2-3.0 | creux 2 (plancher 2.5 vs 2.0-2.3 à γ=0.99) |
| 6000-6500 | 3.80 | **4.40** | 2e vague — **wood 70% (record)** |
| **7500** | 3.50 | **4.40** | ⭐ **place_table=10% — PREMIER achievement couche 2 en 20 runs** |
| 8000 (fin) | 3.70 | **4.40** | place_table 10%, wood 60%, stable-haut |

### Verdict H_310 (γ=0.997) : RETENU pour le run 30k

**Pour** (tous les indicateurs de PROFONDEUR — ce qui compte pour atteindre 5-7) :
- **place_table débloqué** : la chaîne wood→table commence — c'est exactement la promesse du long horizon (valeur des chaînes profondes visible)
- sample 4.40 (record projet, atteint 3× : 6000, 7500, 8000) ; wood 70% (record, vs 40% max à γ=0.99)
- returns imaginés VIVANTS pendant les creux (0.5-0.96 vs ~0 à γ=0.99) → plancher d'oscillation monté (~2.5 vs 2.0)
- épisodes plus longs (190-208 steps : meilleure survie)

**Contre (honnête)** : best argmax 3.90 ≤ 4.00 (v18) — non-discriminant vu la variance ±1-2, mais le plafond argmax n'a PAS été franchi sur 8k. L'argmax mesure mal une policy qui s'élargit ; le sample et la composition sont les vrais signaux de progrès ici.

**Structurel** : sur 30k iter, le coût du burn-in (2000 iter) s'amortit ; le paper utilise 0.997 précisément pour le régime 1M steps.

### Pattern confirmé : oscillation en vagues, chaque vague plus PROFONDE (pas plus haute en argmax — plus riche en composition). La vague 2 (6000-8000) a la même hauteur argmax que la vague 1 mais : +place_table, wood ×1.75, sample +1.0.

### ⚠️ Leçon variance inter-run (importante pour TOUTES les comparaisons)

v20 et v20b ont **la même config ET le même seed (42)** : v20 = 0.00 @ 1500, v20b = 2.80 @ 1500. Le non-déterminisme GPU (réductions XLA non bit-déterministes) suffit à faire bifurquer les runs — système chaotique. Conséquences :
- la variance inter-run est de l'ordre de **±1-2 achievements**
- les écarts fins entre runs uniques (v18 « 4.00 » vs v19b « 3.40 ») sont en partie du **bruit**
- les conclusions fiables = tendances longues + multi-seeds ; les pics single-run ne classent pas les configs

---

## v21 — LE RUN BENCHMARK 1M : best 4.40 @ 13k, puis DÉRIVE TARDIVE (nouveau problème)

**Statut : TERMINÉ** (158.7 min, ~$4.7, 30 000 iter ≈ 965k env_steps ≈ benchmark 1M). Config : γ=0.997 + fast critic + entropy 3e-4 + buffer per-env + free bits + ratio 128.

### Trajectoire (EVAL toutes les 1000)

```
7000 : 4.20 ★ (7/22, place_table 20%)        ← pic vague 1, table présente
8000-12000 : vagues 2.6-3.5
13000 : 4.40 ★★ BEST (6/22, drink 70%, wood 60%)
14000-19000 : vagues 2.6-4.1 (19000 : 4.10, sample 4.20)
20000-30000 : DÉRIVE DESCENDANTE — plafonds de vagues 3.4 → 3.3 → 2.8 → 2.6
30000 (fin) : 2.20 (3/22 : sapling, wake_up, place_plant 20%)
```

### Confrontation à la prédiction (notée avant le run)

- Prédit : médiane 4.5-5.5 final, bas 3.5-4.0. **Réel : best 4.40** (fourchette basse ✓) **mais FIN 2.20** — la dérive descendante tardive n'était dans AUCUN scénario. Découverte, pas confirmation.
- Rainbow (4.3) : battu **en pic** (4.40 @ 420k steps) — PAS battu sur la métrique officielle (perf finale à 1M : 2.20).
- place_table : vu à 7000 (20%) puis plus jamais — la couche 2 a régressé au lieu de consolider.

### H_312 (nouveau) — la dérive coïncide avec le buffer PLEIN

Le buffer 500k se remplit à iter ~15 600 (32 steps/iter). La dérive nette commence ~19k. Mécanisme proposé : le FIFO écrase les données anciennes/diverses → tout le buffer = la policy récente → le WM perd la couverture, le reward head perd les contre-exemples → imagination myope → érosion mutuelle policy/WM. (+ wrap-around per-env : séquences chevauchant le ptr d'écriture = discontinuités sans done, ~0.2%/env — secondaire.)

**Fix candidat trivial** : BUFFER_CAPACITY 1M (= paper). Coût VRAM +6GB (12.3GB total uint8) — large sur RTXP 96GB (on était calibré L4 24GB). Couvre l'intégralité d'un run 30k sans wrap.

### Incident pipeline : checkpoints v21 PERDUS (upload 502)

Le sync (boucle ET final) a échoué sur les .npz : **HTTP 502 systématique sur les gros fichiers depuis les jobs** (meta.json 747B passe, npz 57MB rejeté — uploader single-part du SDK vs gateway). Le test local 57MB passait (44s). Les poids du run benchmark sont morts avec le conteneur ; les logs/EVAL (le résultat scientifique) sont saufs. Fix à investiguer pour v22 : `path_mappings` (mécanisme natif des jobs pour les artefacts) ou push HF Hub depuis le job.

### Bilan global du projet à ce point

| Époque | Niveau |
|---|---|
| Avant audit (v1-v14) | 0.8-2.7, actor cassé |
| Post-audit (v18-v20b) | 3.3-4.0 en 4-8k iter |
| **v21 best** | **4.40 @ 420k steps** (~Rainbow en pic) |
| v21 @ 1M (métrique officielle) | 2.20 — la dérive tardive est LE problème ouvert |

---

## v14 — Anti-spam OFF + entropy revert + adaptive ON (échec)

**Statut : code validé, smoketest OK, en attente run Modal.**

### Changements vs v13 (3 reverts ciblés)

| Param | v13 | v14 | Rationale |
|---|---|---|---|
| `SPAM_PENALTY` (env.py) | 0.02 | **0.0** | Anti-spam env interférait avec hyperparams paper (cf pattern #4) |
| `ENTROPY_COEF` | 3e-4 (paper) | **0.005** | Paper 3e-4 trop bas pour notre setup, 0.005 = v3 baseline qui marchait |
| `adaptive_alpha` default | False | **True** | Filet de sécurité pour réguler α dynamiquement |
| `auto_explore` default | False | **True** | Filet de sécurité contre stagnation (boost α si EVAL stagne) |

### Fixes architecturaux gardés (NE PAS TOUCHER)

- ✓ P1 : `recon_loss = ((decoded - obs) ** 2).sum(pixels).mean()` (corrige 12288× sous-évaluation)
- ✓ P2 : KL `clamp(per_step, free_bits).mean()` (vs `clamp(mean, free_bits)`)
- ✓ P3 : `LR_WM = LR_AC = 1e-4`, `GRAD_CLIP = 1.0` (vs 3e-4 + 100.0)
- ✓ P4 : actor `unimix = 0.01` dans `get_dist()` (1% uniforme)
- ✓ P5a : zero init sur `RewardHead.out` et `Critic.out` kernels
- ✓ P5c : `ELU → SiLU` dans tous les modules (encoder, decoder, RSSM, heads, actor, critic)
- ✓ P5d : `loss_critic = loss_main + 1.0 × loss_slowreg` (régularisation critic vers slow_critic)

### Validation smoketest (3 iter local)

```
Config : entropy=0.005  train_iter=3  n_envs=4  batch=16
Adaptive α : ON   H_target=2.000   init α=0.0050   lr_α=0.001
Auto-explore : ON  (threshold=0.05, patience=2, max=5.0)

iter 1/3 | wm=1757.310 recon=1750.5476 actor=-0.003 critic=11.083 H=2.67 α=0.0050 H*=1.13
iter 2/3 | wm=1771.135 recon=1764.8262 actor=-0.005 critic=11.003 H=2.70 α=0.0050 H*=1.13
iter 3/3 | wm=1782.649 recon=1776.4865 actor=-0.004 critic=10.916 H=2.72 α=0.0050 H*=1.13
```

Observations smoketest :
- `loss_recon ≈ 1750` (P1 actif, decoder produit du gradient utile) ✓
- `α = 0.0050` apparait (adaptive_alpha ON confirmé) ✓
- `H = 2.67-2.72` (proche max log(17)=2.83, normal en init random) ✓
- `loss_critic ≈ 11` en early (élevé, décroît déjà 11.08 → 10.92) ⚠️
- Pas de NaN, pas d'explosion, syntax OK ✓

⚠️ **Note critic_exploding** : la valeur ~11 dépasse le seuil `health_consec_threshold` (loss_critic > 10). Le health monitor pourrait warner "Critic_exploding" en early iter. C'est attendu avec slowreg + reward scale en init mais à surveiller — si ça persiste > 100 iter, faut investiguer.

### Hypothèse à valider

La combo `anti-spam OFF + entropy 0.005 + adaptive régulateurs` + tous les fixes architecturaux doit débloquer l'actor (qui restait random en v13) tout en gardant le WM réparé (recon convergent comme v12).

### Cibles

| Iter | Cible ach | Si atteint | Si raté |
|---|---|---|---|
| 500 | ≥ 1 | trajectoire saine | early signal mauvais |
| 1000 | ≥ 2 | mieux que v13 | sleep/noop spam de retour ? |
| 2000 | ≥ 3 | bat v3 baseline (2.70) | plafond similaire à v12 |
| 4000 | ≥ 4 | **SUCCESS** → lance long run 30k | autres investigations nécessaires |

### Commande pour lancer v14

```bash
.venv/bin/modal run --detach crafter_dreamer/scripts/modal_train_jax.py::main \
    --train-iter 4000 \
    --eval-interval 500 \
    --batch-size 16 \
    --run-name v14_no_antispam_baseline_entropy \
    --no-use-rnd
```

Coût attendu : ~$1, durée ~45 min sur Modal L4.

### Résultats v14 (run complet 4000 iter, ~38 min, ~$0.7)

| Iter | EVAL ach | Notes |
|---|---|---|
| 250  | 0.50 | early signal |
| 500  | 0.00 | recul |
| 750  | 0.00 | auto_explore boost 1.0→1.5 |
| 1000 | 0.00 | stagne |
| 1250 | 0.00 | boost 1.5→2.25 |
| 1500 | 0.00 | stagne |
| 1750 | 0.00 | boost 2.25→3.38 |
| 2000 | 0.00 | stagne |
| 2250 | 0.00 | boost 3.38→5.00 (max) |
| 2500 | 0.00 | stagne |
| 2750 | 0.50 | légère remontée |
| 3000 | 0.70 | progress, boost 5.00→4.25 |
| 3250 | 0.70 | stable |
| 3500 | **0.80** | ⭐ peak final |
| 3750 | 0.70 | recul |
| 4000 | 0.40 | régression finale |

WM converge bien :
- recon 674 → 47 (P1 fix actif et utile)
- wm loss 677 → 56
- critic stable ~1.0

H descend sur la fin : 2.83 → 1.75 (jamais target 1.13, mais quand même)
α descend très bas : 0.0048 → 0.0001 (adaptive_alpha pousse fort)

### Verdict v14 : ÉCHEC

**Best peak 0.80 ach** (vs v3 random baseline = 2.70, v12 = 1.60).

Hypothèse H_201 (anti-spam OFF + entropy revert + adaptive ON) : **INVALIDÉE**.

Tous les fixes architecturaux paper-exact n'ont **PAS suffi** à débloquer l'actor.

WM converge mais l'actor reste random (H stuck ~2.8 pendant 3000 iter).

### Synthèse globale après 14 runs

| Métrique | Valeur |
|---|---|
| Best peak score | 2.70 ach (v3 baseline, "par chance") |
| Cible paper-S 15M | 5-7 ach |
| Gap | -2 à -4 ach (sub-Rainbow 4.3) |
| Score paper XL | 11.7 ach |
| Score paper minimal possible (XS 8M) | ~6 ach |

**Notre score est PIRE qu'un agent random** sur le paper-attendu pour cette archi.

### Causes possibles restantes (non testées)

1. **Notre archi RSSM est mal proportionnée** : `deter=384, hidden=768, classes=24×24`. Paper size12m utilise `deter=2048, hidden=256, classes=32×16`. Nos proportions sont inversées (small RSSM, big MLP).

2. **Notre `n_envs=16` est trop élevé** vs paper `n_envs=1`. Collect/train ratio différent même avec train_ratio nominal aligné.

3. **Notre PyTorch baseline n'est peut-être pas non plus fonctionnel sur Modal** (jamais retesté récemment). v3 peak 2.70 pourrait être de la chance pure.

4. **Bug subtil restant** que ni audit modules ni audit training loop n'ont attrapé.

### Décision recommandée

3 options ouvertes (cf HYPOTHESES.md H_302-H_307) :

- **Option B** (recommandée) : lancer dreamerv3 officiel size12m sur Modal pour avoir LA RÉFÉRENCE ground truth. Coût ~$1.5.
- **Option C** : adapter symoon11/dreamerv3-flax (17.65 ach JAX) à notre Modal wrapper. ~$2 + 1-2h dev.
- **Option A** : accepter que 2-3 ach est notre limite pratique avec cette archi, pivoter vers autre projet ou autre angle.

