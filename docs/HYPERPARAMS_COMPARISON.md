# Oneiro vs DreamerV3 — comparaison des hyperparamètres

## Pourquoi ce document

Oneiro plafonne à un **crafter_score ~2.45%** (notre best, v26), soit **sous Rainbow (4.3%)** — un algo *sans* world model. Or un DreamerV3 correct doit battre Rainbow. Plutôt que de tâtonner run par run, ce doc compare **systématiquement** tous nos hyperparamètres aux deux références :
- **danijar/dreamerv3** — implémentation officielle du paper (config XL / Crafter).
- **symoon11/dreamerv3-flax** — réf JAX/Flax qui atteint **17.65%** sur Crafter.

Sources : `dreamerv3-official/dreamerv3/configs.yaml` + `agent.py`/`rssm.py` ; `dreamerv3-flax-ref/dreamerv3_flax/*.py` ; nos constantes `crafter_dreamer/scripts/train_dreamer_jax.py`.

## Repères Crafter (crafter_score, moyenne géométrique)

| Random | **Oneiro (best v26)** | Rainbow | PPO | DreamerV3 paper (XL) | symoon11 |
|---|---|---|---|---|---|
| 1.6% | **2.45%** | 4.3% | ~4.6% | 14.5% | 17.65% |

---

## Tableau comparatif

Légende : ✅ aligné · ⚠️ écart (capacité/réglage) · 🔴 bug identifié

### Architecture

| Hyperparamètre | Oneiro 14M (v26) | Oneiro 75M (actuel) | DreamerV3 XL (danijar) | symoon11 | Statut |
|---|---|---|---|---|---|
| deter (GRU) | 1280 | 2048 | **8192** | 4096 | ⚠️ on est petit |
| stochastique (cat × classes) | 32×16 = 512 | 32×32 = 1024 | 32×64 = 2048 | 32×32 = 1024 | ⚠️ |
| MLP hidden (RSSM/heads) | 256 | 1024 | 1024 | 1024 | ⚠️ (14M) / ✅ (75M) |
| CNN depth (base channels) | 16 | 32 | (mults) | 96 | ⚠️ |
| actor/critic (couches × largeur) | 2 × 256 | 5 × 1280 | **3 × 1024** | 5 × 1024 | ⚠️ (14M) |
| normalisation | LayerNorm | LayerNorm | **RMSNorm** | LayerNorm | ⚠️ mineur (danijar) |
| activation | SiLU | SiLU | SiLU | SiLU | ✅ |
| **params totaux** | **14.4M** | **75.5M** | ~200M+ | ~200M | ⚠️ capacité |

### Optimisation

| Hyperparamètre | Oneiro 14M | Oneiro 75M | DreamerV3 (paper) | symoon11 | Statut |
|---|---|---|---|---|---|
| LR World Model | 1e-4 | 1e-4 | ~1e-4 | 1e-4 | ✅ |
| LR Actor-Critic | 1e-4 | 3e-5 | ~3e-5 | 3e-5 | ✅ (75M) — 1e-4 cassait le gros actor (H_collapse) |
| grad clip | **1.0** | 1000 / 100 | AGC | 1000 / 100 | ⚠️ 1.0 écrasait les gradients (corrigé) |
| optimiseur | Adam | Adam | LaProp | Adam | ✅ ~ |

### RL / returns

| Hyperparamètre | Oneiro | DreamerV3 (danijar) | symoon11 | Statut |
|---|---|---|---|---|
| gamma | 0.997 | 0.997 (horizon 333) | 0.997 | ✅ |
| lambda (λ-return) | 0.95 | 0.95 | 0.95 | ✅ |
| imagination horizon | 16 | ~15 | 15 | ✅ |
| **train_ratio** | **128** | **512** (Crafter) | **512** | ⚠️ on est 4× sous |
| batch size | 16 | 16 | 16 | ✅ |
| seq length | 64 | 64 | 64 | ✅ |

### Losses / régularisation

| Hyperparamètre | Oneiro | DreamerV3 (danijar) | symoon11 | Statut |
|---|---|---|---|---|
| free bits | 1.0 (sommé sur cat) | 1.0 (free_nats) | 1.0 | ✅ |
| KL balance (β_dyn / β_rep) | 0.5 / 0.1 | 0.5 / 0.1 | 0.5 / 0.1 | ✅ |
| unimix | 0.01 | 0.01 | 0.01 | ✅ (au sampling ; **+ dans la KL** depuis le pack stabilité) |
| entropy coef | 3e-4 | 3e-4 (actent) | 3e-4 | ✅ |
| slow critic (EMA τ) | 0.98 | 0.98 (rate 0.02) | 0.98 (decay) | ✅ |

### Heads / signal

| Hyperparamètre | Oneiro | DreamerV3 (danijar) | symoon11 | Statut |
|---|---|---|---|---|
| twohot bins | 255 | 255 | 255 | ✅ |
| symlog / symexp | oui | oui | oui | ✅ |
| return normalization | Percentile-EMA P5/P95, decay 0.99 | retnorm + advnorm | Percentile-EMA decay 0.99 | ✅ ~ (danijar a 2 niveaux) |
| **reward/continue prédits sur** | **new_state s_{t+1}** ❌ → fix : `state` s_t | state s_t | state s_t | 🔴 **BUG (corrigé)** |

---

## Écarts notables (priorisés)

1. 🔴 **BUG reward/continue — alignement temporel** *(trouvé, fix appliqué, en validation)*
   La reward head est entraînée `R_head(s_t) ≈ r(s_t,a_t)` mais l'imagination la prédisait sur `s_{t+1}` → reward **décalé d'un cran** dans la λ-return → crédit temporel faussé → l'imagination guide vers le mauvais reward → **plateau sous Rainbow**. Les 2 réfs prédisent sur le même état qu'à l'entraînement. Fix : prédire sur `state_vec`. Illustration de *"parité numérique ≠ correctness"* (la baseline PyTorch avait le bug, reproduit fidèlement en JAX).

2. ⚠️ **Capacité** : deter (1280/2048 vs 4096/8192), stochastique, profondeur actor-critic. Notre 14M ↔ ~200M des réfs. → plafond de score, mais **pas la cause du "sous-Rainbow"** (un 14M correct bat Rainbow).

3. ⚠️ **train_ratio 128 vs 512** : on est 4× sous le paper. Le train_ratio élevé nous *dégradait* (v25/v28) — peut-être **à cause du bug reward** (plus d'entraînement sur signal faussé = pire). À re-tester une fois le reward corrigé.

4. ⚠️ **Normalisation RMSNorm (danijar) vs LayerNorm (nous + symoon11)** : mineur, symoon11 marche en LayerNorm.

5. ✅ **Tout le scalaire est aligné** : gamma, lambda, free bits, β KL, unimix, entropy, twohot, batch, seq, slow critic. Pas de piste là-dedans (déjà audité).

---

## Synthèse des runs (Kaggle TPU)

| run | archi | train_ratio | particularité | crafter_score | verdict |
|---|---|---|---|---|---|
| v24 | 14M | 128 (4:1) | — | 2.11% | couche 1 + bois |
| v25 | 14M | 512 (16:1) | déséquilibre WM:AC | 1.37% | régression (WM étouffe AC) |
| **v26** | 14M | 128 (1:1) | **best** | **2.45%** | plateau couche 2 |
| v27b | 14M | 512 (1:1) | LR 1e-4 | ~1.50% | stagne |
| v28 | 14M | 512 (1:1) | LR 4e-5 | 1.69% | pic puis dégrade |
| v29 | 14M | 128 | pack stabilité + LR_AC 3e-5 | 1.86% | régression (LR_AC trop bas pr petit AC) |
| v31 | 75M | 128 | — | ~1.34% (coupé) | H_collapse (gros actor + LR 1e-4) |
| v32 | 14M | 128 | **fix reward** | _(à compléter)_ | _(en cours de récup)_ |
| v33 | 75M | 128 | fix reward + LR_AC 3e-5 | — | à lancer |

**Conclusion** : nos hyperparamètres sont conformes aux réfs sur tout le scalaire. Les seuls vrais écarts sont (a) le **bug reward** (corrigé, le candidat n°1 pour le plateau sous-Rainbow), (b) la **capacité** (plafond de score, pas la cause du sous-Rainbow), (c) le **train_ratio** (à re-tester après le fix). Le test décisif reste : est-ce que le fix reward fait **passer Rainbow** ?
