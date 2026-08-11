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
| **reward/continue prédits sur** | `state` s_t (depuis 058beff) | **état APRÈS l'action** | ? (non revérifié) | 🔴 **RÉGRESSION** |
| critic loss pondérée par le discount | non | **oui** (`sg(weight[:,:-1]) * value.loss(...)`) | ? | 🔴 écart |
| bootstrap λ-return / baseline advantage | **fast critic** (H_309) | **slow critic** (`slowtar=True` par défaut) | ? | ⚠️ divergence assumée |

---

## Écarts notables (priorisés)

1. 🔴 **RÉGRESSION reward/continue — alignement temporel** *(vérifié verbatim sur
   `danijar/dreamerv3@main` le 2026-07-30 ; l'entrée précédente de ce tableau était FAUSSE
   et a justifié le commit `058beff`)*

   Ce que fait réellement danijar, `rssm.py::imagine` :
   ```python
   action = policy(sg(carry))                                  # action tirée de l'état courant
   deter  = self._core(carry['deter'], carry['stoch'], actemb) # on APPLIQUE l'action
   feat   = dict(deter=deter, stoch=stoch, logit=logit)        # feat[t] = état APRÈS action[t]
   ```
   puis `agent.py` : `imag_loss(imgact, self.rew(inp, 2).pred(), ...)` avec
   `inp = feat2tensor(imgfeat)`. Donc **`rew[t] = R_head(état produit PAR l'action)`** — la
   récompense immédiate de l'advantage *dépend* de l'action créditée.

   `lambda_return` compense l'indexation en interne :
   ```python
   interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
   ```
   et `imag_loss` apparie `adv[t] = ret[t] - tarval[:, :-1][t]` avec
   `logpi = logp(act)[:, :-1][t]`. L'action créditée et la récompense qu'elle produit sont
   du même côté. (Le nom de variable `imgprevact` au site d'appel confirme que les actions
   rendues par `imagine` sont « précédentes » relativement aux `feat`.)

   Oneiro, lui, calcule `reward_pred = reward_head.predict(state_vec)` **avant** le sample
   de l'action (train_dreamer_jax.py:435 vs 419), et sa `compute_lambda_returns` n'applique
   **aucun** décalage. Résultat : le terme immédiat de l'advantage est une **constante**
   vis-à-vis de l'action créditée → il se simplifie → **aucun crédit du premier ordre**.
   Effet mesuré : cf. `docs/HYPOTHESES.md` **H_317** (crédit inversé, −14 rangs).

   Le fix demande **deux** changements cohérents, pas un :
   - imagination : prédire sur `new_state_vec` (revert de 058beff) ;
   - entraînement WM : convention **entrante** — décaler la cible `rewards` **et** `dones`
     d'un cran, sinon la cible reste non identifiable (`state_vec[t]` contient `a_{t-1}`,
     jamais `a_t`, cf. rssm.py:349-352).

   Avec les deux, `adv[t] = r(s_t,a_t) + γV(s_{t+1}) − V(s_t)` redevient le Bellman standard
   et la λ-return d'Oneiro n'a besoin d'**aucun** slicing (son récurrence est déjà non
   décalée, là où danijar décale en interne — les deux formulations deviennent équivalentes).

2. 🔴 **Critic loss non pondérée par le discount** *(vérifié verbatim)* — danijar applique le
   même `weight` aux deux pertes :
   ```python
   policy_loss     = sg(weight[:, :-1]) * -(logpi * sg(adv_normed) + actent * sum(ents))
   losses['value'] = sg(weight[:, :-1]) * (value.loss(...) + slowreg * value.loss(...))[:, :-1]
   ```
   Oneiro pondère l'actor (`discount_cum`) mais **pas** le critic (`critic.loss(...)` en
   moyenne uniforme, l.713). L'état t=15 de l'imagination — issu de 15 rollouts de prior
   successifs, donc le moins fiable — pèse donc autant que t=0 dans l'entraînement du critic.

3. ⚠️ **Bootstrap : fast vs slow critic** *(vérifié verbatim)* — danijar :
   `tarval = slowval if slowtar else val`, avec `slowtar=True` par **défaut** ; `tarval` sert
   à la fois au bootstrap de la λ-return et de baseline à l'advantage. Oneiro a délibérément
   basculé sur le **fast** critic (H_309), motivé par le « pic 4.0 puis oscillations » de v18
   — or **H_314 a montré que cette lecture pic-puis-oscillation est un artefact**. La
   justification de H_309 tombe donc, et l'écart au paper reste. À re-tester.

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
