# Hypothèses — Projet World Model DreamerV3 JAX

> **Objectif** : tracker toutes les hypothèses (testées ou non) sur ce qui fait/casse l'apprentissage.
> Évite d'oublier les pistes et permet de mesurer la qualité des intuitions.

## Format

Chaque hypothèse :
```
### H_NNN — Titre court

**Énoncé** : ...
**Origine** : observation / paper / audit / intuition
**Run de test** : v_XX
**Statut** : à tester / en cours / VALIDÉE / INVALIDÉE / partielle
**Résultat** : ...
**Conclusion** : ...
```

---

## Hypothèses VALIDÉES ✓

### H_000 — Le gradient policy ne doit pas remonter via le WM (stop_gradient sur state_vec dans imagine) 🎯 LE BUG FINAL

**Énoncé** : Dans `imagine_trajectory`, le `state_vec` passé à `actor.get_dist()` n'avait PAS de `stop_gradient`. Du coup le gradient policy remontait via la chaîne RSSM (GRU + sample STE × 16 imagination steps) → bruit énorme dans le signal PG → actor reste random.

**Origine** : Audit ciblé actor (Phase 2-ACTOR) vs danijar/dreamerv3 agent.py:196 qui utilise `sg(imgfeat)`.

**Run de test** : v15 (1 ligne : `state_vec_sg = jax.lax.stop_gradient(state_vec)`)

**Statut** : ✓ VALIDÉE — LE BUG FINAL

**Résultat** :
- v14 (sans fix) : H stuck 2.8 pendant 4000 iter, peak 0.80 ach
- v15 (avec fix) : H descend de 2.4 → 0.95 en 1000 iter, peak 2.0 ach à iter 4000
- Trajectoire MONTAIT encore à la fin du run (2.0 ach + courbe croissante)

**Conclusion** : LE bug critique. C'est l'unique cause qui explique "WM converge bien + actor reste random". Le fix est 1 ligne. Aligné officiel paper (`sg(imgfeat)` dans agent.py:196 + `ac_grads: False` config default Crafter).

---

### H_001 — Recon loss `.mean()` vs `.sum(pixels).mean()` est LE bug principal

**Énoncé** : Notre recon loss `jnp.mean((decoded - obs)**2)` moyenne sur (B,T,C,H,W) au lieu de sum sur pixels puis mean batch+time. Différence : 12288× sous-évaluation du gradient recon.

**Origine** : Audit minutieux vs danijar/dreamerv3 (Phase 2-DEEP). Logs D-pilot officiel montrent recon=469 vs notre 0.01.

**Run de test** : v12 (avec fix .sum(pixels).mean())

**Statut** : ✓ VALIDÉE

**Résultat** : v12 montre recon descendre de 598 → 55 en 3250 iter (vs v11 stagne à 0.01). Le WM converge maintenant comme attendu.

**Conclusion** : Bug critique. À garder. Le decoder ne s'entraînait jamais vraiment avant ce fix.

---

### H_002 — KL clamp doit être per-step (vs mean puis clamp)

**Énoncé** : Paper DreamerV3 fait `clamp(KL_per_step, free_bits).mean()`, on faisait `clamp(mean(KL), free_bits)`. La version paper encourage chaque step à avoir KL ≥ free_bits.

**Origine** : Audit Phase 2-DEEP

**Run de test** : v12

**Statut** : ✓ VALIDÉE

**Résultat** : Posterior collapse partiel résolu. Pas mesuré indépendamment, mais cohérent avec paper.

**Conclusion** : Paper-exact à garder.

---

### H_003 — GRAD_CLIP 100.0 ne se déclenche jamais

**Énoncé** : Avec gradient norm typique < 1.0, clip à 100 = pas de clipping effectif. Le paper utilise AGC 0.3 (per-param).

**Origine** : Audit Phase 2-DEEP

**Run de test** : v12 (clip 1.0)

**Statut** : ✓ VALIDÉE

**Résultat** : GRAD_CLIP=1.0 stable, pas d'explosion de gradients observée.

**Conclusion** : Clip 1.0 OK. AGC potentiellement encore mieux (pas testé).

---

### H_004 — Module encoder NHWC vs PyTorch NCHW (flatten order)

**Énoncé** : Notre encoder reshape (B, H, W, C) → flat (Flax NHWC) au lieu de (B, C, H, W) → flat (PyTorch NCHW). Features arrivent dans un ordre différent au proj_linear.

**Origine** : Test parité numérique Phase 16

**Run de test** : v11 (avec fix transpose NHWC → NCHW)

**Statut** : ✓ VALIDÉE (parité numérique)

**Résultat** : test_numerical_parity.py max_diff passe de 4.93 → 7.4e-06 sur encoder.

**Conclusion** : Bug structurel corrigé. Mais à lui seul n'a pas débloqué l'apprentissage (encore stagnation 2-3 ach).

---

### H_005 — GRU Flax NNX manque bias_hh

**Énoncé** : `nnx.GRUCell` a `use_bias=False` hardcoded sur dense_h. PyTorch GRU a 2 biais (bias_ih + bias_hh). Différence mathématique : pour le gate `n`, le bias_hh doit être multiplié par `r` (reset gate).

**Origine** : Test parité numérique Phase 16

**Run de test** : v11

**Statut** : ✓ VALIDÉE

**Résultat** : test_numerical_parity.py max_diff passe de 2.7e-2 → 1.2e-06 sur RSSM. CustomGRUCell créée matche paper exactement.

**Conclusion** : Bug structurel corrigé. Même remarque que H_004.

---

### H_006 — Anti-spam env interfère avec hyperparams paper

**Énoncé** : Notre SPAM_PENALTY=0.02 sur sleep/noop injecte du noise dans la distribution returns. Combiné à entropy_coef=3e-4 (paper), l'agent reste random pour minimiser la variance des penalties.

**Origine** : Investigation post-v13 (score -15.56 à iter 1250)

**Run de test** : v14 (anti-spam OFF)

**Statut** : partielle — l'anti-spam ajoutait du bruit (gardé OFF) mais n'était PAS la cause de la stagnation (v14 a stagné quand même ; le vrai bloqueur était H_007/H_000).

---

### H_007 — Adam fait du sign-step sur log_alpha : le décollage = érosion d'α jusqu'au coef paper

**Énoncé** : Avec `optax.adam(LR_ALPHA)` sur le scalaire log_alpha et un gradient quasi-constant (H figé à 2.83, H*=1.13 → grad=1.70), Adam normalise → `d(log α)/dt = -LR_ALPHA` exactement. L'actor quitte l'uniforme quand α croise ~3e-4 (le coef entropie FIXE du paper). `t_décollage = ln(α_init/3e-4)/LR_ALPHA`.

**Origine** : Audit 46-agents post-v16 (panel actor-alpha-dynamics)

**Run de test** : rétro-validation v15 + v16

**Statut** : ✓ VALIDÉE (vérification adversariale)

**Résultat** :
- v16 : α prédit aux **5 points de logs** à la précision d'affichage (0.0050→0.0046). Test discriminant : sans normalisation Adam on aurait lu 0.0043 — le log dit 0.0046.
- v15 (LR 1e-3) : décollage prédit ~2 800, observé ~3 000 ✓
- v16 (LR 1e-4) : décollage prédit ~28 000 → run 2000 plat par construction ✓

**Conclusion** : Le "breakthrough" v15 = α redescendant à la valeur paper, pas H* ni auto_explore (qui a même retardé ×2.25). L'alpha adaptatif est un détour : toutes les références (danijar, symoon11, NM512, sheeprl) utilisent un coef FIXE 3e-4. → ENTROPY_COEF=3e-4 fixe, adaptive OFF par défaut.

---

### H_008 — Le buffer entrelace les 16 envs : les séquences RSSM sont des trajectoires fictives

**Énoncé** : `buffer.add()` dans `for i in range(n_envs)` + stockage plat + sampling contigu → chaque séquence de 64 steps change d'env À CHAQUE STEP. Le prior apprend une dynamique inter-envs qui n'existe pas ; l'imagination roule sur ce prior mort.

**Origine** : Audit 46-agents (panel data-replay-ratio), re-vérifié à la main dans `src_jax/buffer.py:108-111`

**Run de test** : v17 (fix dans le bundle)

**Statut** : ✓ VALIDÉE sur le code (impact à mesurer en v17)

**Résultat** : La recon convergeait quand même (posterior = fonction de la frame courante) — c'est ce qui a masqué le bug pendant 16 runs. Explique le plafond ~2 ach de v15.

**Fix** : buffer per-env `(n_envs, per_env_cap)`, séquences mono-env. Testé localement : mono-env ✓, timesteps consécutifs ✓, wrap FIFO ✓.

---

### H_009 — Free bits appliqué par catégorie (24 nats/step au lieu de 1) : prior quasi pas entraîné

**Énoncé** : `maximum(kl, free_bits)` sur shape (B,T,24) clampe CHAQUE catégorie à 1 nat → free bits effectif = 24 nats/step. La KL réelle étant dessous, gradient ≈ 0 → le prior n'apprend presque rien.

**Origine** : Audit 46-agents (panel wm-imagination-quality), re-vérifié à la main dans `src_jax/model/rssm.py:498`

**Run de test** : v17 (fix dans le bundle)

**Statut** : ✓ VALIDÉE (preuve dans les logs : `kl=0.62 ≈ 0.5×1.0+0.1×1.0` = le floor exact)

**Fix** : somme sur les z_cat catégories AVANT le clamp (KL jointe par step, comme l'officiel). Testé : floor exact à KL=0, gradient vivant sinon. Synergie avec H_008 : le prior était à la fois entraîné sur des transitions fictives ET sans gradient.

---

## Hypothèses INVALIDÉES ✗

### H_101 — Le forgetting du buffer 50k est la cause de la stagnation

**Énoncé** : À iter 3000, buffer 50k écrasé ~10× → l'agent oublie ce qu'il a appris.

**Origine** : Observation v3 peak iter 2000 puis baisse

**Run de test** : v4 (buffer 500k FIFO)

**Statut** : ✗ INVALIDÉE

**Résultat** : v4 (500k) atteint 0.70 vs v3 (50k) atteint 1.30. Pas mieux, peut-être pire (mais variance).

**Conclusion** : Le forgetting n'était PAS la cause. Buffer 50k OU 500k = équivalent sur 30k iter. Le problème était ailleurs (bugs structurels).

---

### H_102 — Replay ratio 4× va aider à exploiter la data

**Énoncé** : Train plus de fois sur chaque batch = plus de gradient descent = meilleur apprentissage.

**Origine** : Intuition supervised learning

**Run de test** : v5 (WM_TRAIN_PER_ITER=4, train_ratio ~40)

**Statut** : ✗ INVALIDÉE

**Résultat** : v5 atteint 0.20 ach à iter 1000 (PIRE que baseline).

**Conclusion** : En model-based RL, train_ratio bas = buffer dominé par policy courante = WM overfit la policy au lieu d'apprendre la dynamique. Paper Crafter utilise 512, pas 40.

---

### H_103 — H_target schedule descendant va naturellement éviter mode collapse

**Énoncé** : Au début explore (H_target haut), à la fin exploite (H_target bas).

**Origine** : Intuition RL classique (epsilon decay)

**Run de test** : v8 (schedule 1.5 → 0.8 sur 20k iter)

**Statut** : ✗ INVALIDÉE

**Résultat** : v8 score 1.60 → 1.30 → 1.00 (régression). L'agent a appris à dormir, score baisse car policy se contracte avant que la reward landscape soit découverte.

**Conclusion** : Sur sparse-reward Crafter, schedule descendant force exploitation prématurée. Paper utilise entropy_coef CONSTANT (pas de schedule).

---

### H_104 — LR_ALPHA boost (3e-4 → 1e-3) va accélérer la convergence α

**Énoncé** : α descend trop lentement, augmenter LR_ALPHA = α converge plus vite vers la cible.

**Origine** : Observation v9 (α descend en 3000 iter)

**Run de test** : v10 (LR_ALPHA=1e-3)

**Statut** : ✗ INVALIDÉE

**Résultat** : v10 score 0.00 à iter 1000. H oscille entre 0.0 et 2.8 (collapse + restoration).

**Conclusion** : SAC adaptive temperature = système couplé α ↔ π. Si α bouge plus vite que π, oscillations boucle PI mal tunée. Règle : `LR_ALPHA ≤ LR_AC`.

---

### H_105 — entropy_coef paper 3e-4 + tous nos fixes = succès

**Énoncé** : Avec tous les fixes architecturaux paper-exact, utiliser entropy_coef paper devrait marcher.

**Origine** : Logique paper-alignment

**Run de test** : v13 (3e-4 + adaptive_alpha OFF + anti-spam env ON)

**Statut** : ✗ INVALIDÉE

**Résultat** : v13 atteint 0.80 ach à iter 1250 avec score=-15.56 (sleep spam cumulé).

**Conclusion** : Notre anti-spam env interfère. Soit on enlève anti-spam (v14), soit on garde entropy_coef plus haut (0.005 + adaptive).

---

## Hypothèses EN COURS DE TEST 🔄

### H_313 — Le vrai goulot est la DURÉE DE VIE, et `rare_weight` l'écrase

**Énoncé** : le plafond à ~3 achievements / crafter_score 2.2-2.5% n'est ni un problème
de WM, ni d'archi, ni de critic : l'agent meurt de soif à ~180 steps, ce qui rend l'arbre
de craft mécaniquement inatteignable. `REWARD_RARE_WEIGHT=10` en est la cause probable :
il noie le signal de santé (±0.1) sous son propre leakage, donc la reward head
n'apprend jamais à prédire la mort et l'actor n'a aucune raison de boire.

**Origine** : audit 2026-07-29/30 des logs v42 / v47 / v50 / v51 (success rates par achievement).

**Preuves mesurées** :
- `length` = **169-205 steps, constant** sur 3 runs × 30k iter. Aucune progression, jamais.
- `collect_drink` = 0-13% et **décroissant** ; `eat_plant` = 0% partout. L'agent ne boit pas.
- `collect_wood` est MAXIMAL au début puis s'effondre : v42 52%→21%, v47 56%→19%,
  v51 31%→5%. L'agent **désapprend** le bois (donc la compétence est atteignable :
  ce n'est pas un manque de capacité mais un collapse vers un optimum local).
- Conséquence en cascade : `place_table` 0-1% → `collect_stone` 0% → aucune table dans
  le replay → le critic n'a aucun exemple de la valeur du bois → repli sur les 3 seuls
  achievements sans prérequis (`collect_sapling` 100%, `wake_up` 99%, `place_plant` ~60%).
- Budget de signal par step : perte de santé jusqu'à la mort ≈ 9 pts × 0.1 / 180 steps
  = **-0.005/step**, contre un leakage mesuré `rew@0` = **+0.013/step**. Le biais de la
  head est 2.6× plus grand que le signal de survie qu'elle doit apprendre.
- Corrélation sur 22 runs : scale médian **2.63 sans** `rare_weight` vs **7.25 avec**
  (×2.8 → advantages ÷2.8). Le record absolu du projet (v26, 2.46%) est **sans**.
- Les critères de succès du fix v44 (`rew@0 ≤ 0.005`, `scale ≤ 5`) sont **ratés** :
  réel 0.008-0.015 et 7.0-7.4.

> ⚠️ **RECTIFICATIF (audit multi-agents 2026-07-30, 15 agents, 511 vérifications)** —
> plusieurs preuves de cette entrée et de H_314 ci-dessous sont **fausses**. Lire
> **H_315** avant de s'appuyer sur quoi que ce soit d'écrit ici. En résumé :
> - **v52 n'est PAS une expérience mono-variable.** Header du log : v42/v47/v50/v51 ont
>   `ac_train/iter=4`, **v52 a `ac_train/iter=1`**. Le launcher Kaggle n'expose pas
>   `--ac_train_per_iter` et le défaut du fichier est 1 ; les runs précédents le passaient
>   via `--extra-args`, pas v52. Le replay ratio de l'AC a donc été divisé par 4 **en même
>   temps** que `rare_weight`. Deux variables. Toute conclusion causale ci-dessous est
>   confondue. [mesuré]
> - **v52 n'est pas le meilleur run du projet.** À itération appariée (20k), sur la
>   métrique d'éval `achievements` : v52 = **2.88 dernier / 3.49 best**, v50 = **3.77 /
>   3.77**, v47 = **3.55 / 3.55**. v52 est *moins bon*. La comparaison qui concluait
>   l'inverse portait sur les success rates individuels (wood/drink/table) en ignorant
>   la substitution avec `place_plant`. [mesuré]
> - **`rew@ach` est mesuré IN-SAMPLE** — sur le batch même qu'on est en train de fitter
>   (l.543-548). Hors échantillon, sur transitions on-policy fraîches, les deux heads
>   valent **0.40** : v50 (`rare_weight`=10) affiche 0.80 → réel 0.40 ; v52 (=1) affiche
>   0.58 → réel 0.40. Les critères « atteints » l'étaient sur le mauvais échantillon,
>   et les deux réglages généralisent identiquement. [mesuré]
> - **`rew@0` n'est pas un offset uniforme** : médiane +0.0014, moyenne +0.057, sd 0.135,
>   **19.5%** des états > 0.05. C'est un champ *localisé* — il ne se simplifie pas dans
>   l'advantage. L'argument « biais constant » est faux dans les deux sens. [mesuré]
>
> Ce qui **reste vrai** : `length` est bien plat (181-194), la survie n'est pas le goulot,
> et `collect_drink` a bien progressé 8% → 47% sur v52 sans allonger la vie.

**Run de test** : v52-rareweight1 — `--rare_weight 1.0`, SEULE variable modifiée.
Config identique à v50-wmfix-t1 (n_envs=4, batch=16, seq_len=64, train_ratio≈128,
buffer CPU 1M, TPU v5e-8, 20k iter, eval tous les 2500). Pas d'unimix (l'échec de v51).

**Statut** : ✗ INVALIDÉE sur son critère — mais le remède marche pour une AUTRE raison

**Critère annoncé** : `length` > 250 → validée ; `length` reste 180-200 → invalidée.

**Résultat v52 (20k iter)** : `length` = **181-194, inchangé**, alors même que
`collect_drink` passe de 8% à **47%**. → Boire ne rallonge PAS la vie : le modèle causal
« ne boit pas → meurt de soif → 180 steps → pas le temps de crafter » est **FAUX**.
L'agent meurt d'autre chose (zombie/faim, non identifié). La durée de vie n'est pas le goulot.

**MAIS `rare_weight=1.0` débloque massivement, à budget égal (20k iter) :**

| | v50 (`rare_weight`=10) | **v52 (`rare_weight`=1)** |
|---|---|---|
| `collect_wood` max | 51% | **80%** ← record projet |
| `place_table` max | 3% | **9%** (×3) |
| `collect_drink` max | 1% | **47%** (×47) |
| `crafter_score` max | 1.81% | **2.02%** |
| best ach | ~3.2 @12.5k | **3.49 @7.5k** |

**Les 2 critères du fix v44, ratés depuis v38, sont enfin atteints** :
`rew@0` 0.013 → **0.002-0.003** (cible ≤0.005) et `scale` 7.5 → **1.9-2.2** (cible ≤5).

**Conclusion** : le mécanisme réel est celui identifié en premier — leakage de la reward
head → `scale` (P95−P5) gonflé ×3.75 → advantages écrasés d'autant → le PG ne consolide
jamais un comportement coûteux (navigation vers un arbre, aller boire). Ce n'était pas
une chaîne « survie → temps ». `rare_weight=1.0` doit devenir le défaut.

**Contrepartie mesurée** : `rew@ach` passe de 0.74 à 0.54-0.62 (la head sous-prédit
davantage les +1). Le `scale` sain compte donc plus que la précision de la reward head.
Un `rare_weight` intermédiaire (2-3) reste à explorer.

→ Ouvre **H_314** (l'effondrement post-pic est désormais LE problème central).

---

### H_314 — L'effondrement post-pic n'est ni le scale, ni le leakage, ni un H_collapse

**Énoncé** : sur tous les runs, la compétence atteint un pic tôt puis se dégrade.
v52 : `collect_wood` 80% @7.5k → 24% @20k, `place_table` 9% → 0%, `collect_drink`
47% → 11%. Le pic de v52 arrive **40% plus tôt** que v50 (7.5k vs 12.5k) et plus haut.

**Causes ÉLIMINÉES par v52** :
- `scale` gonflé : stable à 1.9-2.2 pendant toute la dégradation. ✗
- leakage reward head : `rew@0` stable à 0.002-0.003. ✗
- collapse d'entropie : H stable à 0.63-0.65, **0% des logs sous 0.3** après 10k.
  Les 39 warnings `H_collapse` sont TOUS à iter 0 (artefact de démarrage). ✗
- wrap FIFO du buffer (H_312) : 20k × 32 = 640k transitions < capacité 1M. ✗

**Statut** : ✗ **INVALIDÉE** (audit multi-agents 2026-07-30) — le phénomène n'existe pas.

**Réfutation** [mesuré, 2 métriques indépendantes] :
- **Test de permutation** sur les 18 runs à ≥6 evals : l'écart `max − dernier` observé vaut
  **0.398**, contre **0.654** pour le null « aucune tendance » (mêmes valeurs, ordre permuté).
  L'observé est *plus petit* que ce que la seule sélection du maximum prédit — dans 14/18
  runs. Sign test **p = 0.015 CONTRE** l'hypothèse. Pentes OLS post-warmup : **12/18
  positives**.
- Métrique indépendante à 6× plus d'échantillons (compteurs `train_ach_counts`
  différenciés entre evals, ~430 ép./fenêtre au lieu de 75) : gap 0.137 vs null 0.531,
  7/8 runs, p = 0.035.
- **v42/v43/v44/v47 terminent exactement à leur maximum.** La dernière eval de v47 (4.21
  @30k) est son max absolu et le meilleur chiffre du projet.
- Les repères « pics @12.5k » de l'énoncé sont faux : v42 pique @27.5k, v47 @30k, v50 @25k.
- L'artefact : lire « best-so-far vs dernier point » sur une série échantillonnée (tous les
  2500 iter) beaucoup plus lentement que le churn réel de la politique produit
  mécaniquement un « drop » dans ~54% des points d'arrêt possibles.

**Ce qui EST réel : une substitution sélective, à total conservé.** Sur v52, compteurs
TRAIN différenciés (~435 ép./fenêtre, erreur binomiale ~2.4 pts) :

| fenêtre | wood | drink | table | plant |
|---|---|---|---|---|
| [7500,10000] | 62.8% | 51.5% | 9.0% | 11.0% |
| [17500,20000] | 19.4% | 18.7% | 1.6% | **68.2%** |

Le total `ach/ep` reste plat (2.70 → 3.16) **par substitution**. En éval :
corr(wood, plant) = **−0.85**, corr(wood, drink) = **+0.93**. L'éval étant déterministe
(mondes 10 000+ep, clés 20 000+ep, argmax), ce n'est pas du bruit d'échantillonnage : la
politique **commute** entre deux modes mutuellement exclusifs.

**Cause de la commutation : NON IDENTIFIÉE.** Aucune signature intermédiaire n'est
mesurable sur v52 10k→20k : `imgR hi` 1.18→1.08%, `img ret` plat, `H` 0.63-0.65, `scale`
monotone −11%, `rew@ach` plat, densité d'achievements dans le batch en **hausse**
(1.6→2.0%). Question ouverte (a) — pourquoi ces modes ne se cumulent pas alors qu'ils
tiennent tous dans 185 steps.

→ La vraie question n'est ni H_313 ni H_314 : voir **H_315**.

**Note méthodo — le `crafter_score` affiché est cumulé depuis l'itération 0**
(`train_ach_counts` init l.1828-1829, jamais remis à zéro, consommé l.2315). Il ne peut
structurellement **pas** baisser : ce n'est ni une preuve ni une contre-preuve de quoi que
ce soit en cours de run. Idem pour tous les diagnostics comportementaux (`ACTIONS top5`,
`BOIS/épisode`) : écarts mesurés jusqu'à **×22** entre cumulé et fenêtré (`place_table
tentée` 3.28% cumulé vs 0.48% fenêtré sur v51). Utiliser les deltas entre evals.

---

### H_315 — Le plafond est un MUR DE CONJONCTION à la profondeur 2, pas un défaut d'optimisation

**Énoncé** : le plateau à 2.0-2.5% n'est pas causé par une instabilité, une dégradation ou
un mauvais réglage. **13 achievements sur 22 sont à exactement 0% sur les 28 runs de
l'histoire du projet**, les deux ères confondues. La moyenne géométrique plafonne alors
mécaniquement vers 4-5% *même avec 100%* sur les 9 restants. C'est un plafond de
**composition**, pas d'*optimisation*.

**Origine** : audit multi-agents 2026-07-30 (7 lentilles + réfutation adversariale).

> ⚠️ **CORRECTION (vérification 2026-07-30)** — le chiffre « `make_wood_pickaxe` = 0% sur
> 28 runs » est **FAUX**. Il vient de la lecture des lignes EVAL `unlocked (k/22)`, qui
> échantillonnent 75 épisodes et affichent des pourcentages entiers : un événement à 0.06%
> y apparaît nécessairement 0 fois. Les **compteurs TRAIN cumulés** donnent la vraie
> mesure, sur 54 208 épisodes (11 runs instrumentés) :
>
> | | total | taux | conditionnel |
> |---|---|---|---|
> | `place_table` | **544** | 1.00% | — |
> | `make_wood_pickaxe` | **31** | 0.057% | P(pioche \| table) = **5.7%** |
> | `make_wood_sword` | **29** | 0.053% | — |
> | `collect_stone` | **0** | 0.000% | P(pierre \| pioche) = **0/31** |
>
> Le mur n'est donc pas une impossibilité : c'est un **problème de taux**. La chaîne perd
> ~20× à chaque maillon (1.00% → 0.057% → 0). 31 événements noyés dans ~640k transitions
> ne peuvent rien entraîner, mais ils **existent dans le replay**. Le seul vrai zéro est
> `collect_stone` — le mur dur est *après* la pioche.

**Prérequis réels** (vérifiés dans `crafter/data.yaml`, jamais contrôlés avant) :
`place_table` coûte **wood=2** ; `make_wood_pickaxe` coûte **wood=1** de plus et exige
`nearby=[table]` ; `collect_stone` exige la pioche. Il faut donc **3 bois**, alors que
l'achievement `collect_wood` ne paie qu'**une seule fois** : les bois #2 et #3 rapportent
**exactement 0**.

**Preuves mesurées** :
- Histogramme du bois par épisode (v52, n=3478) : `0 bois=70.1% 1=19.6% 2=6.6% 3=2.0%
  4=1.2% 5+=0.5%` → **≥3 bois (requis pour la pioche) = 3.7% des épisodes**.
- **L'accumulation de bois EST apprise** (contre-mesure d'une hypothèse « collecte non
  dirigée », réfutée) — baseline random 250 épisodes vs v52 :

  | | ≥1 bois | ≥2 | ≥3 | P(≥3\|≥2) |
  |---|---|---|---|---|
  | random | 19.6% | 5.2% | 0.4% | **0.08** |
  | v52 | 29.9% | 10.3% | 3.7% | **0.36** |

  Une fois en « mode bois », l'agent continue 4.5× mieux qu'un random. Le goulot n'est pas
  l'accumulation.
- **Le goulot est le PREMIER maillon, celui qui est DIRECTEMENT récompensé** :
  `collect_wood` = 29.9% contre 19.6% pour un random. Un +1 immédiat, une action atomique
  à 1 step, et l'agent ne gagne que 10 points sur le hasard. C'est l'anomalie centrale —
  et elle est expliquée par **H_317**.
- Mort par **soif**, confirmée arithmétiquement (`crafter/objects.py:138-141`) :
  `thirst += 1`/step, à >20 → `drink -= 1` ; `drink` part de 9 → **9 × 21 = 189 steps**.
  Les longueurs observées (181-194) sont *exactement* cette échéance. La faim tue plus tard
  (9 × 26 = 234). Boire une fois ne donne que **+21 steps** — d'où le +20 observé sur v52
  quand `collect_drink` est passé de 8% à 47% sans que `length` bouge vraiment.
- **Ce n'est PAS un problème d'exploration** : `make_wood_pickaxe` est *tentée* 1.58-2.30%
  des steps (≈3 fois par épisode), `place_table` 0.41-1.91%, et 16 actions sur 17 sont
  au-dessus de 1%. Les actions atomiques sont jouées ; **la séquence n'est jamais assemblée**.
- Le goulot chiffré : `≥2 bois dans un épisode` = 4.9-12.8%, et
  **`P(place_table | ≥2 bois)` = 0.255 STABLE sur tout le run** (0.194/0.310/0.250/0.255/
  0.148/0.358/0.267). Ce n'est donc pas le craft qui est désappris — c'est l'accumulation
  de la **2ᵉ bûche** qui n'arrive jamais.
- Le mur préexiste au commit 058beff : `place_table` atteignait 20% en v21 (mais sur
  `EVAL_EPISODES`=10, soit 2 épisodes → non significatif), `make_wood_pickaxe` 0% déjà.

**Mécanisme candidat** (3 blocages, tous mesurés, tous non traités) :
1. **La 2ᵉ bûche rapporte exactement 0** (Crafter paie chaque achievement une fois par
   épisode, `env.py:119-120`). Le pont `bois#2 → table` repose donc *entièrement* sur `V`.
2. **Le champ de reward ne classe rien.** AUC de `R(s)` sur les états à reward nul pour
   prédire « un +1 dans les 8 steps suivants » = **0.518 [0.487, 0.552]** (v50 sur sa
   propre distribution) — indiscernable du hasard. Utilisable exigerait > 0.7.
   [mesuré : micro-test apparié 2 politiques × 2 WM, checkpoints v50@35k et v52@20k,
   2304 steps on-policy, 6 seeds]
3. **L'horizon de valeur réalisé ≈ 25 steps, pas 119.** Identité au point fixe
   `E[V] = E[r]/(1−γc)` sur les deux termes déjà loggés : v52 → **25.75** contre un
   nominal `1/(1−0.997×0.9946)` = 119. [mesuré, 201 lignes]

**Statut** : 🔄 OUVERTE — c'est LA question du projet. Question (b) : pourquoi la
conjonction `2 bois + table + position + make` n'est jamais assemblée en 28 runs alors que
chaque action atomique est jouée 1.6-2.3% des steps.

---

### H_316 — La cible de la reward head n'est pas une fonction de son entrée (convention sortante)

**Énoncé** : `rewards[t] = r(obs_t, a_t)` (convention **sortante**) est prédit depuis
`state_vec[t]`, qui contient `a_{t-1}` mais **jamais** `a_t`. La cible n'est donc pas une
fonction de l'entrée : l'optimum de Bayes est `Σ_a π(a|s)·r(s,a)`, une quantité
**policy-dependante et non stationnaire**. La head n'apprend pas `r(s)`, elle apprend la
propension de la politique courante.

**Chaîne vérifiée** [code-read, confirmée ligne à ligne] :
- `buffer.add(obs_list[i], a, r_final, ...)` — train_dreamer_jax.py:2036 → même index
- `prev_act_T = concat([0, act_T[:-1]])` — rssm.py:349-352 → `s_t` ne voit pas `a_t`
- `loss_reward = reward_head.loss(state_vec, rewards)` — l.534
- `reward_pred = reward_head.predict(state_vec)` — l.435, soit **avant** le sample de
  l'action (l.419) dans l'imagination

**Conséquence sur le PG** : dans `compute_lambda_returns`, `rewards[t]` est littéralement
indépendant de `a_t` → il se simplifie dans `A_t = returns[t] − V(s_t)`. Le seul terme qui
classe les actions est `γ·c_t·V(s_{t+1})` — exactement la quantité mesurée **au hasard**
(AUC 0.518, cf. H_315-2). L'actor ne reçoit **aucun crédit du premier ordre** pour l'action
qui déclenche le +1. C'est précisément ce qui manque pour l'**acte terminal d'une chaîne**
(`place_table` quand on a 2 bois).

**Corollaire** : le couple de critères exigé par le projet (`rew@ach ≥ 0.9` ET
`rew@0 ≤ 0.005`) est **mathématiquement insatisfiable** sous cette convention — mesuré
w=10 → (0.82 ; +0.029), w=1 → (0.58 ; +0.003). Le dilemme n'est pas un mauvais réglage,
c'est la non-identifiabilité de la cible.

**Second facteur, indépendant** : `observe_sequence` sans `initial_state` (l.513,
rssm.py:325-326) remet `h=z=0` tous les 64 steps, `start` uniforme (buffer.py:620-623) →
le modèle ne peut pas savoir qu'un achievement a déjà été décroché dans l'épisode.
Signature cohérente : `wake_up` (non répétable) prédit 0.998-0.999 vs `collect_sapling`
(répétable) ~0.006-0.10.

**Origine** : le commit **058beff** (2026-06-22) a déplacé la prédiction de `new_state`
(s_{t+1}) vers `state` (s_t), justifié par `docs/HYPERPARAMS_COMPARISON.md:73` qui
affirme que danijar prédit sur `s_t`.

✅ **VÉRIFIÉ le 2026-07-30** sur `danijar/dreamerv3@main` (lecture verbatim de `rssm.py` et
`agent.py`) : **058beff est bien une régression.** `rssm.py::imagine` applique l'action
*avant* de construire `feat` (`deter = self._core(carry['deter'], carry['stoch'], actemb)`
puis `feat = dict(deter=deter, ...)`) → **`feat[t]` est l'état APRÈS `action[t]`**, et
`imag_loss` reçoit `self.rew(inp, 2).pred()` avec `inp = feat2tensor(imgfeat)`. La
récompense immédiate de l'advantage **dépend donc de l'action créditée**. `lambda_return`
compense l'indexation en interne (`interm = rew[:, 1:] + ...`) et `imag_loss` apparie
`adv[t]` avec `logpi = logp(act)[:, :-1]`.

L'entrée `docs/HYPERPARAMS_COMPARISON.md:73` qui affirmait « danijar prédit sur state s_t »
et qui a justifié 058beff était **fausse** — corrigée, avec le verbatim, dans ce même
commit. La confusion venait du nom : le « state » de danijar est déjà post-action.

**Statut** : ✓ **VALIDÉE sur le plan de la correctness** (écart au paper prouvé, mécanisme
prouvé par lecture de code). Reste à mesurer l'effet sur l'apprentissage → candidat n°1.

**Test** : convention entrante (`_shift` de `rewards` ET `dones`, prédiction sur
`new_state_vec`). **Gate à l'itération 2000** (~20 min de TPU, pas 3h) : le couple
(`rew@ach` ≥ 0.90, `rew@0` ≤ 0.005) à `rare_weight=1` devient-il atteignable ? PASS → la
cible est devenue identifiable, laisser tourner. `rew@ach` reste 0.6-0.7 → thèse morte,
c'était un simple sous-apprentissage de classe rare (levier suivant : `W_REWARD` 1→10).
Ne prédit **pas** le score : les blocages H_315-2 et H_315-3 ne sont pas touchés.

**Note sur 058beff** : le commit était justifié par un argument de *cohérence* —
l'entraînement fait `R_head(s_t) ≈ r(s_t,a_t)`, donc l'imagination doit prédire sur `s_t`.
L'argument est valide en soi ; le problème est que la convention d'**entraînement** est
elle-même la source du défaut. Le fix doit donc déplacer **les deux** (cible ET
imagination), pas l'un sans l'autre. Une tentative de vérification de la convention de
danijar via l'API GitHub est restée **ambiguë** (le résumé obtenu se contredit entre
imagination et entraînement) → à trancher en lisant `agent.py` à la main avant de committer.

---

### H_317 — Le crédit est INVERSÉ sur l'action récompensée : la politique évite `do` précisément quand il paie

**Énoncé** : ce n'est pas que le signal de crédit est faible ou bruité — il est **de signe
opposé**. Sur les états où presser `do` rapporte immédiatement du bois (+1), la politique
classe `do` **dernière des 17 actions**.

**Mesure** [mesuré, `experiments/credit_assignment_probe.py`, checkpoint v50@35k, CPU 2 min] :
on déroule la politique dans le vrai Crafter, on *fork* l'env à chaque step pour tester si
`do` incrémenterait `wood`, puis on classe les 17 actions dans chaque état (même clé PRNG
pour les 17, moyenne sur 4 clés).

| groupe (n=78 chacun) | rang de `do` par `r+γV` | rang par **π** (actor seul) | `do` #1 |
|---|---|---|---|
| `do` **rapporte** du bois | **16.0**/17 | **17.0**/17 | **0%** |
| contrôle : `do` ne rapporte rien | 4.0/17 | 3.0/17 | 38% |
| *hasard* | *9.0* | *9.0* | *5.9%* |

**Écart : −12 rangs (valeur), −14 rangs (politique).** Un système non informatif donnerait
9/17 dans les deux groupes et un écart de 0. Ici la préférence est **anti-corrélée** avec
le paiement.

Le rang par **π** ne passe par *aucun* world model (fonction déterministe de `s_t` seul) :
ce n'est donc pas un artefact de l'imagination ni de mon proxy à un pas. La politique
apprise évite bel et bien la collecte de bois quand elle est à portée.

**Mécanisme candidat — boucle de verrouillage, conséquence directe de H_316** :
sous la convention sortante, l'optimum de la reward head est `E_a[r|s] = Σ_a π(a|s)·r(s,a)`
— la **propension de la politique courante**, pas la valeur de l'état. D'où :

```
la politique presse "do" sur l'herbe (sapling 98.8%) et rarement près des arbres
   -> R_head apprend « états-herbe = payants », « états-arbre = non payants »
      -> V, entraînée sur des λ-returns bâties sur R_head, dévalue les états-arbre
         -> la politique fuit encore plus les arbres
```

Auto-renforçant. Cela explique d'un seul mécanisme : (i) `collect_wood` bloqué à 29.9%
alors qu'il est directement récompensé, (ii) l'inversion mesurée ci-dessus, (iii) la
**commutation de modes** bois ↔ plant de H_314 (corr −0.85) — la cible de `R_head` suit la
politique, donc le système est *bistable*, (iv) pourquoi `rare_weight` n'a rien changé : il
repondère la CE, il ne rend pas la cible identifiable.

**Statut** : 🔄 OUVERTE, mais c'est la piste la mieux étayée du projet — un effet mesuré à
−14 rangs, pas une inférence.

**Signature de succès** : relancer le probe sur un checkpoint post-fix H_316. L'écart de
rang par π doit passer de **−14 à positif**. C'est un critère binaire, mesurable en 2 min
sur CPU, sans attendre les achievements.

**Réplication sur un 2ᵉ checkpoint indépendant** (v52@20k) — l'inversion tient, et sa
**magnitude corrèle avec le comportement** :

| checkpoint | écart de rang par π | écart par `r+γV` | `0 bois`/épisode |
|---|---|---|---|
| v50 @35k | **−14** | −12 | 75.9% (v51, même config) |
| v52 @20k | **−4** | −8 | **70.1%** |

Moins d'inversion → plus de bois collecté. Deux points seulement, mais dans le sens prédit
par le mécanisme, et sur deux configs différentes. Le probe est reproductible en 2 min :
`experiments/credit_assignment_probe.py --checkpoint <npz>`.

**Réserve restante** : `r+γV` est un proxy à **un pas**, pas la λ-return sur horizon 16 sur
laquelle l'actor est réellement entraîné. Le rang par **π**, lui, est sans réserve : c'est
la politique elle-même.

---

### H_201 — La combo v14 va débloquer l'actor

**Énoncé** : Anti-spam OFF + entropy_coef 0.005 + adaptive_alpha ON + tous les fixes architecturaux devrait donner > 3 ach à iter 4000.

**Origine** : Synthèse de toutes les leçons apprises (v3 baseline OK, v13 mauvais à cause interférence)

**Run de test** : v14 (terminé)

**Statut** : ✗ INVALIDÉE

**Résultat** :
- iter 500 : 0.00 ach (attendu ≥ 1) ❌
- iter 2000 : 0.00 ach (attendu ≥ 3) ❌
- iter 4000 : 0.40 ach (attendu ≥ 4) ❌
- Best peak : 0.80 ach @ iter 3500

**Conclusion** : Tous les fixes architecturaux paper-exact + entropy revert + anti-spam off n'ont PAS suffi à débloquer l'actor. WM converge bien (recon 674 → 47) mais actor reste random pendant 3000 iter. Le problème n'est pas l'anti-spam env ni l'entropy_coef paper.

→ Bascule vers H_301/H_302/H_303 (archi déséquilibrée, baseline PyTorch buguée, ou besoin référence externe).

---

## Hypothèses OUVERTES (pas testées)

> **Note post-audit (2026-06-09)** : ces hypothèses datent d'avant la découverte
> de H_007/H_008/H_009. H_302 (baseline PyTorch buguée) est quasi certaine mais
> sans intérêt désormais. H_303 (étudier symoon11) a été partiellement faite par
> l'audit externe — conclusion principale : entropy fixe 3e-4 partout. H_306 est
> recadrée par H_308 (le levier est le train_ratio, pas n_envs). Priorité
> actuelle : valider le bundle v17, puis H_308 → H_309 → H_310 dans l'ordre.

### H_308 — train_ratio 32 (16× sous le paper 512) limite la vitesse de convergence

**Énoncé** : On rejoue 32 timesteps par env_step collecté, le paper Crafter en rejoue 512. Le WM (surtout le reward head) mûrit trop lentement relativement à la policy → celle-ci, libérée par entropy fixe 3e-4, se commit sur du bruit (collapse v17).

**Origine** : Audit 46-agents (confirmé high). Les commentaires l.69-70 prétendaient "ratio paper" — faux, corrigés.

**Run de test** : v18 (`--wm-train-per-iter 4`, ratio 128, un seul changement vs v17)

**Statut** : ✓ VALIDÉE — LE déverrouilleur

**Résultat** : v17 (ratio 32) : collapse sur bruit, 0.00 ach argmax. v18 (ratio 128) : 0.70 → 2.30 → 3.50 → **4.00 @ iter 2000** (record projet, ×2 v15, ≈ Rainbow 4.3, en 64k env_steps). Le verrou initial sapling se desserre tout seul — le mécanisme advantage-s'éteint-puis-explore fonctionne dès que le signal reward arrive à temps. Coût wall-clock : ips 1.8 → 1.1 (acceptable).

---

### H_309 — Bootstrap des lambda returns par le slow critic cause l'instabilité post-pic

**Énoncé** : `train_step_ac` bootstrappait avec le slow critic (tau=0.98 → values en retard de ~50 iter) ; l'officiel utilise le FAST critic (le slow = régularisateur slowreg seulement). Values en retard → advantages bruités quand les returns évoluent vite → la policy oscille entre comportements au lieu de les empiler.

**Origine** : Audit 46-agents (confirmé medium), promu suspect principal par le pattern v18 post-pic : 4.00 @ 2000 puis 2.30 → 2.00, perte des achievements profonds, `sample > argmax` systématique, val/ret qui oscillent.

**Run de test** : v19b (Lightning RTXP 6000, $1.05)

**Statut** : partielle — fix CONSERVÉ mais pas la cause racine

**Résultat** : v19b = 1.20 → 3.20 → 3.30 (pic) → creux 2.30 @ 3000 → **3.40 @ 3500-4000 (best en fin de run, sample 3.80)**. Gates stricts ✗ (best < 4.5, creux < 3.5). Le fast critic accélère le early (3.20 @ 1000 vs 2.30 v18) et le profil finit stable-montant au lieu de déclinant, mais l'OSCILLATION demeure. Cause réelle identifiée dans les logs : returns imaginés 0.68 → ~0 post-pic (saturation des achievements one-shot → reward head prédit ~0 → advantages morts → l'entropie dilue → cycle). → H_310 (gamma) devient le test suivant.

---

### H_312 — Buffer FIFO plein → perte de diversité → dérive descendante tardive

**Énoncé** : Le buffer 500k devient plein à iter ~15.6k ; le FIFO écrase alors les données anciennes (warmup random, exploration early) → la distribution d'entraînement du WM se rétrécit à la policy récente → reward head sans contre-exemples, imagination myope → érosion mutuelle policy/WM → les plafonds de vagues BAISSENT au lieu de monter.

**Origine** : v21 — la dérive nette commence iter ~19k, le buffer est plein à ~15.6k. Best 4.40 @ 13k → fin 2.20 @ 30k. place_table vu à 7k puis plus jamais.

**Statut** : ouvert — candidat n°1 pour v22, AFFINÉ par la review des métriques internes v21

**Signature multi-métrique (review logs v21, échantillonnage /2000 iter)** :
- `p95` des returns fond en continu après 10-12k : 2.36 → 1.54 (les bonnes trajectoires imaginées disparaissent)
- `ret/val` : 0.57 @ 10k → 0.07 @ 14k → **négatifs** @ 26k/30k (l'imagination ne prédit plus que les pénalités — le pattern H_311 revient mais sans récupération)
- `kl` haute persistante (9-11.6) : le prior court après une dynamique qui change sans cesse
- `rec` anormalement bas en fin (9.25, record projet) : buffer devenu homogène = facile à reconstruire = **l'indice direct de la perte de diversité**
- `H` sain tout le run (0.5-0.86) : ce n'est PAS un collapse d'entropie

**Diagnostic affiné — H_311 déclenche, H_312 verrouille** : le cycle de saturation (H_311) existe depuis toujours, mais les récupérations des vagues 1-2 (6-13k) étaient possibles parce que le buffer contenait encore la diversité early (warmup random + exploration). Le buffer est plein à 15.6k → cette réserve s'écrase → chaque creux devient plus dur à remonter → dernière bonne vague @ 19k (4.10), puis plafonds descendants (3.4 → 2.8 → 2.2). La dérive des métriques internes commence dès 12-14k (avant le plein strict), mais la PERTE DE RÉCUPÉRATION date de ~19k.

**Fix candidat** : BUFFER_CAPACITY 500k → 1M (= paper). +6GB VRAM (12.3GB total), large sur RTXP 96GB. Couvre un run 30k entier sans wrap. Compléments : réserve permanente de données early (reservoir, jamais écrasée), et noter que le scale EMA (0.99) redescend lentement → écrase les advantages dans les creux.

---

### H_311 — Cycle de saturation des rewards one-shot = la vraie dynamique de l'oscillation

**Énoncé** : Les achievements Crafter ne paient qu'une fois par épisode. Une fois un comportement maîtrisé, le buffer se remplit d'états post-achievement à reward ~0 → le reward head converge vers ~0 sur la distribution courante → returns imaginés morts (`ret` ≈ 0 observé) → advantages nuls → seul le terme entropique agit → H remonte → la policy se dilue → la perf retombe → le reward redevient « surprenant » → ré-apprentissage. Oscillation entretenue.

**Origine** : analyse des logs v19b (ret 0.68 → -0.08 post-pic, H 0.45 → 0.9, pg → ±0.003)

**Statut** : ouvert — mécanisme cohérent avec v18 ET v19b. Mitigations candidates : γ=0.997 (H_310, étend l'horizon → la valeur des chaînes profondes reste visible quand les quick-wins saturent), train_ratio ↑, buffer plus divers.

---

### H_310 — GAMMA 0.99 vs 0.997 paper écrase les returns lointains

**Énoncé** : À t=200 steps, 0.99^200=0.13 vs 0.997^200=0.55 → les achievements profonds (chaînes longues) sont 4× sous-valorisés dans les returns.

**Origine** : Audit externe (refs comparison)

**Runs de test** : v20 (4000 iter), v20b (8000 iter, en cours)

**Statut** : ✓ VALIDÉE (sur la profondeur) — γ=0.997 RETENU pour le 30k. v20b 8000 iter : place_table 10% (1er achievement couche 2 du projet), wood 70% (record), sample 4.40 (record), plancher d'oscillation 2.5 (vs 2.0 à γ=0.99), returns imaginés vivants pendant les creux. Best argmax 3.90 ≈ inchangé (non-discriminant, variance) : le gain est dans la COMPOSITION, pas le pic.

**Résultats v20** :
- COÛT confirmé : burn-in 4× plus lent (0.00 ach jusqu'à iter 2000). Mécanisme : avec horizon 330, les returns imaginés early sont NÉGATIFS (-0.4) — ils capturent la mort inévitable de la policy débutante → le PG optimise la survie passive d'abord (« l'agent voit sa mort »).
- BÉNÉFICE en signature : post-décollage, les unlocked s'empilent SANS pertes (vs cycle H_311 à γ=0.99), scale monte à 2.81 (vs plafond 1.9), pente +0.8/1000 iter encore à 4000, wood 40% (record à 4k).
- v20b @ 1500 : 2.80, 6/22 unlocked (répertoire le plus large du projet à cette itération).

**⚠️ Caveat variance** : v20 vs v20b = même config + même seed → 0.00 vs 2.80 @ 1500 (non-déterminisme GPU, système chaotique). Variance inter-run ±1-2 ach → les comparaisons single-run fines sont du bruit ; juger sur tendances longues.

---

### H_301 — Notre archi RSSM est trop déséquilibrée (deter trop petit)

**Énoncé** : `H_DIM=384` (deter) trop petit vs proportions officielles, `HIDDEN_DIM=768` (mlp) trop grand. Le deter étant la mémoire récurrente, un deter famélique empêche de modéliser les chaînes longues (wood→table→pickaxe) → plafond couche 1-2.

**Origine** : Audit + comparaison config officielle (size12m : deter 2048, units 256, ratio 8:1 vs notre 0.5:1).

**Run de test** : v24 (deter 384→1280, hidden 768→256, z 32×16, cnn 16, embed 512 ; ~14.4M, MÊME budget réalloué). Kaggle P100, 20k iter, buffer 1M, gratuit.

**Statut** : ✓ VALIDÉE — l'archi rééquilibrée OUVRE la couche 3

**Résultat (lu dans les success rates TRAINING, PAS l'EVAL argmax)** :
| achievement | v23 (deter 384) | v24 (deter 1280) |
|---|---|---|
| place_table | 3.0% | **5.7%** |
| make_wood_pickaxe | **0.0%** | **0.8%** (~28 fois) |
| make_wood_sword | **0.0%** | **0.7%** |
| collect_stone | **0.0%** | **0.1%** |

v24 a fabriqué pickaxe/sword en bois et collecté de la pierre — **couche 3 atteinte**, ce que deter 384 ne faisait JAMAIS. Cohérent à 100% avec "il oubliait". PREUVE qu'un 14.4M PEUT atteindre la couche 3 → ce n'est PAS une limite de capacité.

**Nuances** : best EVAL argmax 3.70 = identique à v23 (les rares succès couche 3 n'apparaissent pas en argmax déterministe) ; crafter_score pic 2.23% (vs 1.97% v23) puis décline à 2.11% en fin (oscillation H_311 toujours là). Taux couche 3 très faible (0.8%) → non consolidé.

**Suite** : la couche 3 est OUVERTE mais pas CONSOLIDÉE. → H_308 (train_ratio 512) doit la consolider : on a ~28 succès rares dans le buffer, rejoués 128× (vs paper 512×) → 4× plus d'apprentissage les stabiliserait. Test : train_ratio 512 (wm_train_per_iter 16) sur TPU single-core.

---

### H_302 — Notre PyTorch baseline ne marche pas non plus

**Énoncé** : On n'a jamais retesté récemment notre PyTorch baseline sur Modal. Peut-être qu'il a les mêmes bugs (recon mean, KL global) et n'atteint pas non plus 5+ ach.

**Origine** : Investigation

**Run de test** : aucun (jamais lancé sur Modal récemment)

**Statut** : ouvert

**Test possible** : 1 run PyTorch sur Modal, 4000 iter, ~$1. Si PyTorch atteint 3-4 ach, c'est qu'on a une vraie référence local. Si PyTorch atteint 1-2 ach, notre référence "5-7 ach" était fausse.

---

### H_303 — symoon11/dreamerv3-flax (17.65 ach) est LA référence à étudier

**Énoncé** : Le repo `symoon11/dreamerv3-flax` atteint 17.65 sur Crafter en JAX/Flax (= notre stack). Mieux que paper officiel. Faire un diff systématique avec notre code révélerait des bugs/optims qu'on n'a pas.

**Origine** : Rapport recherche Phase 2-DEEP

**Run de test** : aucun

**Statut** : ouvert (priorité haute si v14 ne suffit pas)

**Test possible** : cloner localement, faire un diff systématique avec notre code, identifier ce qu'il fait différemment.

---

### H_304 — Le repo officiel danijar/dreamerv3 size12m atteint 5+ ach à 250k steps

**Énoncé** : Si on lance le code officiel sur notre Modal L4, il devrait atteindre 4-6 ach à 250k env_steps (premier signe convergence paper).

**Origine** : Plan Phase 0 (D-pilot)

**Run de test** : D-pilot lancé puis interrompu accidentellement à step 5580 (= 2.2% du total)

**Statut** : ouvert, à relancer si v14 stagne

**Résultat partiel** : à step 5580, score 1.1 (early signal positif, recon convergent).

**Test possible** : relancer 250k complet (~$1.5, 2h) et laisser finir sans interrompre.

---

### H_305 — Scale up vers size50m (50M params) débloquerait par capacité

**Énoncé** : Notre 15M est peut-être juste limité en capacité pour Crafter. Paper utilise 165M par défaut sur Crafter. Avec 50M (M-size), on devrait atteindre 6-8 ach.

**Origine** : Tableau scaling paper

**Run de test** : aucun

**Statut** : ouvert (à considérer si toutes les corrections de bugs ne suffisent pas)

**Test possible** : scale archi vers M (deter=4096, hidden=512, classes=32) + run 4000 iter Modal. Coût ~$3 (modèle plus lent).

---

### H_306 — Notre n_envs=16 collecte trop, le WM voit trop peu chaque transition

**Énoncé** : Avec n_envs=16 et COLLECT_PER_ITER=32, on collecte 512 transitions par iter mais on fait 1 seul train_step. Paper Crafter utilise n_envs=1 + train_ratio=512.

**Origine** : Différence config notable

**Run de test** : aucun

**Statut** : ouvert (variable couplée à train_ratio)

**Test possible** : passer à n_envs=4, COLLECT_PER_ITER=8 (= 32 transitions/iter, train_ratio 32). Plus proche du paper en n_envs.

---

### H_307 — Switch vers Craftax (env JAX) débloque tout

**Énoncé** : Craftax est un port JAX de Crafter qui tourne 250× plus vite. Pipeline 100% JAX (env + train) permet d'expérimenter beaucoup plus vite et d'atteindre des résultats type SOTA.

**Origine** : Recherche externe

**Run de test** : aucun

**Statut** : ouvert (option "nucléaire" si on veut pousser au max)

**Test possible** : port 1-2 jours de notre wrapper, puis run 1M env steps en ~2h Modal.

---

## Méta-hypothèses

### M_001 — Empiler les fixes sans isoler = brouillard total

**Énoncé** : Changer plusieurs variables en parallèle rend impossible d'identifier ce qui aide vs casse.

**Origine** : Expérience v5-v13 (souvent 3-5 changes par run)

**Statut** : ✓ VALIDÉE (confirmé en pratique)

**Action** : Pour v14+, idéalement isoler UN changement à la fois. Pragmatique : 2-3 changes max si liés conceptuellement.

---

### M_002 — Le paper est un système, picker des valeurs au pifomètre casse l'ensemble

**Énoncé** : Les hyperparams paper sont co-tunés avec normalisations + archi + env. Utiliser une valeur paper dans un setup custom = risque d'interférence.

**Origine** : Méta-leçon Agent B investigation

**Statut** : ✓ VALIDÉE

**Action** : Soit aller paper-exact partout, soit garder set custom cohérent. Pas de mélange.

---

### M_003 — Le test parité numérique ≠ correctness du training

**Énoncé** : Avoir des modules en parité numérique avec PyTorch ne garantit PAS que l'apprentissage convergera correctement. Le PyTorch baseline lui-même peut avoir des bugs (ex: notre recon mean au lieu de sum).

**Origine** : v11 a atteint parité numérique mais stagnait toujours

**Statut** : ✓ VALIDÉE

**Action** : Comparer avec le paper officiel (pas juste avec notre PyTorch).
