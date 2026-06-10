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

**Statut** : ouvert — candidat n°1 pour v22

**Fix candidat** : BUFFER_CAPACITY 500k → 1M (= paper). +6GB VRAM (12.3GB total), large sur RTXP 96GB. Couvre un run 30k entier sans wrap. Alternatives/compléments : réserve permanente de données early, ou ratio↑.

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

### H_301 — Notre archi RSSM est trop déséquilibrée

**Énoncé** : Notre `H_DIM=384` (RSSM deter) est trop petit vs paper `size12m=2048`. Notre `HIDDEN_DIM=768` (mlp) est trop grand vs paper 256. Architecture asymétrique non standard.

**Origine** : Audit Phase 2-DEEP (table comparaison hyperparams)

**Run de test** : aucun

**Statut** : ouvert

**Test possible** : refaire l'archi pour matcher size12m officiel : `deter=2048, hidden=256, classes=16, depth=16, units=256`. Mais c'est un gros refactor.

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
