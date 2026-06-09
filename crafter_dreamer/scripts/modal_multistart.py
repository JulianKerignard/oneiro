"""
Multistart sur Modal : lance N runs en parallèle avec des seeds différents.

Réutilise la fonction `train` définie dans modal_train.py.
Setup multistart : 2 runs en parallèle sur Modal L4 + 16 CPU + 32GB.

Usage :
    modal run crafter_dreamer/scripts/modal_multistart.py

Coût estimé :
    2 runs × 30k iter × 17M params × ~11h sur L4 = ~$24 total
    (À comparer à 1 run × 8h = ~$9 pour palier 1)

Stratégie :
    - Seeds différents (42, 123) pour variance d'optim
    - Mêmes hyperparams (entropy_coef=0.005, auto_explore=True)
    - Sélectionner le meilleur run à la fin via achievements
"""

from modal_train import app, train


@app.local_entrypoint()
def multistart(
    train_iter: int = 30000,
    eval_interval: int = 2000,
    n_envs: int = 16,
    batch_size: int = 64,
    wm_train_per_iter: int = 1,
    ac_train_per_iter: int = 1,
    entropy_coef: float = 0.005,
    auto_explore: bool = True,
):
    """
    Lance 2 runs Modal en parallèle avec seeds différents.
    Modal les exécute en parallèle automatiquement via .spawn().
    """
    print("=" * 60)
    print("MULTISTART Crafter Dreamer (2 runs en parallèle)")
    print("=" * 60)

    runs = [
        {"seed": 42,  "run_name": "multistart_seed42_15M"},
        {"seed": 123, "run_name": "multistart_seed123_15M"},
    ]

    handles = []
    for cfg in runs:
        print(f"  Lancement run : seed={cfg['seed']}  name={cfg['run_name']}")
        handle = train.spawn(
            train_iter=train_iter,
            eval_interval=eval_interval,
            n_envs=n_envs,
            batch_size=batch_size,
            wm_train_per_iter=wm_train_per_iter,
            ac_train_per_iter=ac_train_per_iter,
            entropy_coef=entropy_coef,
            auto_explore=auto_explore,
            seed=cfg["seed"],
            run_name=cfg["run_name"],
        )
        handles.append((cfg["run_name"], handle))

    print()
    print(f"  {len(handles)} runs lancés en parallèle.")
    print(f"  Les containers démarrent indépendamment.")
    print(f"  Tu peux fermer ce terminal, ils continuent (les jobs sont 'detached').")
    print()
    print("  Pour suivre les progrès :")
    print("    modal app list")
    print("    modal app logs <APP_ID>")
    print()
    print("  Pour récupérer les outputs après :")
    print("    modal volume get worldmodel-outputs / ./modal_outputs")
    print()
    print("  Comparaison à la fin :")
    print("    Ouvre les JSON summaries dans modal_outputs/runs/")
    print("    Compare 'achievements' final pour sélectionner le best run.")
    print()

    # Optionnel : attendre tous les runs avant de retourner
    # Décommenter pour BLOQUER le terminal jusqu'à fin de tous les runs
    # for name, handle in handles:
    #     print(f"  Attente de {name}...")
    #     handle.get()
    # print("Tous les runs terminés.")
