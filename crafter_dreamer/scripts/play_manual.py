"""
Play Crafter manuellement via pygame.

Utilise NOTRE wrapper CrafterEnv (pas crafter pip direct) pour valider
le pipeline bout-en-bout avec interaction humaine.

Lance :
    .venv/bin/python crafter_dreamer/scripts/play_manual.py
    .venv/bin/python crafter_dreamer/scripts/play_manual.py --seed 42

Contrôles (clavier QWERTY-style) :
    ↑ ← ↓ →  : déplacement (move_up/left/down/right)
    ESPACE   : do (action contextuelle : miner, attaquer, ramasser)
    TAB      : sleep
    R        : place_stone
    T        : place_table
    F        : place_furnace
    P        : place_plant
    1 / 2 / 3 : make_wood / stone / iron pickaxe
    4 / 5 / 6 : make_wood / stone / iron sword
    N        : noop (rien)
    ESC      : quitter

L'image 64×64 native est upscalée à 600×600 pour la jouabilité.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pygame

from crafter_dreamer.env import CrafterEnv, ACTION_NAMES, ACHIEVEMENTS


# ============================== Mapping clavier → index action

KEYMAP = {
    pygame.K_UP:    "move_up",
    pygame.K_DOWN:  "move_down",
    pygame.K_LEFT:  "move_left",
    pygame.K_RIGHT: "move_right",
    pygame.K_SPACE: "do",
    pygame.K_TAB:   "sleep",
    pygame.K_r:     "place_stone",
    pygame.K_t:     "place_table",
    pygame.K_f:     "place_furnace",
    pygame.K_p:     "place_plant",
    pygame.K_1:     "make_wood_pickaxe",
    pygame.K_2:     "make_stone_pickaxe",
    pygame.K_3:     "make_iron_pickaxe",
    pygame.K_4:     "make_wood_sword",
    pygame.K_5:     "make_stone_sword",
    pygame.K_6:     "make_iron_sword",
    pygame.K_n:     "noop",
}

# Inverse : nom → index
ACTION_NAME_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}


def render_obs_pygame(obs_chw_float, window_size):
    """
    Convertit obs (3, 64, 64) float [0,1] de notre wrapper en surface pygame.
    Upscale à window_size pour visibilité.
    """
    # (C, H, W) → (H, W, C) en uint8 [0, 255]
    img = (obs_chw_float.transpose(1, 2, 0) * 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))   # pygame veut (W, H, C)
    surf = pygame.transform.scale(surf, window_size)
    return surf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--window", type=int, default=600,
                        help="Taille fenêtre en pixels (par côté)")
    args = parser.parse_args()

    # Crée l'env via notre wrapper
    env = CrafterEnv(seed=args.seed)
    obs = env.reset()

    pygame.init()
    window_size = (args.window, args.window)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Crafter — playing via CrafterEnv wrapper")
    clock = pygame.time.Clock()

    print("=" * 50)
    print("Crafter manual play")
    print("=" * 50)
    print("Contrôles :")
    for key, action_name in KEYMAP.items():
        print(f"  {pygame.key.name(key):8s} → {action_name}")
    print(f"  ESC      → quit")
    print("=" * 50)
    print()

    total_reward = 0.0
    step_count = 0
    achievements_so_far = set()
    running = True
    last_action_name = "none"

    while running:
        # Affichage
        surf = render_obs_pygame(obs, window_size)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Attendre une action utilisateur
        action_idx = None
        while action_idx is None and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if event.key in KEYMAP:
                        action_name = KEYMAP[event.key]
                        action_idx = ACTION_NAME_TO_IDX[action_name]
                        last_action_name = action_name
            clock.tick(60)

        if not running:
            break

        # Step env
        obs, reward, done, info = env.step(action_idx)
        total_reward += reward
        step_count += 1

        # Détection nouveaux achievements
        for name, val in info.get("achievements", {}).items():
            if val > 0 and name not in achievements_so_far:
                achievements_so_far.add(name)
                print(f"  🏆 [step {step_count}] {name}")

        # Log step
        if reward > 0:
            print(f"  step {step_count:4d} | action={last_action_name:20s} | "
                  f"reward=+{reward:.1f} | total={total_reward:.1f}")

        if done:
            print()
            print(f"=== EPISODE TERMINÉ ===")
            print(f"  Steps           : {step_count}")
            print(f"  Total reward    : {total_reward:.1f}")
            print(f"  Achievements    : {len(achievements_so_far)} / {len(ACHIEVEMENTS)}")
            print(f"  Liste           : {sorted(achievements_so_far)}")
            print()
            print("Appuie sur SPACE pour rejouer, ESC pour quitter.")
            # Attendre confirmation
            waiting = True
            while waiting and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                        elif event.key == pygame.K_SPACE:
                            obs = env.reset()
                            total_reward = 0.0
                            step_count = 0
                            achievements_so_far = set()
                            print("\n--- New episode ---\n")
                            waiting = False
                clock.tick(30)

    print()
    print("=== Session ended ===")
    if step_count > 0:
        print(f"  Steps          : {step_count}")
        print(f"  Total reward   : {total_reward:.1f}")
        print(f"  Achievements   : {len(achievements_so_far)} / {len(ACHIEVEMENTS)}")
        if achievements_so_far:
            print(f"  Débloqués      : {sorted(achievements_so_far)}")
    pygame.quit()


if __name__ == "__main__":
    main()
