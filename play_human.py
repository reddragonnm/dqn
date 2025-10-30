"""
human_play_breakout.py

Play Atari Breakout with keyboard using Gymnasium + pygame.

Controls:
  Left / A  -> move left
  Right / D -> move right
  Space     -> FIRE (launch ball / start)
  Esc / Window close -> quit
"""

# ! This code is AI generated

import gymnasium as gym
import ale_py
import pygame
import numpy as np
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_a, K_d, K_SPACE


gym.register_envs(ale_py)


def find_action_indices(env):
    """
    Query env.unwrapped.get_action_meanings() when available and build a mapping
    from action-names to their indices. Fallback to common default indices
    when the env doesn't expose meanings.
    """
    try:
        meanings = env.unwrapped.get_action_meanings()
    except Exception:
        meanings = None

    mapping = {}
    if meanings:
        # meanings is something like ['NOOP', 'FIRE', 'RIGHT', 'LEFT', ...]
        for idx, name in enumerate(meanings):
            mapping[name.upper()] = idx

    # Fallback guesses for common ALE ordering
    fallback = {"NOOP": 0, "FIRE": 1, "RIGHT": 2, "LEFT": 3}
    for k, v in fallback.items():
        mapping.setdefault(k, v)

    return mapping


def play_breakout():
    # Some gym versions use "Breakout-v0" or "ALE/Breakout-v5". Try a common id and fallback.
    env_ids = [
        "ALE/Breakout-v5",
        "Breakout-v0",
        "Breakout-v4",
        "BreakoutNoFrameskip-v4",
    ]
    env = None
    for eid in env_ids:
        try:
            env = gym.make(eid, render_mode="rgb_array")
            print(f"Using env id: {eid}")
            break
        except Exception:
            continue
    if env is None:
        raise RuntimeError(
            "Could not create Breakout env. Install gymnasium[atari] and ROMs (AutoROM)."
        )

    obs, info = env.reset()
    action_map = find_action_indices(env)
    print("Detected action meanings (fallbacks used if not exposed):", action_map)
    # Build convenient mapping for keys -> actions
    action_left = action_map.get(
        "LEFT", action_map.get("RIGHT", 3)
    )  # note: sometimes LEFT/RIGHT swapped; we'll treat RIGHT/LEFT correctly below
    action_right = action_map.get("RIGHT", action_map.get("LEFT", 2))
    action_fire = action_map.get("FIRE", action_map.get("FIRE", 1))
    action_noop = action_map.get("NOOP", 0)

    pygame.init()
    frame = env.render()  # (H, W, 3)
    H, W = frame.shape[:2]
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Atari Breakout - Human Play (←/A, →/D, Space=FIRE)")

    clock = pygame.time.Clock()
    running = True

    # We will use "hold last action" behavior if no key pressed
    last_action = action_noop
    # For Breakout, you normally need to press FIRE at the beginning to serve the ball.
    started = False

    while running:
        frame = env.render()  # rgb_array
        # pygame wants width x height ordering for make_surface
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        action = last_action

        # Mapping: prefer explicit keys for left/right and fire
        if keys[K_LEFT] or keys[K_a]:
            action = action_left
            started = True
        elif keys[K_RIGHT] or keys[K_d]:
            action = action_right
            started = True
        elif keys[K_SPACE]:
            action = action_fire
            started = True
        else:
            # If episode not started (ball not launched), default to FIRE to make it easier:
            if not started:
                action = action_fire
            else:
                action = action_noop  # hold last action when no key pressed

        obs, reward, terminated, truncated, info = env.step(int(action))
        print(
            f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
        )

        screen.blit(surface, (0, 0))
        pygame.display.flip()

        last_action = action

        if terminated or truncated:
            obs, info = env.reset()
            started = False
            last_action = action_noop

        # Adjust fps if it feels too fast/slow. Atari runs at ~60Hz native; 30-60 is OK.
        clock.tick(30)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    play_breakout()
