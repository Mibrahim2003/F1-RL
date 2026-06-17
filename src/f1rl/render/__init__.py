"""Offscreen eval-clip renderer (training-only).

:mod:`f1rl.render.renderer` turns a recorded trajectory into an mp4 via headless Pygame
+ imageio. It is never imported by ``env/`` or the training hot path.
"""
