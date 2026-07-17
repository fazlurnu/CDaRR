"""Shared setup for the equivalence suite.

The sim reads several files cwd-relative — e.g. ``envs/pairwise_conflict.py`` opens
``"envs/pairwise_params.json"`` at *import* time, and config paths are relative — so the
whole suite runs with the worktree root as cwd and on ``sys.path``.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
