"""
checkpoint.py — Auto-resume support for the AAM-TRANS Q1 pipeline.

A long Phase-2 run (≥10 h on Vast.ai) is fragile: one OOM, one transient
attack-eval crash, one disk hiccup, and you lose the whole night.

This module provides per-mode, per-task auto-checkpoints that save the
ENTIRE recoverable state after every successful Task-N stage of
run_aam_trans():

    run_dir/
    └── checkpoints/
        ├── resume_binary.pkl       ← latest snapshot of `binary` mode
        ├── resume_multiclass.pkl   ← latest snapshot of `multiclass` mode
        └── completed_modes.json    ← list of fully-completed modes

State saved per Task-N snapshot:
    * stage_done                 : int (1, 2, or 3 — which task is finished)
    * task_matrix                : np.ndarray [T, T] of CL accuracies
    * all_rows                   : list[dict] — clean + adversarial metrics so far
    * incremental_accs           : dict — per-stage per-dataset accuracies
    * fwt_pre                    : dict — forward-transfer baselines
    * memory_buffer              : pickled memory buffer (with all task data)
    * ewc_fisher                 : dict[str, Tensor] of Fisher info on params
    * old_model_state            : KD teacher state_dict
    * latest_model_path          : path to the .pth on disk (already saved)
    * surrogates                 : dict[task_name -> surrogate state_dict]

Resume logic:
    1. main.py --resume scans runs/ for the most recent run with any
       resumable mode and re-uses its run_dir.
    2. run_aam_trans() at the start of each mode:
          state = load_state(run_dir, mode)
          stage_done = state['stage_done'] if state else 0
          → skip ahead to the first un-done task in the loop
    3. After every successful Task-N completion, save_state() fires;
       any later crash leaves an intact snapshot of N.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# File-system helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ckpt_dir(run_dir: str) -> str:
    d = os.path.join(run_dir, "checkpoints")
    os.makedirs(d, exist_ok=True)
    return d


def _state_path(run_dir: str, mode: str) -> str:
    return os.path.join(_ckpt_dir(run_dir), f"resume_{mode}.pkl")


def _completed_path(run_dir: str) -> str:
    return os.path.join(_ckpt_dir(run_dir), "completed_modes.json")


# ──────────────────────────────────────────────────────────────────────────────
# Save / load per-mode snapshot
# ──────────────────────────────────────────────────────────────────────────────

def save_state(run_dir: str, mode: str, stage_done: int, state: Dict[str, Any]) -> str:
    """
    Persist a snapshot of training state for `mode` after Task `stage_done`.

    `state` is an arbitrary dict — caller decides what to include. Anything
    pickle-able is fine. Common keys: task_matrix, all_rows, incremental_accs,
    fwt_pre, memory_buffer, ewc_fisher, old_model_state, latest_model_path,
    surrogates.
    """
    path = _state_path(run_dir, mode)
    full = dict(state)
    full["mode"]       = mode
    full["stage_done"] = stage_done
    try:
        # Write to a temp file then rename — atomic, so a crash mid-write
        # never leaves a half-written pickle on disk.
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(full, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        logger.info(
            f"  💾 Checkpoint saved: mode={mode}, stage={stage_done} "
            f"({size_mb:.1f} MB) → {path}"
        )
    except Exception as e:                           # pragma: no cover
        logger.warning(f"  Checkpoint save FAILED: {e}")
    return path


def load_state(run_dir: str, mode: str) -> Optional[Dict[str, Any]]:
    """Load the latest snapshot for `mode`, or None if no checkpoint exists."""
    path = _state_path(run_dir, mode)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
        logger.info(
            f"  📂 Resume state loaded: mode={mode}, "
            f"stage_done={state.get('stage_done', '?')}"
        )
        return state
    except Exception as e:                           # pragma: no cover
        logger.warning(f"  Checkpoint load FAILED: {e}")
        return None


def mark_mode_completed(run_dir: str, mode: str) -> None:
    """Mark a mode as fully done so --resume skips it on the next run."""
    path = _completed_path(run_dir)
    completed: List[str] = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                completed = json.load(f)
        except Exception:
            completed = []
    if mode not in completed:
        completed.append(mode)
        with open(path, "w") as f:
            json.dump(completed, f)


def is_mode_completed(run_dir: str, mode: str) -> bool:
    path = _completed_path(run_dir)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            completed = json.load(f)
        return mode in completed
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Find the most recent resumable run
# ──────────────────────────────────────────────────────────────────────────────

def find_latest_resumable_run(runs_root: str) -> Optional[str]:
    """
    Return the most-recent run_dir under `runs_root` that has at least one
    saved checkpoint *and* is not 100 % completed. Returns None otherwise.
    """
    if not os.path.exists(runs_root):
        return None
    candidates = sorted(
        [d for d in os.listdir(runs_root)
         if os.path.isdir(os.path.join(runs_root, d))]
    )
    for ts in reversed(candidates):
        rd = os.path.join(runs_root, ts)
        ckpt_dir = os.path.join(rd, "checkpoints")
        if not os.path.exists(ckpt_dir):
            continue
        # Has any resume_*.pkl?
        pkls = [f for f in os.listdir(ckpt_dir) if f.startswith("resume_")
                                                  and f.endswith(".pkl")]
        if not pkls:
            continue
        return rd
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Memory-buffer serializer helpers (defensive — buffers vary in class shape)
# ──────────────────────────────────────────────────────────────────────────────

def serialize_memory_buffer(mem) -> Dict[str, Any]:
    """
    Return a picklable representation of either MemoryBuffer or
    ClassBalancedReservoir. We support both via duck-typing.
    """
    out: Dict[str, Any] = {"type": type(mem).__name__}
    # ClassBalancedReservoir
    if hasattr(mem, "per_class"):
        out["per_class"]          = mem.per_class
        out["k"]                  = getattr(mem, "k", 100)
        out["_n_seen_per_class"]  = getattr(mem, "_n_seen_per_class", {})
    # vanilla MemoryBuffer
    elif hasattr(mem, "_X"):
        out["_X"]            = list(mem._X)
        out["_y"]            = list(mem._y)
        out["_class_counts"] = dict(getattr(mem, "_class_counts", {}))
        out["buffer_size_per_class"] = getattr(mem, "buffer_size_per_class", 100)
    return out


def restore_memory_buffer(mem, payload: Dict[str, Any]) -> None:
    """Restore in-place from a serialize_memory_buffer payload."""
    if not payload:
        return
    ty = payload.get("type", "")
    if ty == "ClassBalancedReservoir" and hasattr(mem, "per_class"):
        mem.per_class         = payload.get("per_class",         {})
        mem._n_seen_per_class = payload.get("_n_seen_per_class", {})
        if "k" in payload:
            mem.k = payload["k"]
    elif ty == "MemoryBuffer" and hasattr(mem, "_X"):
        mem._X            = list(payload.get("_X",            []))
        mem._y            = list(payload.get("_y",            []))
        mem._class_counts = dict(payload.get("_class_counts", {}))


def serialize_ewc(ewc) -> Dict[str, Any]:
    """Pickle EWC fisher info (dict of Tensors) + lambda."""
    if ewc is None:
        return {}
    out = {"lambda": getattr(ewc, "lam", getattr(ewc, "lambda_", 1000.0))}
    if hasattr(ewc, "fisher"):
        out["fisher"] = {k: v.cpu() for k, v in ewc.fisher.items()}
    if hasattr(ewc, "params"):
        out["params"] = {k: v.cpu() for k, v in ewc.params.items()}
    return out


def restore_ewc(ewc, payload: Dict[str, Any]) -> None:
    if not payload or ewc is None:
        return
    if "fisher" in payload and hasattr(ewc, "fisher"):
        ewc.fisher = payload["fisher"]
    if "params" in payload and hasattr(ewc, "params"):
        ewc.params = payload["params"]
