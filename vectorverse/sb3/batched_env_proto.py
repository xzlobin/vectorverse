# batched_env_proto.py
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np

try:
    import gymnasium as gym # type: ignore # Prefer Gymnasium
except Exception:
    import gym # type: ignore # fallback


# =========================
# Core protocols (interfaces)
# =========================

@runtime_checkable
class BatchedEnvProto(Protocol):
    """
    Protocol for a *scalable/batched* environment that manages N logical sub-envs inside ONE object
    and emits vectorized (batched) arrays. Compatible with a VecEnv adapter.

    REQUIRED ATTRIBUTES (single-env spaces):
        num_envs: int
        observation_space: gym.Space  # for ONE sub-env (not batched)
        action_space: gym.Space       # for ONE sub-env (not batched)

    REQUIRED METHODS (Gymnasium-style):
        reset(seed: Optional[int] = None) -> (obs_batch, info)
            obs_batch shape: (num_envs, *obs_shape)
            info: dict of arrays (first dim num_envs) OR list[dict] of len num_envs

        step(actions: np.ndarray) -> (obs, reward, terminated, truncated, info)
            actions shape: (num_envs, *action_shape)
            obs        : (num_envs, *obs_shape)
            reward     : (num_envs,)
            terminated : (num_envs,) bool
            truncated  : (num_envs,) bool
            info       : dict of arrays OR list[dict] (len num_envs)
    """

    # --- required attributes ---
    num_envs: int
    observation_space: gym.Space
    action_space: gym.Space

    # --- required methods ---
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Union[Dict[str, Any], List[Dict[str, Any]]]]: ...
    def step(
        self, actions: np.ndarray
    ) -> Tuple[
        np.ndarray,              # obs batch
        np.ndarray,              # reward batch
        np.ndarray,              # terminated mask
        np.ndarray,              # truncated mask
        Union[Dict[str, Any], List[Dict[str, Any]]],  # info
    ]: ...


@runtime_checkable
class PerEnvResetProto(Protocol):
    """
    OPTIONAL per-env reset hooks (any ONE is sufficient).
    The VecEnv adapter will use whichever is available to implement *SB3 per-env auto-reset*.

    Provide at least one:
        reset_at(index: int, seed: Optional[int] = None) -> obs_i
        reset_done(done_mask: np.ndarray) -> obs_batch
        reset(indices: Sequence[int]) -> obs_batch    # may also return (obs_batch, info)
    """
    def reset_at(self, index: int, seed: Optional[int] = None) -> np.ndarray: ...
    def reset_done(self, done_mask: np.ndarray) -> np.ndarray: ...
    def reset(self, indices: Sequence[int]) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]: ...


@runtime_checkable
class SeedableProto(Protocol):
    """OPTIONAL global/per-env seeding hooks."""
    def seed(self, seed: int) -> None: ...
    def seed_all(self, seed: int) -> None: ...


@runtime_checkable
class RenderableProto(Protocol):
    """OPTIONAL render passthrough."""
    def render(self, mode: str = "human") -> Any: ...


@runtime_checkable
class ClosableProto(Protocol):
    """OPTIONAL cleanup hook."""
    def close(self) -> None: ...


# =========================
# Runtime validators (helpful errors)
# =========================
def _maybe_move_to_cpu(*x: Any) -> Any:
    """If x is a torch.Tensor, move to CPU and convert to numpy."""
    res = []
    try:
        import torch
        for xi in x:
            if isinstance(xi, torch.Tensor):
                res.append(xi.cpu())
            else:
                res.append(xi)
    except ImportError:
        pass
    return tuple(res) if len(x) > 1 else res[0]

def validate_batched_env(env: Any) -> None:
    """
    Assert that `env` conforms to BatchedEnvProto at runtime (lightweight checks).

    Raises:
        AssertionError with a precise message when something is off.
    """
    assert isinstance(env, BatchedEnvProto), (
        "Object does not satisfy BatchedEnvProto (missing attrs/methods or wrong signatures)."
    )
    assert isinstance(env.num_envs, int) and env.num_envs > 0, "num_envs must be a positive int"
    # Spaces
    assert hasattr(env, "observation_space") and isinstance(env.observation_space, gym.Space), \
        "observation_space must be a gym.Space (single-env space)"
    assert hasattr(env, "action_space") and isinstance(env.action_space, gym.Space), \
        "action_space must be a gym.Space (single-env space)"

    # Quick smoke test: reset shapes
    ob, info = env.reset()
    ob = np.asarray(_maybe_move_to_cpu(ob))
    assert ob.shape[0] == env.num_envs, "reset() must return batched obs: shape[0] == num_envs"
    # Step signature/shape smoke test (zero/first action)
    if env.action_space.shape:
        example_actions = np.zeros((env.num_envs, *env.action_space.shape), dtype=np.int64)
    else:
        # Discrete or scalar action; use integers
        example_actions = np.zeros((env.num_envs,), dtype=np.int64)
    step_out = env.step(example_actions)
    assert isinstance(step_out, tuple) and len(step_out) == 5, "step() must return 5-tuple"
    obs, rew, term, trunc, info2 = step_out
    obs, rew, term, trunc = map(np.asarray, _maybe_move_to_cpu(obs, rew, term, trunc))
    assert obs.shape[0] == env.num_envs, "step(): obs must be batched with shape[0] == num_envs"
    assert rew.shape == (env.num_envs,), "step(): reward must be shape (num_envs,)"
    assert term.shape == (env.num_envs,), "step(): terminated must be shape (num_envs,)"
    assert trunc.shape == (env.num_envs,), "step(): truncated must be shape (num_envs,)"


def has_per_env_reset(env: Any) -> bool:
    """
    Return True if env offers ANY per-env reset API (reset_at / reset_done / reset(indices)).
    """
    if hasattr(env, "reset_at") and callable(getattr(env, "reset_at")):
        return True
    if hasattr(env, "reset_done") and callable(getattr(env, "reset_done")):
        return True
    # Only treat `reset` as per-env if it looks like it accepts indices
    sig = getattr(env, "reset", None)
    if callable(sig):
        # heuristic: check code object or annotations if available
        ann = getattr(sig, "__annotations__", {})
        if "indices" in ann or getattr(sig, "__code__", None) and "indices" in sig.__code__.co_varnames:
            return True
    return False


# =========================
# Convenience shims/helpers
# =========================

def split_info_to_list(info: Union[Dict[str, Any], List[Dict[str, Any]]], n: int) -> List[Dict[str, Any]]:
    """
    Convert a dict of batched arrays into a list[dict] of length n (or pass-through if already a list).
    """
    if isinstance(info, list):
        assert len(info) == n, "info list length must equal num_envs"
        return info
    if isinstance(info, dict):
        out: List[Dict[str, Any]] = []
        for i in range(n):
            di = {}
            for k, v in info.items():
                try:
                    di[k] = v[i] if hasattr(v, "__getitem__") else v
                except Exception:
                    di[k] = v
            out.append(di)
        return out
    return [{} for _ in range(n)]


# =========================
# Minimal example to depend on
# =========================

class DummyCountersEnv(BatchedEnvProto, SeedableProto):
    """
    Minimal batched env example implementing REQUIRED methods.
    - num_envs independent counters terminate when reaching a random horizon.
    - Provides seed(seed) for determinism.
    - DOES NOT provide per-env reset hooks — adapter will assume step() returns next-episode obs
      OR you can add reset_at/reset_done to enable explicit per-env resets.
    """

    def __init__(self, num_envs: int = 8, max_horizon: int = 20):
        self.num_envs = int(num_envs)
        self.max_horizon = int(max_horizon)

        self.observation_space = gym.spaces.Box(low=0, high=max_horizon, shape=(1,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(2)

        self._rng = np.random.default_rng(0)
        self._t = np.zeros((self.num_envs,), dtype=np.int32)
        self._h = self._rng.integers(5, self.max_horizon + 1, size=(self.num_envs,), dtype=np.int32)

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _obs(self) -> np.ndarray:
        return self._t.reshape(self.num_envs, 1)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed(seed)
        self._t[:] = 0
        self._h = self._rng.integers(5, self.max_horizon + 1, size=(self.num_envs,), dtype=np.int32)
        return self._obs(), {}

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions).reshape(self.num_envs, *(() if self.action_space.shape is None else self.action_space.shape))
        inc = (actions != 0).astype(np.int32)
        self._t += inc

        terminated = self._t >= self._h
        truncated = np.zeros_like(terminated, dtype=bool)

        # Reward: +1 only when episode ends
        reward = np.where(terminated, 1.0, 0.0).astype(np.float32)

        # Auto-reset internally for demo (so adapter can pass-through if no per-env hook)
        if np.any(terminated):
            # store final obs, then reset those slots
            term_idx = np.where(terminated)[0]
            self._t[term_idx] = 0
            self._h[term_idx] = self._rng.integers(5, self.max_horizon + 1, size=term_idx.size, dtype=np.int32)

        obs = self._obs()
        info: Dict[str, Any] = {}  # could also be list[dict] of len num_envs
        return obs, reward, terminated, truncated, info
