# batched_to_sb3_vecenv.py
from warnings import warn
import numpy as np
from typing import Any, List, Tuple, Optional

from stable_baselines3.common.vec_env import VecEnv

from .batched_env_proto import (
    BatchedEnvProto,
    SeedableProto,
    RenderableProto,
    ClosableProto,
    split_info_to_list,
    validate_batched_env,
)


class BatchedToSB3VecEnv(VecEnv):
    """
    Adapter that wraps a *batched/scalable* environment (following BatchedEnvProto)
    and exposes a Stable-Baselines3-compatible VecEnv interface.

    - Supports:
        - Proper SB3 per-env auto-reset (resets done sub-envs individually)
        - Optional seeding (`seed`, `seed_all`)
        - Correct terminal observation storage in `infos[i]["terminal_observation"]`
        - Pass-through rendering and closing

    - Expected env interface (BatchedEnvProto):
        num_envs: int
        observation_space: gym.Space
        action_space: gym.Space
        reset() -> (obs_batch, info)
        step(actions) -> (obs_batch, reward_batch, terminated_batch, truncated_batch, info)

    - Optional per-env reset hooks (PerEnvResetProto):
        reset_at(i: int)
        reset_done(done_mask: np.ndarray)
        reset(indices: Sequence[int])

    If none of the above are implemented, we assume the env already returns next-episode
    observations internally after a done step (auto-reset internally).

    Example
    -------
    >>> from batched_env_proto import DummyCountersEnv
    >>> env = DummyCountersEnv(num_envs=8)
    >>> sb3_env = BatchedToSB3VecEnv(env)
    >>> obs = sb3_env.reset()
    >>> sb3_env.step_async(np.zeros((8,)))
    >>> obs, rew, done, infos = sb3_env.step_wait()
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, batched_env: BatchedEnvProto):
        # Validate contract early
        validate_batched_env(batched_env)

        self.env: BatchedEnvProto = batched_env
        self.num_envs = int(batched_env.num_envs)
        self.observation_space = batched_env.observation_space
        self.action_space = batched_env.action_space
        self.single_observation_space = batched_env.observation_space
        self.single_action_space = batched_env.action_space

        # Per-env reset detection
        self._has_reset_at = hasattr(batched_env, "reset_at") and callable(getattr(batched_env, "reset_at"))
        self._has_reset_done = hasattr(batched_env, "reset_done") and callable(getattr(batched_env, "reset_done"))
        self._has_reset_indices = hasattr(batched_env, "reset") and callable(getattr(batched_env, "reset"))

        self._waiting = False
        self._actions_cache = None
        self._warned_no_autoreset = False

    # ---------- SB3 VecEnv API ----------

    def reset(self) -> np.ndarray:
        """Reset all sub-envs (vectorized) and return batched obs."""
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) >= 1:
            obs = out[0]
            info = out[1] if len(out) > 1 else {}
        else:
            obs, info = out, {}
        self._last_info = split_info_to_list(self._to_gym(info), self.num_envs)
        return self._to_gym(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions_cache = actions
        self._waiting = True

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        if not self._waiting:
            raise RuntimeError("step_wait called without step_async() first.")
        self._waiting = False

        obs, rew, terminated, truncated, info = self.env.step(self._actions_cache)

        # Convert to numpy arrays with consistent shapes
        obs = np.asarray(self._to_gym(obs), dtype=self.observation_space.dtype)
        rew = np.asarray(self._to_gym(rew), dtype=np.float32).reshape(self.num_envs,)
        term = np.asarray(self._to_gym(terminated), dtype=bool).reshape(self.num_envs,)
        trunc = np.asarray(self._to_gym(truncated), dtype=bool).reshape(self.num_envs,)
        dones = np.logical_or(term, trunc)
        infos = split_info_to_list(self._to_gym(info), self.num_envs)

        # ---------- Per-env auto-reset ----------
        if np.any(dones):
            # Save terminal obs into infos before overwriting with reset obs
            final_obs = np.array(obs, copy=True)
            done_indices = np.where(dones)[0].tolist()
            for i in done_indices:
                infos[i] = dict(infos[i])
                infos[i]["terminal_observation"] = final_obs[i]

            # Try per-env reset methods in order
            reset_success = False
            exceptions = []
            try:
                if self._has_reset_done:
                    new_obs = self.env.reset_done(dones)
                    new_obs = self._to_gym(new_obs)
                    if new_obs.shape[0] == len(done_indices):
                        for j, idx in enumerate(done_indices):
                            obs[idx] = new_obs[j]
                        reset_success = True
            except Exception as e:
                exceptions.append(e)

            if not reset_success and self._has_reset_at:
                try:
                    for idx in done_indices:
                        obs[idx] = self._to_gym(self.env.reset_at(int(idx)))
                    reset_success = True
                except Exception as e:
                    exceptions.append(e)

            if not reset_success and self._has_reset_indices:
                # Try reset(indices=[...]) or reset([...])
                try:
                    out = None
                    try:
                        out = self.env.reset(indices=done_indices)
                    except TypeError:
                        out = self.env.reset(done_indices)
                    if isinstance(out, tuple):
                        out = out[0]
                    out = self._to_gym(out)
                    if out.shape[0] == len(done_indices):
                        for j, idx in enumerate(done_indices):
                            obs[idx] = out[j]
                        reset_success = True
                except Exception as e:
                    exceptions.append(e)

            # Warn if none worked → assume auto-reset internally
            if not reset_success and not self._warned_no_autoreset:
                warn(
                    "[BatchedToSB3VecEnv] No per-env reset method detected; "
                    "assuming batched env auto-resets internally."
                    " Latest exceptions: " + "; ".join([str(e) for e in exceptions])
                )
                self._warned_no_autoreset = True

        return obs, rew, dones, infos

    # ---------- Optional extensions ----------

    def seed(self, seed: Optional[int] = None) -> None:
        """Propagate seeding to the underlying batched env if supported."""
        if seed is None:
            return
        if isinstance(self.env, SeedableProto):
            if hasattr(self.env, "seed_all") and callable(self.env.seed_all):
                try:
                    self.env.seed_all(seed)
                    return
                except Exception as e:
                    warn(f"[BatchedToSB3VecEnv] Exception during env.seed_all(): {e}")
            if hasattr(self.env, "seed") and callable(self.env.seed):
                try:
                    self.env.seed(seed)
                except Exception as e:
                    warn(f"[BatchedToSB3VecEnv] Exception during env.seed(): {e}")

    def close(self) -> None:
        """Close the underlying batched env if supported."""
        if isinstance(self.env, ClosableProto) and hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception as e:
                warn(f"[BatchedToSB3VecEnv] Exception during env.close(): {e}")

    def render(self, mode: str = "human"):
        """Delegate render call if supported."""
        if isinstance(self.env, RenderableProto) and hasattr(self.env, "render"):
            try:
                return self.env.render(mode=mode)
            except Exception as e:
                warn(f"[BatchedToSB3VecEnv] Exception during env.render(): {e}")
        return None
    
    def _to_gym(self, *args: Any):
        """Convert torch.Tensors in args to numpy arrays recursively."""
        try:
            import torch
        except ImportError:
            torch = None
        out = []
        for a in args:
            if torch is not None and isinstance(a, torch.Tensor):
                out.append(a.cpu().numpy())
            elif isinstance(a, dict):
                dict_keys = a.keys()
                dict_values = [self._to_gym(v) for v in a.values()]
                out.append(dict(zip(dict_keys, dict_values)))
            elif isinstance(a, tuple) or isinstance(a, list):
                list_values = [self._to_gym(v) for v in a]
                out.append(type(a)(list_values))
            else:
                try:
                    out.append(np.asarray(a))
                except Exception as e:
                    raise ValueError(f"Unsupported type {type(a)} in _to_gym conversion.") from e
        return tuple(out) if len(out) > 1 else out[0]
    
    # ---------- Abstract VecEnv plumbing required by SB3 ----------

    def env_is_wrapped(self, wrapper_class, indices=None):
        """
        We don't keep a list of individual Gym envs/wrappers;
        this adapter wraps a single batched env. Report 'False' for all.
        """
        idxs = self._get_indices(indices)
        return [False] * len(idxs)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """
        Call a method on the underlying batched env.
        If indices are given, try common per-index conventions:
          - <method>_at(i, *args, **kwargs)
          - <method>(*args, indices=[i], **kwargs)
          - <method>(i, *args, **kwargs)
        Fallback: call once without indices and broadcast the result.
        """
        idxs = self._get_indices(indices)
        results = []

        # Resolve the base method once if we might broadcast later
        base_fn = getattr(self.env, method_name, None)
        per_env_fn = getattr(self.env, f"{method_name}_at", None)

        if len(idxs) == self.num_envs and per_env_fn is None and base_fn is not None:
            # Likely a batched method; call once and broadcast result(s)
            try:
                out = base_fn(*method_args, **method_kwargs)
                return [out for _ in idxs]
            except TypeError:
                pass  # fall through to per-index loop

        for i in idxs:
            # Try <method>_at(i, ...)
            if callable(per_env_fn):
                results.append(per_env_fn(int(i), *method_args, **method_kwargs))
                continue

            # Try <method>(..., indices=[i])
            if callable(base_fn):
                try:
                    results.append(base_fn(*method_args, indices=[int(i)], **method_kwargs))
                    continue
                except TypeError:
                    pass
                # Try <method>(i, ...)
                try:
                    results.append(base_fn(int(i), *method_args, **method_kwargs))
                    continue
                except TypeError:
                    pass
                # Try <method>(...,) and broadcast
                results.append(base_fn(*method_args, **method_kwargs))
                continue

            raise AttributeError(f"Underlying env has no method '{method_name}'")

        return results

    def get_attr(self, attr_name: str, indices=None):
        """
        Read an attribute from the underlying batched env.
        If the value looks vectorized (length == num_envs), slice per index.
        Otherwise broadcast the same value per requested index.
        """
        idxs = self._get_indices(indices)
        if not hasattr(self.env, attr_name):
            raise AttributeError(f"Underlying env has no attribute '{attr_name}'")

        value = getattr(self.env, attr_name)
        # If it's a callable "getter", call it
        if callable(value):
            value = value()

        # If vector-shaped, pick per index
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == self.num_envs:
            return [value[i] for i in idxs]

        # Otherwise broadcast the single value
        return [value for _ in idxs]

    def set_attr(self, attr_name: str, value, indices=None):
        """
        Set an attribute on the underlying batched env.
        If indices are given and the attribute is vector-shaped, set per index.
        Otherwise set the single attribute/value on the batched env.
        """
        idxs = self._get_indices(indices)

        # If env exposes a native set_attr(indices=...) API, use it
        if hasattr(self.env, "set_attr") and callable(getattr(self.env, "set_attr")):
            try:
                return self.env.set_attr(attr_name, value, indices=idxs)
            except TypeError:
                pass  # fall back to generic handling

        if not hasattr(self.env, attr_name):
            # Create blindly (mirrors Gym behavior)
            if len(idxs) == self.num_envs and isinstance(value, (list, tuple, np.ndarray)):
                # Store a full vector if provided
                setattr(self.env, attr_name, list(value))
            else:
                setattr(self.env, attr_name, value)
            return

        cur = getattr(self.env, attr_name)

        # Vector-shaped attribute → assign per index
        if isinstance(cur, (list, np.ndarray)):
            cur_list = list(cur)
            if isinstance(value, (list, tuple, np.ndarray)):
                for idx, v in zip(idxs, value):
                    cur_list[int(idx)] = v
            else:
                for idx in idxs:
                    cur_list[int(idx)] = value
            setattr(self.env, attr_name, cur_list)
            return

        # Fallback: scalar attribute set once
        setattr(self.env, attr_name, value)
