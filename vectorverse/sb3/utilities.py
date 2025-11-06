# env_builders.py
import random
from typing import Optional, Dict, Any, Callable, Type, Union
from warnings import warn

import numpy as np
import torch

try:
    import gymnasium as gym  # type: ignore  # prefer Gymnasium
except Exception:
    import gym               # type: ignore  # fallback

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    VecEnv,
    DummyVecEnv,
    SubprocVecEnv,
    VecCheckNan,
    VecNormalize,
    VecMonitor,
)

# Your protocol+adapter modules
from .batched_env_proto import (
    SeedableProto,
    validate_batched_env,
)
from .batched_adapter import BatchedToSB3VecEnv


# ======================================================================================
# Public API: seeding
# ======================================================================================

def seed_everything(seed: int) -> None:
    """
    Seed Python/NumPy/PyTorch (CPU+CUDA) for reproducible RL runs.

    Notes
    -----
    - Sets cuDNN to deterministic mode (at a small perf cost).
    - You still must seed environments; the builders below do that for ids/classes/callables.
    """
    if seed is None:
        raise ValueError("seed must be an integer (not None)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # make determinism explicit
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================================
# Helpers to build single-env factories for ids/classes/callables
# ======================================================================================

def _seed_spaces(env: gym.Env, seed: int) -> None:
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        try:
            env.observation_space.seed(seed)
        except Exception:
            # some spaces don't like being seeded
            pass


def make_env_factory_from_id(env_id: str, env_kwargs: Optional[Dict[str, Any]], seed: int) -> Callable[[], gym.Env]:
    """
    Factory for registered ids (Gym/Gymnasium). Applies Monitor + robust seeding.
    """
    env_kwargs = env_kwargs or {}

    def _init():
        env = gym.make(env_id, **env_kwargs)
        # gymnasium-style reset
        try:
            env.reset(seed=seed)
        except TypeError:
            # legacy gym
            if hasattr(env, "seed"):
                env.seed(seed)
        _seed_spaces(env, seed)
        return Monitor(env)

    return _init


def make_env_factory_from_class(env_cls: Type[gym.Env], env_kwargs: Optional[Dict[str, Any]], seed: int) -> Callable[[], gym.Env]:
    """
    Factory for a custom env class (subclass of gym.Env). Applies Monitor + robust seeding.
    """
    env_kwargs = env_kwargs or {}

    def _init():
        env = env_cls(**env_kwargs)
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
        _seed_spaces(env, seed)
        return Monitor(env)

    return _init


def make_env_factory_from_callable(env_fn: Callable[..., gym.Env], env_kwargs: Optional[Dict[str, Any]], seed: int) -> Callable[[], gym.Env]:
    """
    Factory for a custom callable returning a gym.Env. Applies Monitor + robust seeding.
    """
    env_kwargs = env_kwargs or {}

    def _init():
        env = env_fn(**env_kwargs) if env_kwargs else env_fn()
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
        _seed_spaces(env, seed)
        return Monitor(env)

    return _init


# ======================================================================================
# Ready-instance adapter for scalable/batched envs
# ======================================================================================

def _from_ready_env_instance(env_instance: gym.Env | Any) -> VecEnv:
    """
    Wrap an already-constructed environment instance:

    - If it's already a VecEnv → return as-is.
    - If it satisfies BatchedEnvProto (your custom batched env contract) → validate & wrap.
    - Else → wrap the single env with DummyVecEnv + Monitor.
    """
    # already vec
    if isinstance(env_instance, VecEnv):
        return env_instance

    # try your custom batched contract
    if not isinstance(env_instance, gym.Env):
        try:
            validate_batched_env(env_instance)  # will raise AssertionError if not valid
            if isinstance(env_instance, SeedableProto):
                # caller can seed it beforehand; we don't pick a seed here
                pass
            return BatchedToSB3VecEnv(env_instance)  # type: ignore[arg-type]
        except AssertionError as e:
            raise TypeError(
                "The provided env instance does not satisfy the BatchedEnvProto contract "
                "and is not a VecEnv or gym.Env. Cannot wrap."
            ) from e

    # fallback: single non-batched gym.Env → wrap
    def _init():
        # don't double-monitor
        return env_instance if isinstance(env_instance, Monitor) else Monitor(env_instance)

    return DummyVecEnv([_init])


# ======================================================================================
# Public API: unified builders (training + evaluation)
# ======================================================================================

EnvSource = Union[str, Type[gym.Env], Callable[..., gym.Env], gym.Env, VecEnv, Any]


def get_vec_env(
    env: EnvSource,
    env_kwargs: Optional[Dict[str, Any]] = None,
    *,
    seed: int = 0,
    num_envs: int | None = None,
    use_subproc: bool = False,
    vec_norm: bool = True,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
    clip_reward: float = 10.0,
    gamma: float = 0.99,
    monitor_dir: Optional[str] = None,
    already_normalized: bool = False,
) -> VecEnv:
    """
    Create a **training** VecEnv from:
      - a registered id (str),
      - a custom env class (subclass of gym.Env),
      - a custom env factory callable,
      - OR an already-constructed (possibly batched) env instance.

    Features:
      - global seeding
      - proper per-env seeding for id/class/callable
      - adapts custom batched envs via `BatchedToSB3VecEnv`
      - VecCheckNan safety wrapper
      - optional VecNormalize (can be skipped if upstream already normalizes)
    """
    seed_everything(seed)

    # CASE 1: already-constructed instance (gym.Env, VecEnv, or your batched proto)
    is_string = isinstance(env, str)
    is_class = isinstance(env, type) and issubclass(env, gym.Env)
    is_callable = callable(env) and not is_class and not is_string  # avoid double counting class as callable

    if isinstance(env, (gym.Env, VecEnv)) or (not is_string and not is_class and not is_callable):
        venv = _from_ready_env_instance(env)
        # ensure monitoring for ready VecEnvs too
        if isinstance(venv, VecEnv) and not isinstance(venv, VecMonitor):
            venv = VecMonitor(venv)
        # user may have requested a different num_envs → warn
        if num_envs is not None and venv.num_envs != num_envs:
            warn(
                f"The provided env instance has num_envs={venv.num_envs}, "
                f"but num_envs={num_envs} was requested."
            )
        num_envs = venv.num_envs

    # CASE 2: registered id
    elif is_string:
        if num_envs is None:
            num_envs = 1
        vec_cls = SubprocVecEnv if (use_subproc and num_envs > 1) else DummyVecEnv
        venv = make_vec_env(
            env_id=env,
            n_envs=num_envs,
            seed=seed,
            env_kwargs=env_kwargs or {},
            vec_env_cls=vec_cls,
            monitor_dir=monitor_dir,
        )

    # CASE 3: custom class
    elif is_class:
        if num_envs is None:
            num_envs = 1

        def make_one(s):
            return make_env_factory_from_class(env, env_kwargs, s)

        env_fns = [make_one(seed + i) for i in range(num_envs)]
        VecCls = SubprocVecEnv if (use_subproc and num_envs > 1) else DummyVecEnv
        venv = VecCls(env_fns)

    # CASE 4: custom callable
    elif is_callable:
        if num_envs is None:
            num_envs = 1

        def make_one(s):
            return make_env_factory_from_callable(env, env_kwargs, s)

        env_fns = [make_one(seed + i) for i in range(num_envs)]
        VecCls = SubprocVecEnv if (use_subproc and num_envs > 1) else DummyVecEnv
        venv = VecCls(env_fns)

    else:
        raise TypeError(
            "env must be one of: str (id), gym.Env subclass, callable returning gym.Env, "
            "or an already-constructed env (including batched/scalable envs)."
        )

    # Numerical sanity guard
    venv = VecCheckNan(venv, warn_once=True)

    # Optional normalization (skip if scalable env already normalizes)
    if vec_norm and not already_normalized:
        venv = VecNormalize(
            venv,
            training=True,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=gamma,
        )

    return venv


def get_eval_vec_env(
    env: EnvSource,
    env_kwargs: Optional[Dict[str, Any]] = None,
    *,
    seed: int = 12345,
    num_envs: int | None = None,
    vecnorm_path: Optional[str] = None,
    use_subproc: bool = False,
    norm_obs: bool = True,
    already_normalized: bool = False,
) -> VecEnv:
    """
    Create an **evaluation** VecEnv mirroring `get_vec_env`, with optional VecNormalize restore.

    If `vecnorm_path` is provided, the function loads running stats and **freezes** normalization:
        env.training = False
        env.norm_reward = False
        env.norm_obs = norm_obs
    """
    seed_everything(seed)

    is_string = isinstance(env, str)
    is_class = isinstance(env, type) and issubclass(env, gym.Env)
    is_callable = callable(env) and not is_class and not is_string

    # Build base VecEnv WITHOUT VecNormalize first
    if isinstance(env, (gym.Env, VecEnv)) or (not is_string and not is_class and not is_callable):
        venv = _from_ready_env_instance(env)
        if isinstance(venv, VecEnv) and not isinstance(venv, VecMonitor):
            venv = VecMonitor(venv)

    elif is_string:
        if num_envs is None:
            num_envs = 1
        vec_cls = SubprocVecEnv if (use_subproc and num_envs > 1) else DummyVecEnv
        venv = make_vec_env(
            env_id=env,
            n_envs=num_envs,
            seed=seed,
            env_kwargs=env_kwargs or {},
            vec_env_cls=vec_cls,
            monitor_dir=None,
        )

    elif is_class:
        if num_envs is None:
            num_envs = 1

        def make_one(s):
            return make_env_factory_from_class(env, env_kwargs, s)

        env_fns = [make_one(seed + i) for i in range(num_envs)]
        VecCls = SubprocVecEnv if (use_subproc and num_envs > 1) else DummyVecEnv
        venv = VecCls(env_fns)

    elif is_callable:
        if num_envs is None:
            num_envs = 1

        def make_one(s):
            return make_env_factory_from_callable(env, env_kwargs, s)

        env_fns = [make_one(seed + i) for i in range(num_envs)]
        VecCls = SubprocVecEnv if (use_subproc and num_envs > 1) else DummyVecEnv
        venv = VecCls(env_fns)

    else:
        raise TypeError(
            "env must be one of: str (id), gym.Env subclass, callable returning gym.Env, "
            "or an already-constructed env (including batched/scalable envs)."
        )

    venv = VecCheckNan(venv, warn_once=True)

    # If env already normalizes, or no stats to restore → return as-is
    if already_normalized or vecnorm_path is None:
        return venv

    # Restore VecNormalize stats and freeze for eval
    env_norm = VecNormalize.load(vecnorm_path, venv)
    env_norm.training = False
    env_norm.norm_reward = False
    env_norm.norm_obs = bool(norm_obs)
    return env_norm
