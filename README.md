# 🌀 Vectorverse

*A modular framework for building, simulating, and training reinforcement learning agents in batched, differentiable vector-field environments.*

---

## 🌍 Overview

**Vectorverse** provides a unified way to define and simulate *vector-field–based dynamical systems* (ODEs) in **batched** form using **PyTorch**.  
It is designed for scalable RL experiments, physics-informed control, and differentiable system modeling.

- Build continuous-time environments driven by ODEs  
- Simulate many environments in parallel on GPU  
- Integrate seamlessly with **Stable-Baselines3** (`VecEnv` interface)  
- Plug in procedural or fixed target-signal generators  
- Render and analyze episodes interactively in Jupyter or as arrays

---

## ⚙️ Installation

Install the latest version directly from GitHub:

```bash
pip install git+https://github.com/xzlobin/vectorverse.git
```

**Requirements:**
- Python 3.11–3.14  
- torch >= 2.9  
- torchdiffeq >= 0.2.5  
- gymnasium >= 1.2  
- stable-baselines3 >= 2.7  
- matplotlib, numpy  

---

## 🧠 Core Concepts

### Batched Vector Field Environment

The **`BatchedVectorFieldEnv`** is an abstract base class that defines continuous-time environments of the form

$$
\dot{x} = f(x, u)
$$

where

- **x** – system state  
- **u** – control action  
- **f(x, u)** – user-defined vector field  

Each environment runs *N* instances of the system in parallel (GPU-accelerated).

**Key Features**
- Procedural or fixed target generation  
- Time integration via `torchdiffeq.odeint`  
- Structured or flattened observations  
- Vectorized rewards and auto-reset  
- Optional rendering and caching  

---

## 💡 How Reward Is Computed

Vectorverse’s base environment already implements a **shaped tracking reward** in `BatchedVectorFieldEnv.reward_fn(...)`. You do **not** have to write a reward for every environment unless you want something custom.

The default logic is:

1. **Select tracked outputs** with `output_mask` (not all state dims must be tracked).  
2. **Normalize** state and target into $[0, 1]$ using the bounds from `state_box`.  
3. **Compute absolute tracking error** $e = |y - y^*|$.  
4. **Shape the error** with a *Möbius–sigmoid remap* so that:
   - small errors inside the **precision band** mapped to $[0, 0.5]$,
   - bigger errors grow smoothly toward 1,
   - the curve is controlled by:
     - `precision_band` (how much error is “ok”),
     - `precision_steepness` (how sharp the transition is).
5. **Penalize action magnitude** using a Smooth-L1 on the **normalized** action.  
6. **Combine everything** into a final reward in $\approx [0, 1]$:

$$
R = 1 - \text{mean}(\phi(e)) - \lambda \cdot \text{mean}(\text{SmoothL1}(u, 0))
$$

where

- $\phi(e)$ is the Möbius–sigmoid shaped error,  
- $\lambda$ is `regularization`,  
- actions are first normalized by `action_box`.

This gives high reward when:
- outputs track the target closely **and**
- actions stay small.

### Overriding the reward

If you need a different metric, **override `reward_fn(...)` in your subclass**.

Signature in the base class:
```python
def reward_fn(
    self,
    state: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    action: torch.Tensor,
    truncated: torch.Tensor,
    terminated: torch.Tensor,
):
    ...
```

You receive everything you need (current state/target/time/action and done flags) and you can return a tensor of shape `(num_envs,)` with your custom reward.

---

## 🔹 Stable-Baselines3 Adapter

`BatchedToSB3VecEnv` wraps any batched environment (following your `BatchedEnvProto`) and exposes it as a proper `VecEnv` for SB3 algorithms.

- Works with all SB3 agents (PPO, SAC, TD3, etc.)  
- Correctly handles per-env resets  
- Converts torch tensors to numpy automatically  

```python
from vectorverse.batched_adapter import BatchedToSB3VecEnv
from stable_baselines3 import PPO

env = MyBatchedEnv(num_envs=32)
vec_env = BatchedToSB3VecEnv(env)
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

---

## 🎯 Example — Stabilizing an Inverted Pendulum

The included notebook `examples/pendulum.ipynb` shows how to:

- Subclass `BatchedVectorFieldEnv`  
- Define the pendulum’s ODE ($\dot{x} = f(x, u)$)  
- Use a **fixed** target (upright)  
- Train an RL agent to keep it there

---

## 🧩 Components

| Module | Description |
|--------|--------------|
| `vectorverse/batched_vector_field.py` | Core ODE-based environment, reward shaping, and target signal generation |
| `vectorverse/sb3/batched_adapter.py` | SB3 adapter for batched environments (`BatchedToSB3VecEnv`) |
| `vectorverse/sb3/utilities.py` | High-level SB3 builders `get_vec_env` and `get_eval_vec_env` |
| `examples/pendulum.ipynb` | Tutorial notebook for control stabilization |

---

## 🧮 Target Signal Generation

The default signal generator supports:

- **Procedural mode:** piecewise-constant sequences with `n_switches`, `n_bins`  
- **Fixed mode:** manually supplied `(fixed_times, fixed_values)` tensors  

Custom generators can subclass `TargetSignalGeneratorBase`.

---

## 🧰 Utilities

Vectorverse integrates smoothly with **Stable-Baselines3** through its high-level environment builders.

### `get_vec_env`

Creates a **training** vectorized environment with automatic seeding, monitoring, and optional normalization.

```python
from vectorverse.sb3 import get_vec_env

venv = get_vec_env(
    env=env_instance,  # or a string id / callable / gym.Env subclass
    seed=0,
    num_envs=None, # None if already batched instance
    vec_norm=True,
)
```

### `get_eval_vec_env`

Creates an **evaluation** environment that mirrors your training setup and can restore normalization stats.

```python
from vectorverse.sb3 import get_eval_vec_env

eval_env = get_eval_vec_env(
    env=env_instance,
    seed=123,
    num_envs=None,
    vecnorm_path="vecnormalize.pkl",
)
```

---

## 🖼 Rendering

Environments can render in multiple modes:

- `"human"` – live Matplotlib visualization (supports Jupyter)  
- `"rgb_array"` – returns RGB frames for logging or video  
- `"episodes"` – returns full trajectory data (states, actions, rewards, etc.)

---

## 📜 License

MIT License © 2025 Danil Zlobin

---

## 📘 Citation

If you use Vectorverse in academic work, please cite:

```bibtex
@software{vectorverse2025,
  author = {Danil Zlobin},
  title = {Vectorverse: Batched Vector-Field Environments for Reinforcement Learning},
  year = {2025},
  url = {https://github.com/xzlobin/vectorverse}
}
```

---

## 💬 Feedback

Have ideas or feature requests?  
Open an issue or start a discussion on [GitHub Issues](https://github.com/xzlobin/vectorverse/issues).