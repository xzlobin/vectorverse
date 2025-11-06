import torch
import numpy as np
import torch.nn.functional as F

try:
    import gymnasium as gym  # type: ignore
except Exception:
    import gym  # type: ignore

from torchdiffeq import odeint
from typing      import Any, Iterable, Mapping, Sequence, Tuple, Type
from collections import deque
from abc         import ABC, abstractmethod
from warnings    import warn


# =========================
# Abstract Base Generator
# =========================
class TargetSignalGeneratorBase(ABC):
    """
    Interface for target-signal generators. Implementations must be batched over envs.

    Recommended constructor kwargs (the env will inject sensible defaults if missing):
      - num_envs: int
      - Tf: float
      - target_box: torch.Tensor
      - device: torch.device
      - rng: torch.Generator

    Optional (for default generator):
      - n_switches: int
      - n_bins: int
      - fixed_times:  torch.Tensor (see DefaultTargetSignalGenerator for shapes)
      - fixed_values: torch.Tensor (see DefaultTargetSignalGenerator for shapes)
    """

    @abstractmethod
    def regenerate(self, indices: torch.Tensor | None = None) -> None:
        """(Re)generate trajectories for selected env indices (or all if None)."""
        ...

    @abstractmethod
    def get(self, t: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        """Return targets at time t (scalar or per-env). Shape: (n_envs, *target_shape)."""
        ...

    @abstractmethod
    def reseed(self, rng: torch.Generator) -> None:
        """Replace RNG and mark cached trajectories stale if appropriate."""
        ...

    @abstractmethod
    def update_params(self, **kwargs) -> None:
        """
        Update parameters dynamically. Common keys:
          - num_envs, Tf, target_box, device, rng
          - (default impl also accepts n_switches, n_bins, fixed_times, fixed_values)
        """
        ...


# =========================
# Default Generator
# =========================
class DefaultTargetSignalGenerator(TargetSignalGeneratorBase):
    """
    Piecewise-constant target generator with two modes:

    1) Procedural mode (no fixed_* provided):
       - Uses n_switches (>=0), n_bins (>0), Tf, target_box to sample bins and values.
       - n_ticks = n_switches + 1

    2) Fixed mode (fixed_times & fixed_values provided in kwargs):
       - Accepts:
           fixed_times:  (n_ticks,) or (1, n_ticks) or (num_envs, n_ticks)
           fixed_values: (n_ticks, *shape) or (1, n_ticks, *shape) or (num_envs, n_ticks, *shape)
         and broadcasts across num_envs as needed.
       - Values are used as-is (no scaling by target_box). Times are absolute in [0, Tf] domain;
         if you pass normalized times in [0,1], multiply by Tf before passing.

    Constructor kwargs (all optional except num_envs/Tf/target_box/device/rng which the env supplies):
      - n_switches (default: 0)
      - n_bins     (default: 10)
      - fixed_times, fixed_values (switches to Fixed mode if both provided)
    """

    def __init__(self, *,
                 num_envs: int,
                 Tf: float,
                 target_box: torch.Tensor,
                 device: torch.device,
                 rng: torch.Generator,
                 n_switches: int = 0,
                 n_bins: int = 10,
                 fixed_times: torch.Tensor | None = None,
                 fixed_values: torch.Tensor | None = None):

        self.num_envs   = int(num_envs)
        self.Tf         = float(Tf)
        self.target_box = target_box
        self.device     = device
        self.rng        = rng

        # procedural params
        self.n_switches = int(n_switches)
        self.n_bins     = int(n_bins)

        # fixed mode (if both present)
        self._fixed_times  = fixed_times
        self._fixed_values = fixed_values

        self._times  : torch.Tensor | None = None   # (N, T)
        self._values : torch.Tensor | None = None   # (N, T, *shape)
        self._ready  : bool = False

        self._validate_mode_and_buffers_init()

    # ---------- public API ----------
    def reseed(self, rng: torch.Generator) -> None:
        self.rng = rng
        self._ready = False

    def update_params(self, **kwargs) -> None:
        """
        Accepts partial updates; marks stale/reallocates as needed.
        Supported keys: num_envs, Tf, target_box, device, rng, n_switches, n_bins, fixed_times, fixed_values
        """
        changed_shape = False

        if 'num_envs' in kwargs and int(kwargs['num_envs']) != self.num_envs:
            self.num_envs = int(kwargs['num_envs'])
            changed_shape = True
        if 'Tf' in kwargs and float(kwargs['Tf']) != self.Tf:
            self.Tf = float(kwargs['Tf'])
            self._ready = False
        if 'target_box' in kwargs and kwargs['target_box'] is not self.target_box:
            self.target_box = kwargs['target_box']
            changed_shape = True  # feature shape might change
        if 'device' in kwargs and kwargs['device'] != self.device:
            self.device = kwargs['device']
            # migrate buffers
            if self._times is not None:  
                self._times  = self._times.to(self.device)
            if self._values is not None: 
                self._values = self._values.to(self.device)
        if 'rng' in kwargs and kwargs['rng'] is not self.rng:
            self.rng = kwargs['rng']
            self._ready = False

        # Procedural params
        if 'n_switches' in kwargs:
            self.n_switches = int(kwargs['n_switches'])
            self._ready = False
            changed_shape = True  # n_ticks changes
        if 'n_bins' in kwargs:
            self.n_bins = int(kwargs['n_bins'])
            self._ready = False

        # Fixed mode params
        if 'fixed_times' in kwargs:
            self._fixed_times = kwargs['fixed_times']
            self._ready = False
            changed_shape = True
        if 'fixed_values' in kwargs:
            self._fixed_values = kwargs['fixed_values']
            self._ready = False
            changed_shape = True

        if changed_shape:
            self._times  = None
            self._values = None

        self._validate_mode_and_buffers_init()

    def regenerate(self, indices: torch.Tensor | None = None) -> None:
        """(Re)fill buffers for selected envs (or all)."""
        # If fixed mode is active, just (re)broadcast into buffers and return.
        if self._is_fixed_mode():
            self._ensure_buffers_fixed()
            self._ready = True
            return

        # Procedural mode
        if self.n_bins <= 0:
            raise ValueError("n_bins must be positive in procedural mode.")

        if (indices is None) or (not self._ready):
            indices = torch.arange(self.num_envs, device=self.device)

        self._ensure_buffers_procedural()

        n_ticks  = self._n_ticks_proc()
        interval = 1.0 / (self.n_switches + 1) if self.n_switches >= 0 else 1.0
        base = torch.arange(0., 1., step=interval, device=self.device, dtype=torch.float32)[:n_ticks]

        jitter_scale = 0.0 if self.n_switches == 0 else (1.0 / self.n_switches)
        jitter = interval * jitter_scale * torch.rand((indices.shape[0], n_ticks),
                                                      device=self.device, generator=self.rng)
        jitter[:, 0] = 0.0  # first tick at t=0
        time_ticks = base.view(1, -1) + jitter
        time_ticks, _ = torch.sort(time_ticks, dim=1)  # normalized [0,1)
        time_ticks = time_ticks * self.Tf               # scale to [0,Tf]

        # bins with no repetition
        target_bins = []
        last_bin = None
        for _ in range(n_ticks):
            last_bin = self._sel_next_target_bin(prev_bins=last_bin, n_envs=indices.shape[0])
            target_bins.append(last_bin)
        target_bins = torch.stack(target_bins, dim=1)  # (n_envs, n_ticks, *shape)

        # sample within each bin ∈ [0,1)
        unit = target_bins.float()/self.n_bins + (1/self.n_bins) * torch.rand(
            target_bins.shape, dtype=torch.float32, device=self.device, generator=self.rng
        )

        # scale to target_box range
        t_min = self.target_box.min(dim=0).values
        t_max = self.target_box.max(dim=0).values
        self._values[indices] = t_min + unit * (t_max - t_min)
        self._times[indices]  = time_ticks
        self._ready = True

    def get(self, t: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        """Return targets at time t for selected envs. t: scalar or (n_envs,)."""
        if not self._ready:
            self.regenerate(indices=None)
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)

        values = self._values[indices, ...]   # (n_envs, n_ticks, *shape)
        times  = self._times[indices, ...]    # (n_envs, n_ticks)

        t = torch.as_tensor(t, device=times.device, dtype=times.dtype).view(-1)
        if t.numel() == 1:
            t = t.expand(values.shape[0])
        if t.shape[0] != values.shape[0]:
            raise ValueError(f"t must have length {values.shape[0]}; got {t.shape[0]}.")

        idx = (t.view(-1, 1) >= times).sum(dim=1) - 1
        idx = idx.clamp(min=0, max=times.shape[1]-1).long()
        env_idx = torch.arange(values.shape[0], device=values.device, dtype=torch.long)
        return values[env_idx, idx]

    # ---------- internals ----------
    def _is_fixed_mode(self) -> bool:
        return (self._fixed_times is not None) and (self._fixed_values is not None)

    def _n_ticks_proc(self) -> int:
        return int(self.n_switches) + 1

    def _ensure_buffers_procedural(self) -> None:
        n_ticks = self._n_ticks_proc()
        tshape  = self.target_box.shape[1:]
        need = (
            self._times  is None or
            self._values is None or
            self._times.shape  != (self.num_envs, n_ticks) or
            self._values.shape != (self.num_envs, n_ticks, *tshape)
        )
        if need:
            self._times  = torch.empty((self.num_envs, n_ticks), device=self.device)
            self._values = torch.empty((self.num_envs, n_ticks, *tshape), device=self.device)
            self._ready  = False

    def _ensure_buffers_fixed(self) -> None:
        """
        Broadcast user-provided fixed_times/values to (num_envs, n_ticks, *shape),
        move to device, and populate internal buffers.
        """
        if not self._is_fixed_mode():
            return

        ft = torch.as_tensor(self._fixed_times, device=self.device, dtype=torch.float32)
        fv = torch.as_tensor(self._fixed_values, device=self.device, dtype=torch.float32)

        # Normalize shapes:
        # ft: (n_ticks,) or (1, n_ticks) or (num_envs, n_ticks)
        if ft.ndim == 1:
            ft = ft.view(1, -1)                           # (1, n_ticks)
        elif ft.ndim != 2:
            raise ValueError("fixed_times must be (n_ticks,) or (1, n_ticks) or (num_envs, n_ticks)")
        if ft.shape[0] == 1 and self.num_envs > 1:
            ft = ft.expand(self.num_envs, ft.shape[1])
        if ft.shape[0] != self.num_envs:
            raise ValueError(f"fixed_times first dim must be num_envs={self.num_envs} or 1.")

        # fv: (n_ticks, *shape) or (1, n_ticks, *shape) or (num_envs, n_ticks, *shape)
        if fv.ndim >= 1 and fv.shape[0] != self.num_envs:
            if fv.ndim >= 2 and fv.shape[0] == 1:
                fv = fv.expand(self.num_envs, *fv.shape[1:])  # (num_envs, n_ticks, *shape)
            else:
                # maybe (n_ticks, *shape) → add env dim
                fv = fv.unsqueeze(0)  # (1, n_ticks, *shape)
                fv = fv.expand(self.num_envs, *fv.shape[1:])
        if fv.shape[0] != self.num_envs:
            raise ValueError(f"fixed_values first dim must be num_envs={self.num_envs} or 1 (or missing env dim).")

        # allocate buffers to match shapes
        n_ticks = ft.shape[1]
        tshape  = fv.shape[2:]

        if tshape != self.target_box.shape[1:]:
            hint = ""
            if len(tshape) == 0 and len(self.target_box.shape[1:]) == 1:
                hint = "Did you forget to unsqueeze the feature dim?"
            raise ValueError(f"fixed_values shape must match target_box shape; "
                             f"got {tshape} vs {self.target_box.shape[1:]}. {hint}")

        need = (
            self._times  is None or
            self._values is None or
            self._times.shape  != (self.num_envs, n_ticks) or
            self._values.shape != (self.num_envs, n_ticks, *tshape)
        )
        if need:
            self._times  = torch.empty((self.num_envs, n_ticks), device=self.device)
            self._values = torch.empty((self.num_envs, n_ticks, *tshape), device=self.device)

        # copy in (assume times are already in absolute seconds; if normalized, user can pre-scale)
        self._times.copy_(ft)
        self._values.copy_(fv)

    def _validate_mode_and_buffers_init(self) -> None:
        if self._is_fixed_mode():
            # Fixed mode: simply broadcast into buffers
            self._ensure_buffers_fixed()
            self._ready = True
        else:
            # Procedural mode: ensure buffers consistent (not filled yet)
            self._ensure_buffers_procedural()
            self._ready = False

    def _sel_next_target_bin(self, prev_bins: torch.Tensor | None, n_envs: int) -> torch.Tensor:
        device = self.device
        dtype  = torch.long
        target_shape = self.target_box.shape[1:]
        shape = (n_envs, *target_shape)
        next_bins = torch.randint(0, self.n_bins, size=shape, device=device, generator=self.rng, dtype=dtype)
        if prev_bins is None:
            return next_bins
        same_mask = next_bins == prev_bins
        if same_mask.any():
            resample = torch.randint(0, self.n_bins - 1, size=same_mask.sum().shape,
                                     device=device, generator=self.rng, dtype=dtype)
            prev_vals = prev_bins[same_mask]
            resample = resample + (resample >= prev_vals).long()
            next_bins[same_mask] = resample
        return next_bins


# =========================
# Environment 
# =========================
class BatchedVectorFieldEnv(ABC):
    r"""
    Batched ODE-based control environment with vector-field dynamics, target tracking,
    and flexible (procedural or fixed) target signal generation.

    This environment integrates a time-invariant vector field
    :math:`\dot{x} = f(x, u)` in *batch* (``num_envs`` parallel environments) using
    `torchdiffeq.odeint`. At each step, it computes a tracking reward between the
    masked outputs of the state and a target signal, with optional action
    regularization. Observations can be flattened or structured (dict).

    Target signals are produced by a pluggable generator implementing
    :class:`TargetSignalGeneratorBase`. By default, the environment uses
    :class:`DefaultTargetSignalGenerator`, which supports:

      • **Procedural mode** (piecewise-constant): driven by ``n_switches`` and ``n_bins``  
      • **Fixed mode**: user-provided time ticks and values (with broadcasting support)

    You select and configure the generator via:

      • ``signal_gen_cls`` (constructor arg): a class implementing
        :class:`TargetSignalGeneratorBase` (defaults to
        :class:`DefaultTargetSignalGenerator`)

      • ``signal_gen_kwargs`` (abstract @property): a mapping of kwargs forwarded to
        the generator. The env injects defaults for ``num_envs``, ``Tf``, ``target_box``,
        ``device``, and ``rng``. If you don't provide fixed signals, the default
        generator also accepts ``n_switches`` (>=0) and ``n_bins`` (>0).

    ----------
    Core flow
    ----------
    • :meth:`step(actions)`: integrates the ODE for duration ``dt`` (single step),
      updates time, queries the target generator at the new time, builds observations,
      computes rewards, and updates a render cache if enabled.

    • :meth:`reset(indices=None)`: initializes state/time/target for all or selected
      environments, applies reset noise to the masked outputs, clamps within
      ``state_box``, and (re)initializes rendering buffers as needed.

    • :meth:`seed(seed)`: reseeds the RNG and reseeds the target generator.

    ----------
    Spaces & shapes
    ----------
    • **State**: real tensor with shape ``(num_envs, *state_box.shape[1:])``.  
      Bounds are given by ``state_box.min(dim=0)/max(dim=0)``.

    • **Action**: real tensor with shape ``(num_envs, *action_box.shape[1:])``.
      The Gym action space is built from ``action_box`` bounds.

    • **Target**: real tensor with shape ``(num_envs, *target_box.shape[1:])``,
      produced by the target generator.

    • **Observation**:
        - If ``flatten_obs=True``: concatenated flattened state and target  
          shape ``(num_envs, state_flat + target_flat)`` with a Box space.
        - Else: dict with keys ``{"state": ..., "target": ...}`` and per-field Box spaces.

    ----------
    Reward
    ----------
    At each step, the environment rewards accurate tracking of the **masked**
    state outputs with a smooth shaping curve and penalizes large actions.

    The per-environment reward is computed as:

        R = 1 - ⟨φ(e; r, k)⟩ - λ ⟨SmoothL1(ũ, 0; β)⟩

    where:

      • e  = |ỹ - ỹ*|                - normalized absolute tracking error  
      • φ(e; r, k)                    - Möbius-sigmoid shaping function  
      • λ  = regularization coefficient  
      • β  = 0.1                      - Smooth-L1 width  
      • ỹ, ỹ*                       - normalized state and target  
      • ũ                            - normalized action


    Normalization:

    The tracked outputs are normalized using state bounds:

        ỹ  = (y - s_min) / (s_max - s_min)
        ỹ* = (y* - s_min) / (s_max - s_min)

    and actions using action bounds:

        ũ = (u - u_min) / (u_max - u_min)


    Shaping via Möbius-sigmoid remap:

    Each component of e ∈ [0, 1] is remapped with a smooth, monotone shaping
    function φ(e; r, k) built from a logistic centered around `precision_band`
    (r) followed by a Möbius transform enforcing:

        φ(0) = 0,   φ(r) = 0.5,   φ(1) = 1

    The parameters:
      • precision_band (r): defines the “half-credit” error band  
      • precision_steepness (k): controls sharpness; larger → more abrupt

    The mean of φ(e; r, k) across masked outputs represents the shaped tracking
    error term. The reward decreases smoothly as tracking error grows beyond r.

    Action regularization:

    A Smooth-L1 penalty is applied to the normalized action:

        P_act = ⟨SmoothL1(ũ, 0; β)⟩

    The regularization term is scaled by λ = `regularization`.


    Final reward:

    The final per-step reward is:

        R = 1 - ⟨φ(e; r, k)⟩ - λ ⟨SmoothL1(ũ, 0; β)⟩

    • The maximum reward (≈1) occurs when tracking is perfect and actions are small.  
    • The shaped tracking term lies in [0, 1].  
    • The action term is always non-negative.  
    • `terminated` and `truncated` are passed to `reward_fn` but not used by the
      base implementation.


    Tuning:

    • precision_band: sets tolerated normalized error (e.g. r=0.05 → ±0.2 if state
      range is [-2, 2]).  
    • precision_steepness: choose 3-8 for gradual to sharp shaping.  
    • regularization: penalizes action magnitude; set to 0 to disable.

    ----------
    Target signal generation
    ----------
    The target generator must implement :class:`TargetSignalGeneratorBase` and supports:

    • **Querying**: ``generator.get(t, indices=None)`` returns the target at time ``t``  
      (scalar or per-env vector) for selected envs. Output shape matches
      ``(n_selected_envs, *target_box.shape[1:])``.

    • **Regeneration**: ``generator.regenerate(indices=None)`` (re)builds internal
      time/value grids (used automatically as needed by the default generator).

    • **Reseeding/updates**: ``generator.reseed(rng)`` and
      ``generator.update_params(**kwargs)`` keep the generator in sync with the env.

    The default generator supports two modes:

      1) **Procedural mode** (no fixed tensors provided)
         - ``n_switches`` (>= 0): number of switches; number of constant segments is
           ``n_switches + 1``.
         - ``n_bins`` (> 0): discrete bins to sample segment values with no immediate
           repetition per element.
         - Values are sampled within each bin and then linearly scaled into the
           elementwise range implied by ``target_box.min/max``.

      2) **Fixed mode** (provide both of the following in ``signal_gen_kwargs``)
         - ``fixed_times``: one of
             * ``(n_ticks,)`` or ``(1, n_ticks)`` → broadcast to all envs, or
             * ``(num_envs, n_ticks)`` for per-env times.
           Times are interpreted as **absolute** (typically in [0, Tf]). If your times
           are normalized in [0, 1], scale by ``Tf`` before passing.
         - ``fixed_values``: one of
             * ``(n_ticks, *target_shape)`` or ``(1, n_ticks, *target_shape)`` → broadcast, or
             * ``(num_envs, n_ticks, *target_shape)`` for per-env values.
           Values are **used as-is** (no scaling by ``target_box``).

    The env forwards ``signal_gen_kwargs`` into ``signal_gen_cls(**kwargs)`` and injects
    defaults: ``num_envs``, ``Tf``, ``target_box``, ``device``, and ``rng``. If you don’t
    provide fixed signals, the default generator also accepts ``n_switches`` and
    ``n_bins`` (with defaults 0 and 10).

    ----------
    Dynamics & integration
    ----------
    • Dynamics are defined by the concrete implementation of
      :meth:`vector_field(state, action) -> dstate/dt`.

    • The integrator is ``solver = torchdiffeq.odeint`` by default; override the class
      attribute if needed. Extra solver options come from ``solver_kwargs`` (abstract
      @property).

    • The environment is time-invariant (the vector field signature is ``f(x, u)``).
      Time is maintained per-env and passed only to the target generator.

    ----------
    Termination & truncation
    ----------
    After each step:

      • ``truncated = (time >= Tf)``  
      • ``terminated`` is true if:
          - any element of the state is non-finite, or
          - the state leaves the bounds implied by ``state_box``.

    ----------
    Rendering
    ----------
    Set ``render_mode`` to one of:
      • ``None`` (default): no rendering  
      • ``"human"``: interactive Matplotlib figure (reused in notebooks, non-blocking
        window outside notebooks)  
      • ``"rgb_array"``: returns a ``(H, W, 3)`` uint8 array of the composed figure  
      • ``"episodes"``: returns buffered episode data (states/targets/actions/rewards/
        times/flags) for external processing/animation

    Rendering buffers are maintained per env with bounded size (``render_nmax``) and a
    small FIFO queue of finished episodes (``render_qmax``).

    ----------
    Parameters
    ----------
    num_envs : int
        Number of environments in the batch.
    dt : float
        Integration step in seconds (passed as the spacing between the two time points
        to the ODE solver each :meth:`step` call).
    Tf : float
        Episode horizon in seconds. Past this time, episodes are marked ``truncated``.
    regularization : float
        L2 penalty coefficient on actions used in the reward.
    reset_noise : float
        Amplitude of zero-mean uniform noise applied to the **masked** output components
        at reset, scaled elementwise by ``init_state_box``.
    precision_band : float, optional
        Bandwidth of the reward shaping, inside the band reward scaled from [0, 0.5] outside lie the
        rest of value, precision band defines tracking capability of the agent.
    precision_steepness : float, optional
        Steepness of the reward shaping curve, higher values result in a more aggressive shaping.
        From 1 to inf.
    device : Any, optional
        Torch device (e.g., ``"cpu"``, ``"cuda:0"``). Defaults to CPU.
    render_mode : str | None, optional
        One of {``None``, ``"human"``, ``"rgb_array"``, ``"episodes"``}.
    render_nmax : int, optional
        Maximum number of time samples stored per episode per env for rendering.
    render_qmax : int, optional
        Maximum number of finished episodes held in the render queue.
    flatten_obs : bool, optional
        If True, observations are a single flat Box; otherwise a dict with separate
        state/target Boxes.
    signal_gen_cls : Type[TargetSignalGeneratorBase], optional
        Generator class used to produce target signals (defaults to
        :class:`DefaultTargetSignalGenerator`).

    ----------
    Abstract properties/methods to implement
    ----------
    state_box : torch.Tensor
        Tensor specifying per-element state bounds with shape ``(2, *state_shape)``,
        or equivalently two rows concatenated; this class uses
        ``state_box.min(dim=0)`` and ``state_box.max(dim=0)``.
    target_box : torch.Tensor
        Same convention as ``state_box`` but for targets.
    action_box : torch.Tensor
        Same convention as ``state_box`` but for actions.
    output_mask : torch.Tensor
        Boolean mask either of shape ``state_box.shape[1:]`` (full mask) or of shape
        ``(state_box.shape[-1],)`` (broadcasted along leading feature dims).
    solver_kwargs : Mapping[str, Any]
        Options forwarded to the ODE solver.
    vector_field(state, action) -> torch.Tensor
        Returns time derivative of the state for the current action.
    signal_gen_kwargs : Mapping[str, Any]
        Kwargs forwarded to ``signal_gen_cls``. Provide either procedural parameters
        (e.g., ``{"n_switches": 3, "n_bins": 8}``) or fixed signals
        (e.g., ``{"fixed_times": ..., "fixed_values": ...}``). The env will inject
        ``num_envs``, ``Tf``, ``target_box``, ``device``, and ``rng`` automatically.

    ----------
    Notes
    ----------
    • The env automatically initializes internal buffers on first :meth:`step` if you
      haven't called :meth:`reset` yet.

    • The default generator supports per-env or broadcasted fixed signals and can be
      swapped out for custom generators via ``signal_gen_cls``.

    • For performance, prefer running on CUDA tensors and keep ``dt`` moderate. If you
      need multi-step integration between actions, use a higher frequency controller or
      call :meth:`step` in a loop.

    ----------
    Example
    ----------
    Minimal subclass with a 1D state, 1D action, 1D target; procedural targets:

    >>> class MyEnv(BatchedVectorFieldEnv):
    ...     @property
    ...     def state_box(self):
    ...         # bounds for x (state) in [-2, 2]
    ...         lo = torch.tensor([-2.0]); hi = torch.tensor([ 2.0])
    ...         return torch.stack([lo, hi], dim=0)
    ...
    ...     @property
    ...     def action_box(self):
    ...         lo = torch.tensor([-1.0]); hi = torch.tensor([1.0])
    ...         return torch.stack([lo, hi], dim=0)
    ...
    ...     @property
    ...     def target_box(self):
    ...         lo = torch.tensor([-1.0]); hi = torch.tensor([1.0])
    ...         return torch.stack([lo, hi], dim=0)
    ...
    ...     @property
    ...     def output_mask(self):
    ...         return torch.tensor([True])  # track the single state dim
    ...
    ...     @property
    ...     def solver_kwargs(self):
    ...         return {"rtol": 1e-4, "atol": 1e-4}
    ...
    ...     def vector_field(self, state, action):
    ...         # simple stable first-order system: xdot = -x + u
    ...         return -state + action
    ...
    ...     @property
    ...     def signal_gen_kwargs(self):
    ...         # procedural: 2 switches (3 segments), 5 bins
    ...         return {"n_switches": 2, "n_bins": 5}
    ...
    ... # construct and roll
    >>> env = MyEnv(num_envs=32, dt=0.05, Tf=2.0, regularization=1e-3, reset_noise=0.1)
    >>> obs, info = env.reset()
    >>> for _ in range(40):
    ...     actions = torch.zeros(32, 1)               # zero control
    ...     obs, rew, term, trunc, info = env.step(actions)

    Example with a fixed, broadcasted target (same for all envs):

    >>> class MyFixedEnv(MyEnv):
    ...     @property
    ...     def signal_gen_kwargs(self):
    ...         times  = torch.tensor([0.0, 0.5, 2.0])          # absolute seconds
    ...         values = torch.tensor([[0.0], [0.8], [0.2]])    # (n_ticks, 1)
    ...         return {"fixed_times": times, "fixed_values": values}
    ...
    >>> env_fixed = MyFixedEnv(num_envs=16, dt=0.05, Tf=2.0, regularization=0.0, reset_noise=0.05)
    >>> env_fixed.reset()
    >>> a = torch.zeros(16, 1)
    >>> obs, rew, term, trunc, info = env_fixed.step(a)
    """
    metadata: dict[str, Any] = {"render_modes": ['human', 'rgb_array', 'episodes']}

    num_envs: int
    observation_space: gym.Space
    action_space:      gym.Space
    dt: float
    Tf: float
    _target_box:  torch.Tensor
    _action_box:  torch.Tensor
    _state_box:   torch.Tensor
    output_mask: torch.Tensor
    regularization: float
    reset_noise: float
    render_mode: str | None
    render_nmax: int
    device: Any
    solver_kwargs: Mapping[str, Any]
    _state:   torch.Tensor | None
    _time:    torch.Tensor | None
    _target:  torch.Tensor | None
    _rng:     torch.Generator
    render_cache: Mapping[str, torch.Tensor] | None
    render_queue: deque | None
    _render_fig:   Any | None
    _render_axes:  Any | None
    _nb_display_handle: Any | None
    _target_gen: TargetSignalGeneratorBase
    signal_gen_cls: Type[TargetSignalGeneratorBase]

    # ---- abstract configuration API ----
    @property
    @abstractmethod
    def state_box(self) -> torch.Tensor:
        """
        Elementwise lower and upper bounds for the environment's state.

        Shape:
            (2, *state_shape)

        The first row defines the lower bounds, and the second row defines the upper
        bounds for each state dimension. These values are used for:
            • Clamping initial conditions on reset
            • Termination checks when the state leaves bounds
            • Determining the shape of the state space (for rendering/buffers)

        Example:
            >>> lo = torch.tensor([-1.0, -1.0])
            >>> hi = torch.tensor([ 1.0,  1.0])
            >>> return torch.stack([lo, hi], dim=0)
        """
        ...

    @property
    @abstractmethod
    def target_box(self) -> torch.Tensor:
        """
        Elementwise lower and upper bounds for the target signal.

        Shape:
            (2, *target_shape)

        The bounds define the valid range of target values and the shape of
        the target signal. They are also used by procedural target generators
        to scale normalized random signals into real-world ranges.
        """
        ...

    @property
    @abstractmethod
    def init_state_box(self) -> torch.Tensor:
        """
        Elementwise lower and upper bounds for sampling initial states on reset.

        Shape:
            (2, *state_shape)
        """
        ...

    @property
    @abstractmethod
    def action_box(self) -> torch.Tensor:
        """
        Elementwise lower and upper bounds for the control action.

        Shape:
            (2, *action_shape)

        Defines the nominal range of actions applied to the system.
        These bounds are used to create the Gym action space and to
        determine the shape of the control input tensor.
        """
        ...

    @property
    @abstractmethod
    def output_mask(self) -> torch.Tensor:
        """
        Boolean tensor selecting which components of the state are considered
        'outputs' for reward computation and tracking.

        Accepted shapes:
            • Full mask:  state_box.shape[1:]
            • Last-dim mask:  (state_box.shape[-1],)  (broadcast along leading dims)

        The mask is applied simultaneously across all feature dimensions after
        the batch (env) dimension. Only masked components contribute to the
        tracking error in the reward.
        """
        ...

    @property
    @abstractmethod
    def solver_kwargs(self) -> Mapping[str, Any]:
        """
        Keyword arguments forwarded to the ODE solver (torchdiffeq.odeint).

        Common options:
            • method: str   - integration scheme ('dopri5', 'rk4', 'euler', etc.)
            • rtol:   float - relative tolerance
            • atol:   float - absolute tolerance
            • options: dict - backend-specific extra parameters

        Example:
            >>> return {"method": "dopri5", "rtol": 1e-4, "atol": 1e-6}
        """
        ...
    
    @abstractmethod
    def vector_field(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Defines the system dynamics as a time-invariant vector field f(x, u).

        Args:
            state:  Tensor (num_envs, *state_shape)
                Current state of each environment.
            action: Tensor (num_envs, *action_shape)
                Control input for each environment.

        Returns:
            Tensor (num_envs, *state_shape)
                The time derivative of the state (dx/dt).

        Example:
            >>> def vector_field(self, state, action):
            ...     x, = state.unbind(-1)
            ...     u, = action.unbind(-1)
            ...     return -x + u
        """
        ...

    @abstractmethod
    def signal_gen_kwargs(self) -> Mapping[str, Any]:
        """
        Returns configuration parameters for constructing the target signal generator.

        You can provide either procedural parameters or fixed signal data.

        Procedural mode (default):
            - n_switches : int ≥ 0   → number of switches (segments = n_switches + 1)
            - n_bins     : int > 0   → number of discrete bins for sampling target levels

        Fixed-signal mode (overrides procedural):
            - fixed_times  : Tensor of shape
                             (n_ticks,) or (1, n_ticks) for broadcast,
                             or (num_envs, n_ticks) for per-env times
            - fixed_values : Tensor of shape
                             (n_ticks, *target_shape) or (1, n_ticks, *target_shape)
                             or (num_envs, n_ticks, *target_shape)

        The environment automatically injects the following defaults if missing:
            • num_envs
            • Tf
            • target_box
            • device
            • rng  (torch.Generator)

        Example:
            >>> return {"n_switches": 3, "n_bins": 8}
            # or fixed-signal example:
            >>> times  = torch.tensor([0.0, 0.5, 1.0])
            >>> values = torch.tensor([[0.0], [1.0], [0.0]])
            >>> return {"fixed_times": times, "fixed_values": values}
        """
        ...

    def __init__(self,
                 num_envs:       int, 
                 dt:             float,
                 Tf:             float,
                 regularization: float,
                 reset_noise:    float,
                 precision_band: float=0.01,
                 precision_steepness: float=4.0,
                 device:         Any=None,
                 render_mode:    str | None=None,
                 render_nmax:    int=1000,
                 render_qmax:    int=20,
                 flatten_obs:    bool=True,
                 signal_gen_cls: Type[TargetSignalGeneratorBase] = DefaultTargetSignalGenerator):
        self.num_envs = num_envs
        self.dt = max(float(dt), 0.)
        self.Tf = max(float(Tf), 0.)
        self.regularization = max(float(regularization), 0.)
        self.reset_noise = max(float(reset_noise), 0.)
        self.precision_band = min(max(float(precision_band), 0.), 1.0)
        self.precision_steepness = max(float(precision_steepness), 1.0)
        self.flatten_obs = bool(flatten_obs)
        self.device = torch.device(device if device is not None else 'cpu')

        self._state  = None
        self._time   = None
        self._target = None

        try:
            # cache boxes on device
            self._state_box  = torch.stack([self.state_box.min(dim=0).values,
                                            self.state_box.max(dim=0).values], dim=0).to(device=self.device, dtype=torch.float32)
            self._target_box = torch.stack([self.target_box.min(dim=0).values,
                                            self.target_box.max(dim=0).values], dim=0).to(device=self.device, dtype=torch.float32)
            self._action_box = torch.stack([self.action_box.min(dim=0).values,
                                            self.action_box.max(dim=0).values], dim=0).to(device=self.device, dtype=torch.float32)
            self._init_state_box = torch.stack([self.init_state_box.min(dim=0).values,
                                               self.init_state_box.max(dim=0).values], dim=0).to(device=self.device, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"Cannot cache bounding boxes during base class initialization: {e}")
        
        # Spaces
        self.action_space = gym.spaces.Box(
            low=self._action_box[0].cpu().numpy().astype(np.float32), 
            high=self._action_box[1].cpu().numpy().astype(np.float32), 
            shape=self._action_box.shape[1:], dtype=np.float32
        )
        s_min = self._state_box[0]
        s_max = self._state_box[1]
        t_min = self._target_box[0]
        t_max = self._target_box[1]

        if self.flatten_obs:
            obs_low  = torch.cat([s_min.view(-1), t_min.view(-1)], dim=0).cpu().numpy().astype(np.float32)
            obs_high = torch.cat([s_max.view(-1), t_max.view(-1)], dim=0).cpu().numpy().astype(np.float32)
            self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape=(obs_low.shape[0],), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict({
                'state':  gym.spaces.Box(low=s_min.cpu().numpy().astype(np.float32), 
                                         high=s_max.cpu().numpy().astype(np.float32), 
                                         shape=self._state_box.shape[1:], dtype=np.float32),
                'target': gym.spaces.Box(low=t_min.cpu().numpy().astype(np.float32), 
                                         high=t_max.cpu().numpy().astype(np.float32), 
                                         shape=self._target_box.shape[1:], dtype=np.float32)
            })

        # Render setup
        self.render_cache = None
        self.render_queue = None
        self.render_mode  = render_mode
        self.render_nmax  = int(max(render_nmax, 1))
        self.render_qmax  = int(max(render_qmax, 1))
        self._render_fig  = None
        self._render_axes = None
        self._nb_display_handle = None  # for Jupyter inline updates
        self.signal_gen_cls = signal_gen_cls

        self.seed()

    def solver(self, func: Any, state0: torch.Tensor, t_seq: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        ODE solver wrapper. By default, this uses ``torchdiffeq.odeint``.

        Args:
            func:    Callable
                Function defining the vector field: ``func(t, y) -> dy/dt``.
            state0:  Tensor (num_envs, *state_shape)
                Initial state at time t_seq[0].
            t_seq:   Tensor (n_times,)
                Sequence of time points to integrate over.
            **kwargs:
                Extra keyword arguments forwarded to the solver.
        Returns:
            Tensor (n_times, num_envs, *state_shape)
                Integrated states at each time point in t_seq.
        """
        return odeint(func, state0, t_seq, **kwargs)

    # ---------- mask & reward ----------
    def _output_mask_flat(self) -> torch.Tensor:
        feat_shape = tuple(self._state_box.shape[1:])
        if not isinstance(self.output_mask, torch.Tensor):
            raise TypeError("output_mask must be a torch.Tensor")
        mask = self.output_mask.to(device=self.device, dtype=torch.bool)
        if mask.shape == feat_shape:
            mask_full = mask
        elif mask.ndim == 1 and feat_shape and mask.shape[0] == feat_shape[-1]:
            mask_full = mask.view(*([1] * (len(feat_shape) - 1)), -1).expand(feat_shape)
        else:
            raise ValueError(f"output_mask must have shape {feat_shape} or ({feat_shape[-1]},), got {tuple(mask.shape)}")
        return mask_full.reshape(-1)
    
    def remap_mobius_sigmoid(self, x: torch.Tensor, r: float, k: float = 8.0, clamp_input: bool = False) -> torch.Tensor:
        """
        Smooth, order-preserving remap with exact constraints:
          f(0)=0, f(r)=0.5, f(1)=1  (for any k>0)

        Construction:
          g(x) = sigmoid(k*(x - r))
          Choose a monotone Möbius h(g) = (a*g + b)/(c*g + 1)
          so that h(g(0))=0, h(0.5)=0.5, h(g(1))=1
          and define f(x)=h(g(x)).
        """
        if clamp_input:
            x = x.clamp(0.0, 1.0)

        r_t = torch.as_tensor(r, dtype=x.dtype, device=x.device)
        k_t = torch.as_tensor(k, dtype=x.dtype, device=x.device)

        g  = torch.sigmoid(k_t * (x - r_t))
        g0 = torch.sigmoid(-k_t * r_t)          # g(0)
        g1 = torch.sigmoid(k_t * (1.0 - r_t))   # g(1)

        # Solve for Möbius params
        a = ( -1.0 + 2.0 * g1 ) / ( g1 + g0 - 4.0 * g1 * g0 )
        b = -a * g0
        c = ( a * (g1 - g0) - 1.0 ) / g1

        denom = c * g + 1.0
        # preserve sign while avoiding division by ~0
        eps = torch.finfo(x.dtype if x.is_floating_point() else torch.float32).eps
        denom = torch.where(denom.abs() < eps, eps * denom.sign(), denom)

        f = (a * g + b) / denom
        return f.clamp(0.0, 1.0)

    def reward_fn(self, 
                  state:      torch.Tensor, 
                  target:     torch.Tensor, 
                  time:       torch.Tensor,
                  action:     torch.Tensor,
                  truncated:  torch.Tensor,
                  terminated: torch.Tensor):
        """
        Compute the reward signal based on the current state, target, and action.
        Override this method in subclasses to customize the reward function.

        Args:
            state:      Tensor (num_envs, *state_shape)
                Current state of each environment.
            target:     Tensor (num_envs, *target_shape)
                Current target signal for each environment.
            time:       Tensor (num_envs,)
                Current time for each environment.
            action:     Tensor (num_envs, *action_shape)
                Control input for each environment.
            truncated:  Tensor (num_envs,)
                Truncation flags for each environment.
            terminated: Tensor (num_envs,)
                Termination flags for each environment.
        Returns:
            Tensor (num_envs,)
                Reward for each environment.
        """
        mask_flat  = self._output_mask_flat()

        shift_y = self._state_box[0].view(-1)[mask_flat].view(1, -1)
        scale_y = (self._state_box[1] - self._state_box[0]).view(-1)[mask_flat].view(1, -1)

        y_flat   = state.view(self.num_envs, -1)[:, mask_flat]
        tgt_flat = target.view(self.num_envs, -1)

        y_flat_normalized = (y_flat - shift_y) / scale_y
        tgt_flat_normalized = (tgt_flat - shift_y) / scale_y

        err = (y_flat_normalized - tgt_flat_normalized).abs()
        err = self.remap_mobius_sigmoid(err, r=self.precision_band, k=self.precision_steepness, clamp_input=True)

        reward = -err.mean(dim=1)

        shift_u = self._action_box[0].view(1, -1)
        scale_u = (self._action_box[1] - self._action_box[0]).view(1, -1)
        act_flat = action.view(self.num_envs, -1)
        act_normalized = (act_flat - shift_u) / scale_u
        u_wash = F.smooth_l1_loss(act_normalized, 
                                  torch.zeros_like(act_flat),
                                  beta=0.1, reduction='none')  # normalized action
        r = self.regularization
        reg = -u_wash.mean(dim=1)
        return 1 + reward + r * reg

    def _build_obs(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor | Mapping[str, torch.Tensor]:
        if self.flatten_obs:
            return torch.cat([state.view(state.shape[0], -1), target.view(target.shape[0], -1)], dim=1)
        else:
            return {'state': state, 'target': target}

    # ---------- core API ----------
    def step(self, actions: Any):
        """
        Take a step in the environment by integrating the vector field over dt.

        Args:
            actions: Tensor (num_envs, *action_shape)
                Control inputs for each environment.
        Returns:
            observations: Tensor | Mapping[str, Tensor]
                Observations after the step.
            rewards: Tensor (num_envs,)
                Rewards obtained after the step.
            terminated: Tensor (num_envs,)
                Termination flags after the step.
            truncated: Tensor (num_envs,)
                Truncation flags after the step.
            infos: Mapping[str, Any]
                Info dict (possibly empty).
            
        Note:   
            - If the environment hasn't been reset yet, this method
              automatically calls :meth:`reset()` before proceeding.
            - Termination occurs if any state element is non-finite or
              if the state leaves the bounds defined by ``state_box``.
            - Truncation occurs when the time exceeds ``Tf``.
            - The info dict is currently empty but may be populated
                in future implementations or subclasses.
            - Rendering buffers are updated if rendering is enabled.
            - This method uses ODE solver defined by ``self.solver`` with
              options from ``self.solver_kwargs``.
              By default, this is ``torchdiffeq.odeint``. 
              Any custom solver must support the same signature.
        """
        if self._state is None or self._time is None or self._target is None:
            self.reset()

        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        state0, time0, target0 = self._state.clone(), self._time.clone(), self._target.clone()

        state1 = self.solver(
            lambda t, y: self.vector_field(y, actions),
            state0,
            torch.tensor([0.0, self.dt], device=self.device, dtype=state0.dtype),
            **self.solver_kwargs
        )[-1, ...]
        time1   = time0 + self.dt
        target1 = self._target_gen.get(time1)

        self._time, self._state, self._target = time1, state1, target1

        observations = self._build_obs(state1, target1)
        truncated  = (time1 >= self.Tf)
        terminated = ~state1.isfinite().view(self.num_envs, -1).all(dim=-1)
        terminated |= (state1 > self._state_box[1]).view(self.num_envs, -1).any(dim=-1)
        terminated |= (state1 < self._state_box[0]).view(self.num_envs, -1).any(dim=-1)
        rewards = self.reward_fn(state1, target1, time1, actions, truncated, terminated)

        infos = {}
        self._maybe_save_for_render(state0, target0, time0, actions, rewards, terminated, truncated, infos)
        return observations, rewards, terminated, truncated, infos

    def seed(self, seed: int | None=None):
        """Reseed the environment's RNG and the target signal generator."""
        if seed is None:
            seed = torch.seed()
        self._rng = torch.Generator(device=self.device)
        if seed is not None:
            self._rng.manual_seed(seed)
        self._seed = seed

        # ---- Target generator: instantiate from class + kwargs ----
        gen_kwargs = dict(self.signal_gen_kwargs or {})
        # Fill sensible defaults if caller didn't provide them
        gen_kwargs.setdefault('num_envs', self.num_envs)
        gen_kwargs.setdefault('Tf', self.Tf)
        gen_kwargs.setdefault('target_box', self._target_box)
        gen_kwargs.setdefault('device', self.device)
        gen_kwargs.setdefault('rng', self._rng)
        # Provide procedural fallbacks ONLY if user didn't supply fixed_* and didn't set these:
        if 'fixed_times' not in gen_kwargs and 'fixed_values' not in gen_kwargs:
            gen_kwargs.setdefault('n_switches', 0)
            gen_kwargs.setdefault('n_bins', 10)

        self._target_gen = self.signal_gen_cls(**gen_kwargs)

    def initial_state(self, 
                      initial_target: torch.Tensor, 
                      noise: float, 
                      indices: torch.Tensor, 
                      rng: torch.Generator) -> torch.Tensor:
        """
        Sample initial states for the specified environments at reset time.
        States are sampled as follows:
            1. Start at the center of ``init_state_box``.
            2. Apply uniform noise in [-max_noise, max_noise], where
               ``max_noise = (init_state_box[1] - init_state_box[0]).abs() / 2. * noise``.
            3. Clamp to ``state_box``.  
                
        Args:
            initial_target: Tensor (n_reset, *target_shape)
                Initial target values for the environments being reset.
            noise: float
                Amplitude of uniform noise applied to the initial state.
            indices: Tensor (n_reset,)
                Indices of environments being reset.
            rng: torch.Generator
                RNG for sampling.
        Returns:
            Tensor (n_reset, *state_shape)
                Initial states for the environments being reset.    
        """
        n_reset = indices.shape[0]      
        state0 = (self._init_state_box[0] + self._init_state_box[1]).unsqueeze(0) / 2.
        state0 = state0.expand(n_reset, *self._init_state_box.shape[1:])
        # apply reset noise on masked outputs (across all feature dims)

        max_noise = (self._init_state_box[1] - self._init_state_box[0]).abs() / 2. * noise

        rnd_vals = torch.rand((n_reset, *self._init_state_box.shape[1:]), 
                              device=self.device, generator=rng) * 2 - 1
        state0 = state0 + rnd_vals * max_noise
        # clamp
        state0.clamp_(min=self._state_box[0], max=self._state_box[1])
        return state0

    def reset(self, indices: Any | None=None):
        """
        Reset all or selected environments to initial state, time, and target.

        Args:
            indices: Tensor (n_reset,) | None
                Indices of environments to reset. If None, resets all envs.
        Returns:
            observations: Tensor | Mapping[str, Tensor]
                Initial observations after reset.
            infos: Mapping[str, Any]
                Empty info dict.
        
        Raises:
            ValueError: if called with empty indices.
        """
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
            self._time  = torch.zeros((self.num_envs,), device=self.device)
            self._state = torch.zeros((self.num_envs, *self._state_box.shape[1:]), device=self.device)
            self._target= torch.zeros((self.num_envs, *self._target_box.shape[1:]), device=self.device)
        else:
            indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)

        if indices.numel() == 0:
            raise ValueError("reset called with empty indices")

        n_reset = indices.shape[0]
        t_init = torch.zeros((n_reset,), device=self.device)
        target0 = self._target_gen.get(t_init, indices=indices)

        state0 = self.initial_state(initial_target=target0, 
                                    noise=self.reset_noise,
                                    indices=indices, 
                                    rng=self._rng)

         # Write into buffers
        self._state[indices, ...]  = state0
        self._time[indices]        = t_init
        self._target[indices, ...] = target0

        observations = self._build_obs(state0, target0)
        infos = {}
        self._maybe_reset_render(indices)
        return observations, infos

    # ---------- rendering ----------
    def _maybe_reset_render(self, indices: Any | None=None):
        if self.render_mode is None:
            return
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)

        if self.render_cache is None:
            def get_buffer_cpu(trail_shape: Sequence[int]) -> torch.Tensor:
                return torch.full(size=(self.num_envs, self.render_nmax, *trail_shape), fill_value=float('nan'))
            self.render_cache = {
                'states':    get_buffer_cpu(self._state_box.shape[1:]),
                'targets':   get_buffer_cpu(self._target_box.shape[1:]),
                'actions':   get_buffer_cpu(self._action_box.shape[1:]),
                'rewards':   get_buffer_cpu([]),
                'terminated':get_buffer_cpu([]),
                'truncated': get_buffer_cpu([]),
                'times':     get_buffer_cpu([]),
                'sizes':     torch.full(size=(self.num_envs,), fill_value=0, dtype=torch.long),
                'seen':      torch.full(size=(self.num_envs,), fill_value=0, dtype=torch.long),
                'order':     torch.arange(self.render_nmax, dtype=torch.long
                                          ).view(1, -1).repeat(self.num_envs, 1)
            }

        if self.render_queue is None:
            self.render_queue = deque()

        for key in self.render_cache.keys():
            if key == 'order':
                self.render_cache[key][indices, ...] = torch.arange(self.render_nmax, dtype=torch.long
                                                                    ).view(1, -1).repeat(indices.shape[0], 1)
            elif key == 'sizes':
                self.render_cache[key][indices] = 0
            elif key == 'seen':
                self.render_cache[key][indices] = 0
            else:
                self.render_cache[key][indices, ...] = float('nan')

    def _maybe_save_for_render(self, 
                               states:     torch.Tensor,
                               targets:    torch.Tensor,
                               times:      torch.Tensor,
                               actions:    torch.Tensor,
                               rewards:    torch.Tensor,
                               terminated: torch.Tensor,
                               truncated:  torch.Tensor,
                               infos:      Mapping[str, Any]):
        if self.render_mode is None:
            return

        to_cache = {
            'states': states,
            'targets': targets,
            'actions': actions,
            'rewards': rewards,
            'terminated': terminated.float(),
            'truncated': truncated.float(),
            'times': times
        }

        seen  = self.render_cache['seen']
        sizes = self.render_cache['sizes']

        not_full_mask = (sizes < self.render_nmax)
        full_mask     = ~not_full_mask
        not_full_idxs = torch.nonzero(not_full_mask, as_tuple=True)[0]
        full_idxs     = torch.nonzero(full_mask, as_tuple=True)[0]

         # --- Append to not-yet-full buffers
        for key, value in to_cache.items():
            self.render_cache[key][not_full_idxs,
                                   sizes[not_full_idxs],
                                   ...] = value[not_full_idxs, ...].cpu()
        sizes[not_full_idxs] += 1

        # --- Reservoir sampling for already-full buffers
        if full_idxs.numel() > 0:
            # k = 1-based count of items seen so far; after increment, this sample is the k-th
            k = seen[full_idxs] + 1  # tensor of ints (length = #full envs)

            # j ~ UniformInt([0, k-1]) per env
            j = torch.floor(torch.rand(k.shape, dtype=torch.float32) * k.to(torch.float32)).to(torch.long)

            # Decide which envs to replace: j < nmax
            replace_mask = (j < self.render_nmax)
            if replace_mask.any():
                # Map indices
                envs_to_rep = full_idxs[replace_mask]
                slots       = j[replace_mask]  # slot to replace per env

                # Write new sample into the chosen slot
                for key, value in to_cache.items():
                    self.render_cache[key][envs_to_rep, slots, ...] = value[envs_to_rep, ...].cpu()

                # Maintain 'order' so the replaced slot becomes the most recent
                order_full = self.render_cache['order'][envs_to_rep, :]           # (m, nmax)
                prev_order = order_full.gather(dim=-1, index=slots.view(-1, 1)).squeeze(-1)
                sub_mask = order_full > prev_order.view(-1, 1)
                sub_rows, sub_cols = torch.nonzero(sub_mask, as_tuple=True)
                self.render_cache['order'][envs_to_rep[sub_rows], sub_cols] -= 1
                self.render_cache['order'][envs_to_rep, slots] = self.render_nmax - 1

        # Always increment seen for full envs (we saw one more item)
        seen[full_idxs] += 1

        self._maybe_push_to_render_queue()

    def _maybe_push_to_render_queue(self):
        dones = (self.render_cache['terminated'] > 0.5) | (self.render_cache['truncated'] > 0.5)
        dones = dones.view(self.num_envs, -1).sum(dim=1) == 1
        dones_idxs = torch.nonzero(dones, as_tuple=True)[0]
        for idx in dones_idxs:
            cutoff = self.render_cache['sizes'][idx]
            order  = self.render_cache['order'][idx, :cutoff]
            def restore_order(tensor: torch.Tensor) -> torch.Tensor:
                content = tensor[idx, :cutoff, ...]
                res = torch.zeros_like(content)
                res[order, ...] = content
                return res

            render_data = {
                key: restore_order(self.render_cache[key]).numpy()
                for key in self.render_cache.keys() if key not in ['sizes', 'order', 'seen']
            }
            render_data.update({'env_index': int(idx.cpu().numpy())})
            self.render_queue.append(render_data)
            if len(self.render_queue) > self.render_qmax:
                self.render_queue.popleft()

    # ---------- rendering helpers ----------
    def _is_notebook_env(self) -> bool:
        try:
            from IPython import get_ipython  # type: ignore
            shell = get_ipython().__class__.__name__
            return shell in ("ZMQInteractiveShell", "Shell")
        except Exception:
            return False

    @staticmethod
    def _robust_limits(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> Tuple[float, float]:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return -1.0, 1.0
        lo_v, hi_v = np.percentile(x, [lo, hi])
        if not np.isfinite(lo_v) or not np.isfinite(hi_v) or lo_v == hi_v:
            lo_v, hi_v = np.min(x), np.max(x)
        if lo_v == hi_v:
            eps = 1.0 if lo_v == 0 else abs(lo_v) * 0.1
            lo_v, hi_v = lo_v - eps, hi_v + eps
        lower = lo_v - 0.2 * (hi_v - lo_v)
        upper = hi_v + 0.2 * (hi_v - lo_v)
        return float(lower), float(upper)

    def _flatten_traj(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr[:, None]
        T = arr.shape[0]
        F = int(np.prod(arr.shape[1:], dtype=int))
        return arr.reshape(T, F)

    def _ensure_figure(self, n_states: int, n_actions: int):
        from matplotlib.gridspec import GridSpec
        need_new = self._render_fig is None or self._render_axes is None
        if need_new:
            if self.render_mode == "human":
                import matplotlib.pyplot as plt
                # Use pyplot in human mode so a live window / inline figure appears
                self._render_fig = plt.figure(figsize=(10, 7), dpi=120, constrained_layout=True)
                gs = GridSpec(3, 1, figure=self._render_fig)
                ax_y = self._render_fig.add_subplot(gs[0, 0])
                ax_u = self._render_fig.add_subplot(gs[1, 0])
                ax_r = self._render_fig.add_subplot(gs[2, 0])
                self._render_axes = {"y": ax_y, "u": ax_u, "r": ax_r}
                # Non-blocking show outside notebooks
                if not self._is_notebook_env():
                    plt.show(block=False)
            else:
                # Avoid pyplot in non-human modes to prevent notebook auto-display
                from matplotlib.figure import Figure
                self._render_fig = Figure(figsize=(10, 7), dpi=120, constrained_layout=True)
                gs = self._render_fig.add_gridspec(3, 1)
                ax_y = self._render_fig.add_subplot(gs[0, 0])
                ax_u = self._render_fig.add_subplot(gs[1, 0])
                ax_r = self._render_fig.add_subplot(gs[2, 0])
                self._render_axes = {"y": ax_y, "u": ax_u, "r": ax_r}
        else:
            # Clear for reuse
            for ax in self._render_axes.values():
                ax.cla()

        ax_y, ax_u, ax_r = self._render_axes["y"], self._render_axes["u"], self._render_axes["r"]
        ax_y.set_title("Outputs vs Target"), ax_y.set_xlabel("t"), ax_y.set_ylabel("y, target")
        ax_u.set_title("Actions"),           ax_u.set_xlabel("t"), ax_u.set_ylabel("u")
        ax_r.set_title("Reward"),            ax_r.set_xlabel("t"), ax_r.set_ylabel("r")

    def _draw_frame(self, render_data: Mapping[str, np.ndarray]) -> np.ndarray:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        states   = render_data["states"]
        targets  = render_data["targets"]
        actions  = render_data["actions"]
        rewards  = render_data["rewards"].reshape(-1)
        times    = render_data["times"].reshape(-1)

        S  = self._flatten_traj(states)
        Tt = self._flatten_traj(targets)
        U  = self._flatten_traj(actions)
        R  = rewards

        try:
            mask_flat = self._output_mask_flat().detach().cpu().numpy().astype(bool)
            Y = S[:, mask_flat] if mask_flat.any() else S
        except Exception:
            warn("Could not apply output_mask during rendering; showing all states instead.")
            Y = S

        y_lo, y_hi = self._robust_limits(np.vstack([Y, Tt]))
        u_lo, u_hi = self._robust_limits(U)
        r_lo, r_hi = self._robust_limits(R)

        self._ensure_figure(n_states=Y.shape[1], n_actions=U.shape[1])
        ax_y, ax_u, ax_r = self._render_axes["y"], self._render_axes["u"], self._render_axes["r"]

        for i in range(Y.shape[1]):
            ax_y.plot(times, Y[:, i], linewidth=1.2, alpha=0.9)
        for j in range(Tt.shape[1]):
            ax_y.plot(times, Tt[:, j], linestyle="--", linewidth=1.2, alpha=0.9)
        ax_y.set_ylim(y_lo, y_hi)
        ax_y.grid(True, linewidth=0.3, alpha=0.4)
        ax_y.legend([*(f"y[{i}]" for i in range(Y.shape[1])),
                     *(f"target[{j}]" for j in range(Tt.shape[1]))],
                    ncols=2, fontsize=8, loc="upper right", frameon=False)

        for i in range(U.shape[1]):
            ax_u.plot(times, U[:, i], linewidth=1.2, alpha=0.9)
        ax_u.set_ylim(u_lo, u_hi) 
        ax_u.grid(True, linewidth=0.3, alpha=0.4)
        ax_u.legend([f"u[{i}]" for i in range(U.shape[1])],
                    ncols=min(U.shape[1], 3), fontsize=8, loc="upper right", frameon=False)

        ax_r.plot(times, R, linewidth=1.2, alpha=0.95)
        ax_r.set_ylim(r_lo, r_hi) 
        ax_r.grid(True, linewidth=0.3, alpha=0.4)

        canvas = FigureCanvasAgg(self._render_fig)
        canvas.draw()
        w, h = self._render_fig.get_size_inches() * self._render_fig.get_dpi()
        w = int(w)
        h = int(h)
        if hasattr(canvas, "buffer_rgba"):
            buf = canvas.buffer_rgba()  # RGBA
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            rgb = arr[..., :3].copy()   # drop alpha
        else:
            buf = canvas.tostring_rgb()  # RGB
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        return rgb

    def render(self, indices: Iterable[int] = None) -> np.ndarray | None | Mapping[str, np.ndarray]:
        """
        Render the environment's trajectory data. It renders only fully recorded episodes. 
        If there are multiple completed episodes in the render queue, it retrieves the first one
        matching the specified indices (or the first available if no indices are given).
        If there is no full episode available, it returns None.
        
        Args:
            indices: Iterable[int] | None
                Indices of environments to render. If None, renders the first available.
        Returns:
            - If render_mode is 'human': None (renders to screen or notebook)
            - If render_mode is 'rgb_array': ndarray (H, W, 3)
                RGB image of the rendered frame.
            - If render_mode is 'episodes': Mapping[str, ndarray]
                Raw trajectory data for the selected environment.
                Keys: 'states', 'targets', 'actions', 'rewards', 'terminated', 'truncated', 'times', 'env_index'.
        Raises:
            ValueError: if render_mode is not set or invalid.
        Note:
            - In 'human' mode, if running in a Jupyter notebook, the figure
              is updated in-place within the output cell.
            - In 'human' mode outside notebooks, a non-blocking window is used.
        """
        if self.render_mode is None:
            raise ValueError("Render mode not specified. Set render_mode to 'human', 'rgb_array' or 'episodes'.")
        if self.render_queue is None:
            return None
        if indices is not None:
            indices = [int(i) for i in indices]

        render_data = None
        while len(self.render_queue) > 0:
            candidate = self.render_queue.popleft()
            if indices is None or candidate['env_index'] in indices:
                render_data = candidate
                break
        if render_data is None:
            return None

        if self.render_mode == "episodes":
            return render_data

        frame = self._draw_frame(render_data)
        if self.render_mode == "rgb_array":
            return frame

        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            if self._is_notebook_env():
                # In notebooks: show once, then update the same output cell.
                from IPython.display import display # type: ignore
        
                # Draw the latest content to the canvas so the display gets a fresh image.
                if getattr(self._render_fig, "canvas", None) is not None:
                    # draw_idle schedules a draw, but in inline it's safer to force it:
                    self._render_fig.canvas.draw()
        
                if self._nb_display_handle is None:
                    # First time: create a live display with an id we can update
                    self._nb_display_handle = display(self._render_fig, display_id=True)
                else:
                    # Subsequent frames: update in-place (no new figures spawned)
                    try:
                        self._nb_display_handle.update(self._render_fig)
                    except Exception:
                        # If the handle died (e.g., cell re-run), recreate it
                        self._nb_display_handle = display(self._render_fig, display_id=True)
                return None
            else:
                # Outside notebooks: pump the GUI event loop
                self._render_fig.canvas.draw()
                self._render_fig.canvas.flush_events()
                plt.pause(0.001)
                return None

        raise ValueError(f"Unknown render_mode='{self.render_mode}'. Use 'human', 'rgb_array' or 'episodes'.")
    
    def render_clear(self):
        """Clear the render queue."""
        if self.render_queue is not None:
            self.render_queue.clear()
