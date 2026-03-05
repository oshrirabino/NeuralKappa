# NeuralKappa

NeuralKappa is a Python package for rate-independent spike-train irregularity analysis.
It provides vectorized computation of $S_I$, fast $S_I$ -> $\kappa$ conversion via LUT interpolation,
and baseline metrics/utilities for neural data workflows.

## Install

From GitHub:

```bash
pip install git+https://github.com/oshrirabino/NeuralKappa.git
```

From local clone:

```bash
pip install .
```

## Public API

- `neuralkappa.metrics`
  - `compute_si(isis) -> float`  
    Computes local spike-train irregularity \(S_I\) from consecutive ISIs (`isis` is a 1D positive interval array).
  - `si_to_kappa(si_array) -> np.ndarray`  
    Converts one or many SI values to \(\kappa\) using the precomputed interpolation LUT.
  - `compute_cv(isis) -> float`  
    Returns coefficient of variation (`std / mean`) for ISIs as a baseline variability measure.
  - `compute_fano(isis, window_ms=500.0) -> float`  
    Reconstructs spike times from ISIs, bins counts per window, and returns variance/mean of counts.
- `neuralkappa.preprocessing`
  - `timestamps_to_isis(timestamps) -> np.ndarray`  
    Converts strictly increasing spike timestamps into ISIs via first differences.
  - `validate_isis(isis, min_length=2) -> np.ndarray`  
    Validates ISI shape/positivity/finiteness and returns normalized `float64` 1D array.
- `neuralkappa.lab_tools`
  - `compute_trial_averaged_kappa(list_of_isi_arrays) -> float`  
    Computes SI per trial, averages SI across trials, then maps that average to \(\kappa\).
  - `compute_kappa_from_timestamps(timestamps) -> float`
    Convenience wrapper: `timestamps -> ISIs -> SI -> kappa`.
  - `compare_kappa_conditions(trials_a, trials_b, n_perm=5000, random_state=None) -> dict`
    Compares two conditions from trial-level ISI arrays using a permutation test on mean kappa.
- `neuralkappa.simulation`
  - `generate_rate_modulated_gamma(base_kappa, rate_profile, random_state=None) -> np.ndarray`  
    Generates synthetic ISIs from a rate-modulated gamma process for validation and benchmarking.

## Minimal Usage

```python
import numpy as np
import neuralkappa.metrics as metrics
import neuralkappa.preprocessing as preprocessing

timestamps_ms = np.array([10.0, 22.0, 35.0, 51.0, 70.0])
isis = preprocessing.timestamps_to_isis(timestamps_ms)

si = metrics.compute_si(isis)
kappa = metrics.si_to_kappa([si])[0]
cv = metrics.compute_cv(isis)
fano = metrics.compute_fano(isis, window_ms=500.0)
```

Time convention: ISIs and `window_ms` are in milliseconds.

## LUT Configuration

Set these environment variables before Python starts:

- `NEURALKAPPA_KAPPA_MIN` (default: `0.01`)
- `NEURALKAPPA_KAPPA_MAX` (default: `1000.0`)
- `NEURALKAPPA_LUT_RESOLUTION` (default: `200000`)

Example:

```bash
export NEURALKAPPA_KAPPA_MIN=0.005
export NEURALKAPPA_KAPPA_MAX=2000
export NEURALKAPPA_LUT_RESOLUTION=300000
```

## Input/Output Notes

### What ISI should look like

`ISI` means *Interspike Interval*: the time gap between two consecutive spikes.

- Expected format: 1D numeric array-like (`list`, `tuple`, or `numpy.ndarray`)
- Values must be finite and strictly positive (`> 0`)
- Units in this package are milliseconds
- Typical example: `np.array([8.2, 11.5, 9.7, 10.1])`

If you start from timestamps, convert first with:

- `timestamps_to_isis(timestamps) -> np.ndarray`

where `timestamps` must be strictly increasing and in the same time unit (ms).

### Lab tools: inputs and outputs

- `compute_trial_averaged_kappa(list_of_isi_arrays) -> float`
  - Input: sequence of trials, where each trial is one ISI array for the same neuron/condition.
  - Output: one scalar `kappa` computed by averaging trial SI values, then converting SI -> kappa.
  - Use this for a single summary irregularity value per neuron across repeated trials.

- `compute_kappa_from_timestamps(timestamps) -> float`
  - Input: one strictly increasing 1D timestamps array.
  - Output: one scalar kappa.
  - Use this when raw timestamps are available and you want a single-step conversion.

- `compare_kappa_conditions(trials_a, trials_b, n_perm=5000, random_state=None) -> dict`
  - Input:
    - `trials_a`: sequence of ISI arrays for condition A.
    - `trials_b`: sequence of ISI arrays for condition B.
  - Output: dictionary with condition means, mean-difference effect size, permutation p-value, and trial counts.
  - Use this for A/B condition inference on kappa using trial-level ISI data.

#### @ Auther: Oshrira
#### @ Using codex agent
