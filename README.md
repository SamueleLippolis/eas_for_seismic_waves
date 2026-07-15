# General description
Trials of various evolutionary algorithms such as es, cmaes, and pso to solve an optimization problem derived from an inverse problem. The topic concerns a geophyscial analysis.

# EAs for Seismic Waves

This project studies an inverse geophysical problem where hidden sediment parameters are recovered from synthetic seismic/electrical observations using evolutionary optimization and symbolic-regression surrogate models.

## Giulio-style evolutionary optimization

Giulio's approach solves the inverse problem by minimizing the mismatch between observed quantities `(Vp, Vs, sigma)` and the forward-model prediction using SA, PSO, or CMA-ES.

## Available optimizers

```text
simulated_annealing_giulio
pso_giulio
cmaes_giulio
```

## Pipeline
- set config/giulio_part1.yaml
    - set optimizer kind 
    - set otpimizer features in optimizers 
- run scripts/run_giulio_part1.py
    - results in reports/giulio_part1/<optimizer_name>/<run_name>_<timestamp>/


# Symbolic regression 
The symbolic-regression approach first learns an explicit surrogate of the forward model, then uses this surrogate to guide inverse optimization.

The learned surrogate has the form:
- x = (phi_percent, C_percent, S_b_percent, sigma_b_inv, xi)
- Vp_hat(x)
- Vs_hat(x)
- sqrt_sigma_hat(x)
- sigma_hat(x) = sqrt_sigma_hat(x)^2

## Pipeline 
- set config/datasets.yaml
    - in dataset, name and number of elements 
- run scripts/build_dataset.py
- set config/symbolic_regression.yaml
    - specify target
- run scripts/train_symbolic_forward.py
- set config/symbolic_surrogate.yaml
    - in models, set the model path for each target
- run scripts/check_symbolic_surrogate.py
- set config/symbolic_inverse.yaml
    - set the otpimizer 
- run scripts/run_symbolic_inverse.py