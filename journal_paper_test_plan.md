## Journal Paper TODO

### Title

Physics‑Informed Bidirectional Mamba Networks with Stochastic Taylor Derivative Estimators for PDE Approximation

## Test Plan

### Recommended PDE Benchmarks

Below are six benchmark problems to test the MAMBA SSM + forward‑mode AD + STDE architecture. For each, we specify the domain, conditions, and metrics.

#### 1. 1D Burgers’ Equation

**PDE**

```
u_t + u·u_x − ν·u_{xx} = 0,  x ∈ [−1,1],  t ∈ [0,1],  ν = 0.01/π  
```

**IC / BC**

* Initial: `u(x,0) = −sin(π x)`
* Dirichlet: `u(±1,t) = 0`

**Metrics**

* Relative L₂ error at t=1
* Max absolute error (L∞) over grid
* Mean PDE residual: (1/N) Σ |N\[u\_pred]|
* Epochs to reach L₂ < 1e‑3
* Total training time & time per epoch

#### 2. 2D Poisson Equation

**PDE**

```
−Δu(x,y) = f(x,y),  (x,y) ∈ [0,1]²  
u_exact(x,y) = sin(πx)·sin(πy)  
```

**BC**

* Dirichlet `u = 0` on ∂\[0,1]²

**Metrics**

* Relative L₂ error on uniform test grid
* Max absolute error (L∞)
* Mean residual over collocation points
* GPU memory footprint
* Time to target L₂ error

#### 3. 1D Allen–Cahn Equation

**PDE**

```
u_t − 0.0001·u_{xx} + 5u³ − 5u = 0,  x ∈ [−1,1],  t ∈ [0,1]  
```

**IC / BC**

* Initial: `u(x,0) = x² cos(πx)`
* Dirichlet: `u(±1,t) = 0`

**Metrics**

* Relative L₂ and L∞ errors at t=1
* Energy functional error:
  ∫ (¼ u⁴ − ½ u² + (ε²/2) u\_x²) dx
* Residual norm vs. epoch

#### 4. 1D Nonlinear Schrödinger Equation

**PDE**

```
i ψ_t + ½ ψ_{xx} + |ψ|² ψ = 0,  x ∈ [−5,5],  t ∈ [0, π/2]  
```

**IC / BC**

* Initial: `ψ(x,0) = sech(x)`
* Periodic boundaries

**Metrics**

* Amplitude RMSE: ‖|ψ\_pred| − |ψ\_exact|‖₂
* Phase error: ‖arg(ψ\_pred) − arg(ψ\_exact)‖₂
* Mass & energy conservation over time
* Residual magnitude

#### 5. Korteweg–de Vries (KdV) Equation

**PDE**

```
u_t + 6 u·u_x + u_{xxx} = 0,  x ∈ [−5,5],  t ∈ [0,1]  
```

**IC / BC**

* Initial: `u(x,0) = sech²(x)`
* Periodic boundaries

**Metrics**

* Relative L₂ and L∞ errors at final time
* Soliton speed error (crest location difference)
* PDE residual

#### 6. 2D Navier–Stokes (Optional Extension)

*(If added, include velocity-pressure fields, Reynolds number, divergence-free constraint, and conservation-of-mass/energy metrics.)*

---

### Ablation Study Strategy

Define variants to isolate each component and run on all benchmarks:

| Variant | SSM Backbone | Forward AD | STDE | Description                      |
| :-----: | :----------: | :--------: | :--: | :------------------------------- |
|    A1   |   No (MLP)   |     No     |  No  | Standard PINN baseline           |
|    A2   |      Yes     |     No     |  No  | SSM backbone only                |
|    A3   |      No      |     Yes    |  No  | Forward AD without stochasticity |
|    A4   |      No      |     Yes    |  Yes | STDE estimator only              |
|    A5   |      Yes     |     Yes    |  No  | Backbone + forward AD            |
|    A6   |      Yes     |     Yes    |  Yes | Full model                       |

**Procedure**

1. Fixed protocol: same optimizer, epochs, learning‑rate schedule, sampling.
2. Run each variant with 5 random seeds; report mean ± std.
3. Hyperparameter sweep (depth, width, STDE samples).
4. Paired t‑tests for significance.

---

### Performance Metrics (per test)

1. **Solution Accuracy**

   * Relative L₂ error: ‖u\_pred − u\_exact‖₂ / ‖u\_exact‖₂
   * Max absolute error (L∞)
   * Mean & max PDE residual

2. **Computational Efficiency**

   * Time to convergence (sec / epochs)
   * Time per epoch
   * Peak GPU memory usage

3. **Stability & Robustness**

   * Std of errors across seeds
   * Sensitivity: 1% Gaussian noise on ICs

4. **Physics‑Specific**

   * Conservation (mass, energy) for Schrödinger & Navier–Stokes
   * Wave‑speed error for KdV
   * Bifurcation accuracy for Allen–Cahn (if parameterized)
