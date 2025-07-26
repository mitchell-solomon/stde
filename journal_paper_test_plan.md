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

#### 2. 2D Poisson Equation

**PDE**

```
−Δu(x,y) = f(x,y),  (x,y) ∈ [0,1]²  
u_exact(x,y) = sin(πx)·sin(πy)  
```

**BC**

* Dirichlet `u = 0` on ∂\[0,1]²

#### 3. 1D Allen–Cahn Equation

**PDE**

```
u_t − 0.0001·u_{xx} + 5u³ − 5u = 0,  x ∈ [−1,1],  t ∈ [0,1]  
```

**IC / BC**

* Initial: `u(x,0) = x² cos(πx)`
* Dirichlet: `u(±1,t) = 0`


#### 4. 1D Nonlinear Schrödinger Equation

**PDE**

```
i ψ_t + ½ ψ_{xx} + |ψ|² ψ = 0,  x ∈ [−5,5],  t ∈ [0, π/2]  
```

**IC / BC**

* Initial: `ψ(x,0) = sech(x)`
* Periodic boundaries

#### 5. Korteweg–de Vries (KdV) Equation

**PDE**

```
u_t + 6 u·u_x + u_{xxx} = 0,  x ∈ [−5,5],  t ∈ [0,1]  
```

**IC / BC**

* Initial: `u(x,0) = sech²(x)`
* Periodic boundaries


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

### Backbone Hyperparameter Sweep

Run a small grid for each backbone on every PDE benchmark:

* **MLP** – vary `depth`, `width`, `block_size` (when weight sharing is enabled)
  and the activation function.
* **MAMBA** – vary `num_mamba_blocks`, `hidden_features`, `expansion_factor`,
  `dt_rank`, the activation function, and whether the blocks are bidirectional.

Use the same training protocol as above for all combinations and report the best
validation error per equation.

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

