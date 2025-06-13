import jax
import numpy as np
import matplotlib.pyplot as plt

from stde.config import Config
import stde.equations as eqns


EQN_NAMES = [
    "HJB_LIN",
    "HJB_LQG",
    "BSB",
    "Wave",
    "Poisson",
    "PoissonHouman",
    "PoissonTwobody",
    "PoissonTwobodyG",
    "PoissonThreebody",
    "AllenCahnTwobody",
    "AllenCahnTwobodyG",
    "AllenCahnThreebody",
    "SineGordonTwobody",
    "SineGordonTwobodyG",
    "SineGordonThreebody",
    "AllenCahnTime",
    "SineGordonTime",
    "SemilinearHeatTime",
    "KdV2d",
    "highord1d",
]


def _title_for_eqn(name: str, cfg: Config) -> str:
    params = []
    if name == "HJB_LQG":
        params.append(f"mu={cfg.eqn_cfg.mu}")
    if name == "HJB_LIN":
        params.append(f"c={cfg.eqn_cfg.c}")
    if name == "BSB":
        params.append(f"sigma={cfg.eqn_cfg.sigma}")
        params.append(f"r={cfg.eqn_cfg.r}")
    if hasattr(cfg.eqn_cfg, "max_radius"):
        params.append(f"max_r={cfg.eqn_cfg.max_radius}")
    parts = ", ".join(params)
    bc = eqns.__getattribute__(name).boundary_cond.__name__
    return f"{name} ({parts})\nBC: {bc}"


def visualize_PDEs(n_interior: int = 500, n_boundary: int = 100, seed: int = 0):
    cfg = Config()
    rng = jax.random.PRNGKey(seed)
    for name in EQN_NAMES:
        cfg.eqn_cfg.name = name
        eqn = getattr(eqns, name)
        if eqn.is_traj or eqn.time_dependent:
            cfg.eqn_cfg.dim = 1
        else:
            cfg.eqn_cfg.dim = 2
        sample_domain_fn = eqn.get_sample_domain_fn(cfg.eqn_cfg)
        if eqn.is_traj:
            x, t, xb, tb, rng = sample_domain_fn(n_interior, cfg.eqn_cfg.n_t, rng)
            x = x.reshape((-1, cfg.eqn_cfg.dim))
            t = t.reshape((-1, 1))
        else:
            x, t, xb, tb, rng = sample_domain_fn(n_interior, n_boundary, rng)
        # try exact solution first, fallback to boundary condition
        try:
            vals = np.array(eqn.sol(x, t, cfg.eqn_cfg))
            vals_b = np.array(eqn.boundary_cond(xb, tb, cfg.eqn_cfg))
        except Exception:
            vals = np.array(eqn.boundary_cond(x, t, cfg.eqn_cfg))
            vals_b = np.array(eqn.boundary_cond(xb, tb, cfg.eqn_cfg))
        if eqn.time_dependent or eqn.is_traj:
            xi, yi = np.array(x[:, 0]), np.array(t[:, 0])
            xbi, ybi = np.array(xb[:, 0]), np.array(tb[:, 0])
            xlabel, ylabel = "x", "t"
        else:
            xi, yi = np.array(x[:, 0]), np.array(x[:, 1])
            xbi, ybi = np.array(xb[:, 0]), np.array(xb[:, 1])
            xlabel, ylabel = "dim0", "dim1"
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={"aspect": "equal"})
        sc = ax.scatter(xi, yi, c=vals, cmap="viridis", s=20, label="interior", marker="o")
        ax.scatter(xbi, ybi, c=vals_b, cmap="viridis", s=40, label="boundary", marker="^")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02, label="PDE value")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_title(_title_for_eqn(name, cfg))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    visualize_PDEs()
