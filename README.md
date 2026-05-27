# Differential Equations for Generative Models

Companion notebooks for the paper *[title]*.  
Each notebook is self-contained and targets a 1-D toy problem so the full training loop runs in minutes on CPU.

## Notebooks

| Notebook | Topic | Key concept |
|---|---|---|
| [`01_ode.ipynb`](notebooks/01_ode.ipynb) | ODE generative model | Exact log-likelihood via instantaneous change of variables |
| [`02_sde.ipynb`](notebooks/02_sde.ipynb) | SDE generative model | Variational ELBO; drift, diffusion and variational networks trained jointly |
| [`03_flow_matching.ipynb`](notebooks/03_flow_matching.ipynb) | Flow matching | Stochastic interpolant; closed-form target drift; full ELBO monitoring |
| [`04_vdm.ipynb`](notebooks/04_vdm.ipynb) | Variational Diffusion Model | SDE specialised to a VP schedule; signal prediction network; learnable SNR endpoints |

The four notebooks form a progression:
- The **ODE** notebook shows exact maximum-likelihood training of a continuous normalizing flow.
- The **SDE** notebook generalises this to a stochastic forward process and derives a tractable ELBO by introducing a Gaussian variational path $q(x_t \mid y) = \mathcal{N}(x_t \mid \alpha(y,t), \beta(y,t)^2)$ with learned $\alpha$, $\beta$, drift $f$, and diffusion $\sigma$.
- The **flow matching** notebook shows that fixing $\alpha(y,t) = ty$ and $\beta(y,t) = \beta_t$ (scalar, optionally learnable) collapses the SDE ELBO to a simple MSE objective with a closed-form target, connecting the variational framework to standard flow matching.
- The **VDM** notebook specialises the SDE notebook to a variance-preserving schedule: $\alpha(y,t) = a_t y$ and $\beta(y,t) = b_t$ are fixed by a linear log-SNR schedule with learnable endpoints, and the drift network is replaced by a signal prediction network $\widehat{y}(x_t,t)$. The drift-matching term reduces to an SNR-weighted prediction loss.

## Running the notebooks

### Google Colab (recommended for a quick start)

Open any notebook directly in Colab. The first cell automatically downloads `utils.py` from GitHub:

```
!wget -q https://raw.githubusercontent.com/olewinther/generative-ode-sde/main/utils.py
```

### Local

```bash
git clone https://github.com/olewinther/generative-ode-sde
cd generative-ode-sde
pip install -r requirements.txt
jupyter notebook notebooks/
```

`utils.py` is found automatically via `sys.path` when Jupyter is started from the repo root or the `notebooks/` directory.

## Repository layout

```
generative-ode-sde/
├── utils.py           # Shared code: data, priors, forward/backward paths, visualisation
├── notebooks/
│   ├── 01_ode.ipynb
│   ├── 02_sde.ipynb
│   ├── 03_flow_matching.ipynb
│   └── 04_vdm.ipynb
└── requirements.txt
```

## Requirements

```
torch >= 2.0
numpy
matplotlib
scipy
torchsummary
```

Install with `pip install -r requirements.txt`.

## Citation

```bibtex
@article{todo,
  title   = {[title]},
  author  = {[authors]},
  year    = {2025},
}
```
