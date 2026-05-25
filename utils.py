import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TrainingSetWithLogLikelihood:
    def __init__(self, distribution, params):
        self.distribution = distribution
        self.params = params

    def generate_training_data(self, n_samples=5000):
        if self.distribution == 'laplace_mixture':
            k = self.params['k']
            spacing = self.params['spacing']
            scale = self.params['scale']
            means = np.arange(-spacing * (k - 1) / 2, spacing * ((k - 1) / 2 + 1), spacing)
            data = self._generate_laplace_mixture(n_samples, means, scale)
            expected_log_likelihood = self._laplace_mixture_log_likelihood(data, means, scale)
        elif self.distribution == 'laplace':
            loc, scale = self.params['loc'], self.params['scale']
            data = torch.tensor(np.random.laplace(loc=loc, scale=scale, size=(n_samples, 1)), dtype=torch.float32)
            expected_log_likelihood = self._laplace_log_likelihood(data, loc, scale)
        elif self.distribution == 'gaussian':
            mean, std = self.params['mean'], self.params['std']
            data = torch.tensor(np.random.normal(loc=mean, scale=std, size=(n_samples, 1)), dtype=torch.float32)
            expected_log_likelihood = self._gaussian_log_likelihood(data, mean, std)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        return data, expected_log_likelihood

    def _generate_laplace_mixture(self, n_samples, means, scale):
        n_components = len(means)
        proportions = np.ones(n_components) / n_components
        data = [np.random.laplace(loc=np.random.choice(means, p=proportions), scale=scale)
                for _ in range(n_samples)]
        return torch.tensor(data, dtype=torch.float32).unsqueeze(1)

    def _laplace_mixture_log_likelihood(self, data, means, scale):
        scale_t = torch.as_tensor(scale, dtype=torch.float32)
        log_likelihoods = torch.stack(
            [-torch.abs(data - mean) / scale_t - torch.log(2 * scale_t) for mean in means], dim=1
        )
        return (torch.logsumexp(log_likelihoods, dim=1) - np.log(len(means))).mean().item()

    def _laplace_log_likelihood(self, data, loc, scale):
        scale_t = torch.as_tensor(scale, dtype=torch.float32)
        return (-torch.abs(data - loc) / scale_t - torch.log(2 * scale_t)).mean().item()

    def _gaussian_log_likelihood(self, data, mean, std):
        return (-0.5 * ((data - mean) ** 2 / std ** 2) - 0.5 * torch.log(2 * torch.pi * std ** 2)).mean().item()


class Prior:
    def __init__(self, sample_func, log_prob_func):
        self.sample_func = sample_func
        self.log_prob_func = log_prob_func

    def sample(self, n_samples):
        return self.sample_func(n_samples)

    def log_prob(self, x):
        return self.log_prob_func(x)


def gaussian_sample(n_samples):
    return torch.tensor(np.random.normal(loc=0, scale=1, size=(n_samples, 1)), dtype=torch.float32)


def laplace_sample(n_samples):
    return torch.tensor(np.random.laplace(loc=0, scale=1, size=(n_samples, 1)), dtype=torch.float32)


def gaussian_log_pdf(x):
    return -0.5 * torch.log(torch.tensor(2 * np.pi)) - 0.5 * x ** 2


def laplace_log_pdf(x):
    return -torch.abs(x) - torch.log(torch.tensor(2.0))


class ForwardPath:
    def __init__(self, mode="ode", f_net=None, sigma_net=None, prior=None, likelihood_func=None):
        assert mode in ["ode", "sde"], "Invalid mode selected."
        self.mode = mode
        self.f_net = f_net
        self.sigma_net = sigma_net
        self.prior = prior
        self.likelihood_func = likelihood_func

    def sample_prior(self, n_samples):
        return self.prior.sample(n_samples)

    def integrate(self, x0, t_grid):
        x_t = x0
        paths = [x0]
        dt = t_grid[1] - t_grid[0]
        for t in t_grid[:-1]:
            t = t.expand_as(x_t)
            if self.mode == "ode":
                x_t = x_t + self.f_net(x_t, t) * dt
            elif self.mode == "sde":
                drift = self.f_net(x_t, t)
                sigma = self.sigma_net(x_t, t)
                noise = torch.randn_like(x_t) * torch.sqrt(dt)
                x_t = x_t + drift * dt + sigma * noise
            paths.append(x_t)
        return torch.stack(paths)

    def sample_data(self, x_1):
        if self.likelihood_func is not None:
            return self.likelihood_func.sample(x_1)
        return x_1

    def sample(self, n_samples, t_grid):
        x0 = self.sample_prior(n_samples)
        path = self.integrate(x0, t_grid)
        y = self.sample_data(path[-1])
        return y, path, x0


class BackwardPath:
    def __init__(self, mode="ode", f_net=None, alpha_net=None, beta_net=None, sigma_net=None, likelihood_func=None):
        assert mode in ["f_net", "ode", "backward sde"], "Invalid mode selected."
        self.mode = mode
        self.f_net = f_net
        self.alpha_net = alpha_net
        self.beta_net = beta_net
        self.sigma_net = sigma_net
        self.likelihood_func = likelihood_func

    def sample_starting_point(self, y):
        if self.likelihood_func is None:
            return y
        q_x1_given_y = Variational(self.alpha_net, self.beta_net, y, torch.full(y.size(), 1.0))
        return q_x1_given_y.sample()

    def integrate(self, x1, t_grid, y=None):
        x_t = x1
        paths = [x1]
        dt = -(t_grid[1] - t_grid[0])
        for t in reversed(t_grid[1:]):
            t = t.expand_as(x_t)
            if self.mode == "ode":
                drift = self.f_net(x_t, t)
            else:
                t.requires_grad = True
                q_xt_given_y = Variational(self.alpha_net, self.beta_net, y, t)
                drift, _ = q_xt_given_y.drift(x_t, mode=self.mode, sigma_net=self.sigma_net)
            x_t = x_t + drift * dt
            if self.mode == "backward sde":
                dW = torch.randn_like(x_t) * torch.sqrt(-dt)
                x_t = x_t + self.sigma_net(x_t, t) * dW
            paths.append(x_t)
        return torch.stack(paths[::-1])

    def sample(self, y, t_grid):
        x1 = self.sample_starting_point(y)
        path = self.integrate(x1, t_grid, y=y)
        return path[0], path, x1


class Variational(nn.Module):
    def __init__(self, alpha_net, beta_net, y, t):
        super().__init__()
        self.alpha_net = alpha_net
        self.beta_net = beta_net
        self.y = y
        self.t = t
        self._forward_result = None

    def _compute_forward(self):
        if self._forward_result is None:
            self._forward_result = (self.alpha_net(self.y, self.t), self.beta_net(self.y, self.t))
        return self._forward_result

    def forward(self, y, t):
        return self.alpha_net(y, t), self.beta_net(y, t)

    def sample(self):
        alpha, beta = self._compute_forward()
        return alpha + beta * torch.randn_like(alpha)

    def log_prob(self, x):
        alpha, beta = self._compute_forward()
        return -0.5 * (((x - alpha) / beta) ** 2 + torch.log(2 * np.pi * beta ** 2))

    def drift(self, x, mode="ode", sigma_net=None):
        alpha, beta = self._compute_forward()
        if hasattr(self.alpha_net, 'd_dt'):
            d_alpha_dt = self.alpha_net.d_dt(self.y, self.t)
            d_beta_dt  = self.beta_net.d_dt(self.y, self.t)
        else:
            d_alpha_dt = torch.autograd.grad(alpha, self.t, grad_outputs=torch.ones_like(alpha), create_graph=True)[0]
            d_beta_dt  = torch.autograd.grad(beta,  self.t, grad_outputs=torch.ones_like(beta),  create_graph=True)[0]
        drift_term = d_alpha_dt + d_beta_dt * (x - alpha) / beta
        if mode == "ode":
            return drift_term
        sigma = sigma_net(x, self.t)
        sigma_squared = sigma ** 2
        if hasattr(sigma_net, 'd_sq_dx'):
            d_sigma2_dx = sigma_net.d_sq_dx(x, self.t)
        else:
            d_sigma2_dx = torch.autograd.grad(sigma_squared, x, grad_outputs=torch.ones_like(sigma_squared), create_graph=True)[0]
        score = -(x - alpha) / beta ** 2
        if mode == "backward sde":
            drift_term -= d_sigma2_dx / 2 + (sigma_squared / 2) * score
        else:
            drift_term += d_sigma2_dx / 2 + (sigma_squared / 2) * score
        return drift_term, sigma_squared


def visualize_paths_and_marginals(data, t_grid, backward_path, forward_path):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x_range = (-10, 10)
    n_samples = data.shape[0]

    hist_vals, bin_edges = np.histogram(data.detach().numpy(), bins=50, range=x_range, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.plot(np.ones_like(bin_centers), bin_centers, hist_vals, color='b', lw=2, label="Training Data (t=1)")

    x0_backward, backward_paths, x1 = backward_path.sample(data, t_grid)
    for i in range(min(10, n_samples)):
        ax.plot(t_grid, backward_paths[:, i].detach().numpy().squeeze(), np.zeros_like(t_grid),
                lw=1, color='r', alpha=0.7, label="Backward Paths" if i == 0 else "")

    y_samples, forward_paths, x0 = forward_path.sample(n_samples, t_grid)
    for i in range(min(10, n_samples)):
        ax.plot(t_grid, forward_paths[:, i].detach().numpy().squeeze(), np.zeros_like(t_grid),
                lw=1, color='purple', alpha=0.7, label="Forward Paths" if i == 0 else "")

    hist_vals_gen, bin_edges_gen = np.histogram(y_samples.detach().numpy(), bins=50, range=x_range, density=True)
    bin_centers_gen = 0.5 * (bin_edges_gen[1:] + bin_edges_gen[:-1])
    ax.plot(np.ones_like(bin_centers_gen), bin_centers_gen, hist_vals_gen, color='orange', lw=2, label="Generated Data (t=1)")

    hist_vals_prior, bin_edges_prior = np.histogram(x0_backward.detach().numpy(), bins=50, range=x_range, density=True)
    bin_centers_prior = 0.5 * (bin_edges_prior[1:] + bin_edges_prior[:-1])
    ax.plot(np.zeros_like(bin_centers_prior), bin_centers_prior, hist_vals_prior,
            color='b', lw=2, linestyle='--', label="Training Latent (t=0)")

    hist_vals_ps, bin_edges_ps = np.histogram(x0.detach().numpy(), bins=50, range=x_range, density=True)
    bin_centers_ps = 0.5 * (bin_edges_ps[1:] + bin_edges_ps[:-1])
    ax.plot(np.zeros_like(bin_centers_ps), bin_centers_ps, hist_vals_ps,
            color='orange', lw=2, linestyle='--', label="Prior Samples (t=0)")

    plt.ylim(-10, 10)
    ax.set_xlabel('Time')
    ax.set_ylabel('x(t)')
    ax.set_zlabel('Density')
    ax.set_title('Prior Densities, Training Data, and Generated Paths')
    ax.legend()
    plt.show()
