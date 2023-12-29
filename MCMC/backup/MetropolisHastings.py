"""
Markov chain Monte Carlo (Metropolis-Hastings Algorithm)
The source code was written by BoMin Wang in November 2023
Beijing institute of technology, Beijing, Republic People of CHINA
"""
import torch
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt


class MetropolisHastings(object):
    def __init__(self, proposal_distribution='uniform', transfer_method='block-wise', GPU=True):

        if proposal_distribution not in ['uniform', 'gauss']:
            raise ValueError("Invalid value for proposal_distribution. Choose 'uniform' or 'gauss'.")

        if transfer_method not in ['block-wise', 'component-wise']:
            raise ValueError(f"Invalid value for transfer_method. Choose 'block-wise' or 'component-wise'.")

        self.proposal_distribution = proposal_distribution
        self.transfer_method = transfer_method
        self.GPU = GPU
        self.device = None
        self.target_distribution = None
        self.width = None
        self.sigma = None

        self.acceptance_rates = []

    @property
    def check_device(self):
        if self.GPU:
            if torch.cuda.is_available():
                print()
                print('The GPU was successfully identified or detected.')
                print(f'The code is set to run on the {torch.cuda.get_device_name()} for computational processing')
                self.device = torch.device("cuda")
            else:
                print('The GPU was not successfully identified or detected.')
                print('The code is configured to execute computations on the CPU in the absence of GPU detection.')
                self.device = torch.device("cpu")
            return self.device
        else:
            print('The code is set to run on the CPU for computational processing.')
            self.device = torch.device("cpu")
            return self.device

    def block_wise_conditional_distribution(self, current_state):
        # performing multidimensional sampling
        conditional_state = None
        size, dim = current_state.shape

        if self.proposal_distribution == 'uniform':
            # x = lower + (upper - lower) * U(0, 1)
            conditional_state = torch.rand(size=[size, dim]).to(self.device)
            delta = self.width.reshape(1, dim) * torch.ones_like(current_state)
            lower, upper = current_state - delta, current_state + delta
            conditional_state = lower + (upper - lower) * conditional_state
        elif self.proposal_distribution == 'gauss':
            # x = mean + N(0, 1) * sigma
            conditional_state = torch.randn(size=[size, dim]).to(self.device)
            sigma = self.sigma.reshape(1, dim) * torch.ones_like(current_state)
            conditional_state = conditional_state * sigma + current_state

        conditional_state = conditional_state.reshape(current_state.shape)
        return conditional_state

    def component_wise_condition_distribution(self, current_state, idx):
        # sampling each dimension independently of to others
        conditional_state = None
        size, dim = current_state.shape
        if self.proposal_distribution == 'uniform':
            # x = lower + (upper-lower) * U(0, 1)
            conditional_state = torch.rand(size=current_state.shape).to(self.device)
            delta = self.width[idx] * torch.ones_like(current_state)
            lower, upper = current_state - delta, current_state + delta
            conditional_state = lower + (upper - lower) * conditional_state
        elif self.proposal_distribution == 'gauss':
            # x = mean + sigma * N(0, 1)
            conditional_state = torch.randn(size=current_state.shape).to(self.device)
            sigma = self.sigma[idx] * torch.ones_like(current_state)
            conditional_state = conditional_state * sigma + current_state
        return conditional_state

    def AcceptanceRateCalculator(self, current_state, conditional_state, idx=None):
        alpha = None
        if idx is None:
            size, dim = current_state.shape
            if self.proposal_distribution == 'uniform':
                delta = self.width.reshape(1, dim) * torch.ones_like(conditional_state)
                lower, upper = conditional_state - delta, conditional_state + delta
                alpha1 = Uniform(lower, upper).log_prob(current_state)
                lower, upper = current_state - delta, current_state + delta
                alpha2 = Uniform(lower, upper).log_prob(conditional_state)
                # log p(theta* -> theta) - log p(theta -> theta*)
                alpha = torch.sum(alpha1, dim=1, keepdim=True) - torch.sum(alpha2, dim=1, keepdim=True)
            elif self.proposal_distribution == 'gauss':
                alpha1 = torch.zeros(size=[size, 1]).to(self.device)
                alpha2 = torch.zeros(size=[size, 1]).to(self.device)
                for idx in range(size):
                    cov = torch.diag(self.sigma ** 2).reshape(dim, dim)
                    mean1 = conditional_state[idx, :].reshape(1, -1)
                    mean2 = current_state[idx, :].reshape(1, -1)
                    alpha1[idx, 0] = MultivariateNormal(loc=mean1, covariance_matrix=cov).log_prob(mean2)
                    alpha2[idx, 0] = MultivariateNormal(loc=mean2, covariance_matrix=cov).log_prob(mean1)
                alpha = alpha1 + alpha2
        if idx is not None:
            current_state = current_state.reshape(-1, 1)
            conditional_state = conditional_state.reshape(-1, 1)
            if self.proposal_distribution == 'uniform':
                delta = self.width[idx] * torch.ones_like(current_state)
                lower, upper = conditional_state - delta, conditional_state + delta
                alpha1 = Uniform(lower, upper).log_prob(current_state)
                lower, upper = current_state - delta, current_state + delta
                alpha2 = Uniform(lower, upper).log_prob(conditional_state)
                alpha = alpha1 - alpha2
            elif self.proposal_distribution == 'gauss':
                cov = self.sigma[idx] * torch.ones_like(conditional_state)
                alpha1 = Normal(conditional_state, cov).log_prob(current_state)
                alpha2 = Normal(current_state, cov).log_prob(conditional_state)
                alpha = alpha1 - alpha2
        return alpha

    def MetropolisHastingsIterator(self, current_state):
        size, dim = current_state.shape
        if self.transfer_method == 'block-wise':
            conditional_state = self.block_wise_conditional_distribution(current_state=current_state)
            alpha = self.AcceptanceRateCalculator(current_state=current_state, conditional_state=conditional_state)
            alpha += self.target_distribution(conditional_state) - self.target_distribution(current_state)
            u = torch.rand(size=alpha.shape).to(self.device)
            condition = (alpha >= 0) | (torch.exp(alpha) > u)
            acceptance_index = torch.where(condition)[0]
            current_state[acceptance_index, :] = conditional_state[acceptance_index, :]

        elif self.transfer_method == 'component-wise':
            size, dim = current_state.shape
            for idx in range(dim):
                conditional_state = current_state.clone()
                component_state = current_state[:, idx].reshape(-1, 1)
                component_state = self.component_wise_condition_distribution(current_state=component_state,
                                                                             idx=idx)
                alpha = self.AcceptanceRateCalculator(current_state=current_state[:, idx],
                                                      conditional_state=component_state, idx=idx)
                conditional_state[:, idx] = component_state.squeeze()
                alpha += self.target_distribution(conditional_state) - self.target_distribution(current_state)
                u = torch.rand(size=alpha.shape).to(self.device)
                condition = (alpha >= 0) | (torch.exp(alpha) > u)
                acceptance_index = torch.where(condition)[0]
                current_state[acceptance_index, :] = conditional_state[acceptance_index, :]
        return current_state

    def Sampling(self, target_distribution, initial_state, num_samples, steps_warm_up=None,
                 width=1.0, sigma=1.0) -> torch.Tensor:

        self.device = self.check_device
        self.target_distribution = target_distribution

        initial_state = initial_state.to(self.device)

        size, dim = initial_state.shape
        samples = torch.zeros_like(initial_state).to(self.device)  # 保存样本

        # 对建议分布超参数进行扩维
        if self.proposal_distribution == 'uniform':
            width = torch.ones(size=[1, dim]) * width
            self.width = width.to(self.device)
        elif self.proposal_distribution == 'gauss':
            sigma = torch.ones(size=[1, dim]) * sigma
            self.sigma = sigma.to(self.device)

        if steps_warm_up is None:
            steps_warm_up = 3 * num_samples

        current_state = initial_state

        print('Running warm-up phase:')
        for _ in range(steps_warm_up):
            conditional_state = self.MetropolisHastingsIterator(current_state=current_state)
            current_state = conditional_state
            progress = int((_ + 1) / steps_warm_up * 20)
            print("\r[{}{}] {:.2f}%".format("=" * progress, " " * (20 - progress), (_ + 1) / steps_warm_up * 100),
                  end="", flush=True)
        print()

        print('Running Sampling phase:')
        for i in range(num_samples):
            conditional_state = self.MetropolisHastingsIterator(current_state=current_state)
            current_state = conditional_state
            samples = torch.vstack([samples, current_state])
            progress = int((i + 1) / num_samples * 20)
            print("\r[{}{}] {:.2f}%".format("=" * progress, " " * (20 - progress), (i + 1) / num_samples * 100), end="",
                  flush=True)
        print()
        samples = samples[size:, :]
        return samples
