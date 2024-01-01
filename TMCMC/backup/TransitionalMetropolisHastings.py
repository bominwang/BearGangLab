"""
Transitional Markov chain Monte Carlo (a.k.a Sequential Monte carlo)
The source code was written by Wang BoMin in November 2023
Beijing institute of technology, Beijing, Republic People of CHINA
"""
import torch
import time
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal


# 计算合理性权重
def calculate_plausible_weights(log_likelihood, delta):
    size, dim = log_likelihood.shape
    return torch.exp(delta * log_likelihood).reshape(size, 1)


# 重抽样
def resampling(samples, weights, size):
    indices = torch.multinomial(weights.view(-1), size, replacement=True)
    return samples[indices]


# 求解多元高斯建议分布的协方差矩阵
def ProposalDistributionSampler(mean, covariance):
    proposal_distribution = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=covariance)
    conditional_state = proposal_distribution.sample()
    return conditional_state


class TransitionalMetropolisHastings(object):
    def __init__(self, Graphic=True, GPU=True):
        # 对数目标分布 = 对数似然函数 × 对数先验函数
        self.prior_function = None  # 对数先验函数
        self.likelihood_function = None  # 对数似然函数
        self.TemperingParameter = 0  # 退火因子
        self.stage_num = 0  # 阶段数
        self.Graphic = Graphic  # 阶段可视化
        self.GPU = GPU  # GPU
        self.device = None
        self.beta = 0.2  # 高斯建议分布协方差矩阵缩放因子
        self.dim = None
        self.start_time = None
        self.end_time = None
        self.History_TemperingParameter = []  # 保存退火因子更新轨迹
        self.History_Samples = []  # 保存阶段代码

    # 记录运行时间
    def start_timer(self):
        self.start_time = time.time()

    def end_timer(self):
        self.end_time = time.time()

    def get_elapsed_time(self):
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        else:
            print('Timer not started or ended')
            return None

    # 检测GPU是否可用
    @property
    def check_device(self):
        if self.GPU:
            if torch.cuda.is_available():
                print()
                print('The GPU was successfully identified or detected.')
                print(f'The code is set to run on the {torch.cuda.get_device_name()} for computational processing')
                self.device = torch.device("cuda")
            else:
                print()
                print('The GPU was not successfully identified or detected.')
                print('The code is configured to execute computations on the CPU in the absence of GPU detection.')
                self.device = torch.device("cpu")
            return self.device
        else:
            print()
            print('The code is set to run on the CPU for computational processing.')
            self.device = torch.device("cpu")
            return self.device

    # 利用二分法求解退火因子
    def TemperingParameterEstimator(self, log_likelihood):
        current_tempering_parameter = None
        delta = torch.empty(size=[1], device=self.device)
        old_tempering_parameter = self.TemperingParameter
        min_tempering_parameter = self.TemperingParameter
        max_tempering_parameter = 2.0
        while max_tempering_parameter - min_tempering_parameter > 1e-8:
            current_tempering_parameter = 0.5 * (min_tempering_parameter + max_tempering_parameter)
            delta[0] = current_tempering_parameter - old_tempering_parameter
            weights = calculate_plausible_weights(log_likelihood=log_likelihood, delta=delta)
            weights = weights / torch.sum(weights)

            cv = torch.std(weights) / torch.mean(weights)
            if cv > 1:
                max_tempering_parameter = current_tempering_parameter
            else:
                min_tempering_parameter = current_tempering_parameter
        tempering_parameter = current_tempering_parameter
        if tempering_parameter > 1:
            tempering_parameter = 1
        return tempering_parameter

    def CovarianceComputer(self, samples, scaled_weights, beta):
        size, dim = samples.shape
        scale_factor = (beta ** 2)
        weighted_samples = scaled_weights * samples
        mean_samples = torch.sum(weighted_samples, dim=0, keepdim=True)

        difference = torch.t(samples - mean_samples)
        covariance = torch.zeros(size=[dim, dim], device=self.device)
        for i in range(size):
            cov = torch.mm(difference[:, i].reshape(dim, 1), difference[:, i].reshape(1, dim))
            covariance += cov * scaled_weights[i]
        covariance = covariance * scale_factor
        return covariance

    def MetropolisHastings(self, current_state, covariance, steps_num):
        size, dim = current_state.shape
        samples = current_state.clone()
        for i in range(steps_num):
            conditional_position = ProposalDistributionSampler(mean=current_state, covariance=covariance)
            # 条件样本的后验分布
            log_likelihood1 = self.likelihood_function(conditional_position) * self.TemperingParameter
            log_prior1 = self.prior_function(conditional_position).reshape(-1, 1)
            # 当前样本的后验分布
            log_likelihood2 = self.likelihood_function(current_state) * self.TemperingParameter
            log_prior2 = self.likelihood_function(current_state).reshape(-1, 1)
            # 转换概率
            log_tp1 = torch.zeros(size=[size, 1])
            log_tp2 = torch.zeros(size=[size, 1])
            for i in range(size):
                gauss1 = MultivariateNormal(current_state[i, :], covariance_matrix=covariance)
                gauss2 = MultivariateNormal(conditional_position[i, :], covariance_matrix=covariance)
                log_tp1[i] = gauss2.log_prob(current_state[i, :])
                log_tp2[i] = gauss1.log_prob(conditional_position[i, :])
            alpha = log_likelihood1 - log_likelihood2 + log_prior1 - log_prior2 + log_tp1 - log_tp2
            # 随机数
            u = torch.rand(size=[size, 1])
            condition = (alpha >= 0) | (torch.exp(alpha) > u)
            acceptance_index = torch.where(condition)[0]
            current_state[acceptance_index, :] = conditional_position[acceptance_index, :]
            samples = torch.vstack([samples, current_state])
        return samples

    def sampling(self, log_likelihood, log_prior, initial_position, bounds, num_samples):
        self.likelihood_function = log_likelihood
        self.prior_function = log_prior
        device = self.check_device
        current_state = initial_position.to(device)
        self.History_TemperingParameter.append(self.TemperingParameter)
        self.History_Samples.append(current_state)
        self.start_timer()
        while self.TemperingParameter < 1:
            self.stage_num += 1
            log_likelihood = self.likelihood_function(current_state).to(device)
            tempering_parameter = self.TemperingParameterEstimator(log_likelihood=log_likelihood)
            delta = tempering_parameter - self.TemperingParameter
            weights = calculate_plausible_weights(log_likelihood=log_likelihood, delta=delta)
            scaled_weights = weights / torch.sum(weights)
            covariance = self.CovarianceComputer(samples=current_state, scaled_weights=scaled_weights, beta=0.2)
            print(f' T-MCMC: Iteration {self.stage_num}, tempering parameter = {tempering_parameter}')
            # 更改退火因子
            self.TemperingParameter = tempering_parameter
            # 归一化权重，执行重采样
            current_state = resampling(samples=current_state, weights=scaled_weights, size=num_samples)
            # 基于MCMC-MH进行随机扰动
            current_state = self.MetropolisHastings(current_state=current_state, covariance=covariance, steps_num=30)
            plt.figure(self.stage_num)
            plt.scatter(initial_position[:, 0].cpu(), initial_position[:, 1].cpu(), color='red', alpha=0.5)
            plt.scatter(current_state[:, 0].cpu(), current_state[:, 1].cpu(), color='blue', alpha=0.5)
            plt.show()
            # self.History_TemperingParameter = torch.vstack([self.History_TemperingParameter, self.TemperingParameter])
            # calculate the covariance of MCMC
            # covariance = calculate_covariance(samples=finial_samples, weights=weights, beta=self.beta)
            # running MCMC

            if self.TemperingParameter == 1:
                break
        self.end_timer()

        # adaptively compute the transitional factor
