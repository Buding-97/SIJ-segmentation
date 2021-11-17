"""
This is the implementation of following paper:
https://arxiv.org/pdf/1802.05591.pdf
"""
from torch.autograd import Variable
import torch
import torch.nn as nn

class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_var=0.5, delta_dist=1.5,
                 norm=1, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=True):
        super(DiscriminativeLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target):
        # with torch.no_grad():
        return self._discriminative_loss(input, target)

    def _discriminative_loss(self, input, target):
        c_means= self._cluster_means(input, target)
        l_var = self._variance_term(input, target, c_means)
        l_dist = self._distance_term(target,c_means)
        l_reg = self._regularization_term(c_means)
        return self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg

    def _cluster_means(self, input, target):
        '''
        :param input: torch.size(B,5,4096)
        :param target: torch.size(B,4096)
        :return: torch.size(B,5,4096)
        '''
        bs, n_feature, num_points = input.size()
        means = []
        for i in range(bs):
            num_cluster = torch.unique(target[i])
            mean = torch.zeros_like(input[i]) #torch.size(5,4096)
            for j in num_cluster:
                target_flage = target[i] == j #torch.size(4096) type(Boolean)
                input_sample = input[i] * target_flage.unsqueeze(0).expand(n_feature,num_points) #torch.size(5,4096) feature of the j-th instance
                mean_sample = input_sample.sum(1) / target_flage.sum() #torch.size(5)
                m = target_flage.unsqueeze(0).expand(n_feature,num_points) * mean_sample.unsqueeze(1) #torch.size(5,4096)
                mean += m
            means.append(mean)
        means = torch.stack(means)
        return means.cuda()

    def _variance_term(self, input, target, c_means):
        '''
        :param input: torch.size(B,5,4096)
        :param target: torch.size(B,4096)
        :param c_means: torch.size(B,5,4096)
        :return:
        '''
        var = (torch.clamp(torch.norm((input - c_means), self.norm, 1) - self.delta_var, min=0) ** 2)
        bs, n_feature, num_points = input.size()
        var_term = 0
        for i in range(bs):
            num_cluster = torch.unique(target[i])
            for j in num_cluster:
                target_flage = target[i] == j #torch.size(4096) type(Boolean)
                c_var = (var[i] * target_flage).sum()/target_flage.sum()
                var_term += c_var
            var_term / len(num_cluster)
        var_term /= bs
        return var_term

    def _distance_term(self, target,c_means):
        '''
        :param c_means: torch.size(B,5,4096)
        :return:
        '''
        bs, n_features, num_points = c_means.size()
        dist_term = 0
        for i in range(bs):
            num_cluster = torch.unique(target[i])
            mean_cluster = c_means[i][:,num_cluster]
            if mean_cluster.shape[1] <= 1:
                continue
            means_a = mean_cluster.unsqueeze(2).expand(n_features, mean_cluster.shape[1], mean_cluster.shape[1])
            means_b = means_a.permute(0, 2, 1)
            diff = means_a - means_b
            margin = 2 * self.delta_dist * (1.0 - torch.eye(mean_cluster.shape[1]))
            if self.usegpu:
                margin = margin.cuda()
            c_dist = torch.sum(torch.clamp(margin - torch.norm(diff, self.norm, 0), min=0) ** 2)
            dist_term += c_dist / (mean_cluster.shape[1] * (mean_cluster.shape[1]-1))
        dist_term /= bs
        return dist_term

    def _regularization_term(self, c_means):
        bs, n_features, num_points = c_means.size()
        c_means = Variable(c_means)
        reg_term = 0
        for i in range(bs):
            # n_features, n_clusters
            mean_cluster = torch.unique(c_means[i],dim=1)
            reg_term += torch.mean(torch.norm(mean_cluster, self.norm, 0))
        reg_term /= bs
        return reg_term