import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    """Taken from https://github.com/ikostrikov/pytorch-ddpg-naf."""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class ac_network(nn.Module):
    """The policy and value networks share weights."""
    def __init__(self, input_shape, action_dim):
        super(ac_network, self).__init__()
        self.hidden1 = nn.Linear(input_shape, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.pi = nn.Linear(64, action_dim * 2)
        self.vf = nn.Linear(64, 1)

    def forward(self, x):
        out = self.hidden1(x)
        out = self.tanh(out)
        out = self.hidden2(out)
        out = self.tanh(out)
        pi = self.pi(out)
        vf = self.vf(out)
        pi = self.tanh(pi)
        return pi, vf

    def set_weights(self, state_dict):
        self.load_state_dict(state_dict)


class mlp(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers=[64, 64],
                 activation='tanh'):
        super(mlp, self).__init__()
        modules = []

        if activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'relu':
            activation = nn.ReLU()

        modules.append(nn.Linear(input_shape, hidden_layers[0]))
        modules.append(activation)
        for i in range(len(hidden_layers) - 1):
            modules.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            modules.append(activation)
        modules.append(nn.Linear(hidden_layers[-1], output_shape))
        modules.append(nn.Tanh())
        self.sequential = nn.Sequential(*modules)
        for m in self.sequential.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.5)

    def forward(self, x):
        return self.sequential(x)

    def hard_update(self, net):
        self.load_state_dict(net.state_dict())

    def set_weights(self, state_dict):
        self.load_state_dict(state_dict)


class ddpg_critic(nn.Module):
    def __init__(self, input_shape, output_shape, action_dim,
                 hidden_layer_size=128, activation='relu'):
        super(ddpg_critic, self).__init__()

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.input_layer = nn.Linear(input_shape, hidden_layer_size)
        self.input_layer_norm = LayerNorm(hidden_layer_size)

        self.hidden = nn.Linear(hidden_layer_size + action_dim, hidden_layer_size)
        self.hidden_norm = LayerNorm(hidden_layer_size)

        self.final_layer = nn.Linear(hidden_layer_size, output_shape)
        self.final_layer.weight.data.mul_(0.1)
        self.final_layer.bias.data.mul_(0.1)

    def forward(self, s, a):
        out = self.input_layer(s)
        out = self.input_layer_norm(out)
        out = self.activation(out)
        out = self.hidden(torch.cat((out, a), dim=1))
        out = self.hidden_norm(out)
        out = self.activation(out)
        out = self.final_layer(out)
        return out

    def hard_update(self, net):
        for (old_p, new_p) in zip(self.parameters(), net.parameters()):
            old_p.data.copy_(new_p)

    def set_weights(self, state_dict):
        self.load_state_dict(state_dict)

    def soft_update(self, net, tau):
        for (old_p, new_p) in zip(self.parameters(), net.parameters()):
            old_p.data.copy_(tau * new_p + (1 - tau) * old_p)


class ddpg_actor(nn.Module):
    def __init__(self, input_shape, output_shape,
                 hidden_layer_size=128, activation='relu'):
        super(ddpg_actor, self).__init__()

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.input_layer = nn.Linear(input_shape, hidden_layer_size)
        self.input_layer_norm = LayerNorm(hidden_layer_size)

        self.hidden = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.hidden_norm = LayerNorm(hidden_layer_size)

        self.final_layer = nn.Linear(hidden_layer_size, output_shape)
        self.final_layer.weight.data.mul_(0.1)
        self.final_layer.bias.data.mul_(0.1)
        self.final_activation = nn.Tanh()

    def forward(self, s):
        out = self.input_layer(s)
        out = self.input_layer_norm(out)
        out = self.activation(out)
        out = self.hidden(out)
        out = self.hidden_norm(out)
        out = self.activation(out)
        out = self.final_layer(out)
        out = self.final_activation(out)
        return out

    def hard_update(self, net):
        for (old_p, new_p) in zip(self.parameters(), net.parameters()):
            old_p.data.copy_(new_p)

    def set_weights(self, state_dict):
        self.load_state_dict(state_dict)

    def soft_update(self, net, tau):
        for (old_p, new_p) in zip(self.parameters(), net.parameters()):
            old_p.data.copy_(tau * new_p + (1 - tau) * old_p)
