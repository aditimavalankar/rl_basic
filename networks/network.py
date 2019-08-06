import torch.nn as nn
import torch
from numpy import sqrt
import pdb


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
        # vf = self.sigmoid(vf)
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

    def set_weights_like(self, net):
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

        # self.pre_action_seq = nn.Sequential()
        # self.post_action_seq = nn.Sequential()

        self.input_layer = nn.Linear(input_shape, hidden_layer_size)
        # nn.init.uniform(self.input_layer.weight, -sqrt(2), sqrt(2))
        self.input_layer_norm = nn.LayerNorm(hidden_layer_size)
        # nn.init.uniform_(self.input_layer.weight, -1/sqrt(input_shape),
        #                  1/sqrt(input_shape))
        # self.pre_action_seq.add_module('input', self.input_layer)
        # self.pre_action_seq.add_module('activation0', self.activation)

        self.hidden = nn.Linear(hidden_layer_size + action_dim, hidden_layer_size)
        self.hidden_norm = nn.LayerNorm(hidden_layer_size)

        # for i in range(len(hidden_layers) - 2):
            # self.hidden.append()
            # nn.init.uniform(self.hidden[-1].weight, -sqrt(2), sqrt(2))
            # self.hidden_norm.append(nn.LayerNorm(hidden_layers[i + 1]))
            # nn.init.uniform_(self.hidden.weight,
            #                  -1/sqrt(hidden_layers[i]),
            #                  1/sqrt(hidden_layers[i]))
            # self.pre_action_seq.add_module('hidden'+str(i+1), self.hidden)
            # self.pre_action_seq.add_module('activation'+str(i+1),
            #                                self.activation)

        # self.hidden.append(nn.Linear(hidden_layers[-2] + action_dim,
        #                              hidden_layers[-1]))
        # nn.init.uniform(self.hidden[-1].weight, -sqrt(2), sqrt(2))
        # self.hidden_norm.append(nn.LayerNorm(hidden_layers[-1]))
        # nn.init.uniform_(self.hidden.weight,
        #                  -1/sqrt(hidden_layers[-1]),
        #                  1/sqrt(hidden_layers[-1]))
        # self.post_action_seq.add_module('hidden'+str(len(hidden_layers)),
        #                                 self.hidden)
        # self.post_action_seq.add_module('activation'+str(len(hidden_layers)),
        #                                 self.activation)

        self.final_layer = nn.Linear(hidden_layer_size, output_shape)
        nn.init.uniform_(self.final_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.final_layer.bias, -3e-3, 3e-3)
        # self.post_action_seq.add_module('final', self.final_layer)
        # post_action_modules.append(nn.Tanh())

        # modules = []
        # modules.append(nn.Linear(input_shape + action_dim, hidden_layers[0]))
        # nn.init.uniform_(modules[-1].weight, -3e-3, 3e-3)
        # modules.append(activation)
        # for i in range(len(hidden_layers) - 1):
        #     modules.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        #     nn.init.uniform_(modules[-1].weight, -3e-3, 3e-3)
        #     modules.append(activation)
        # modules.append(nn.Linear(hidden_layers[-1], output_shape))
        # self.sequential = nn.Sequential(*modules)

    def forward(self, s, a):
        out = self.input_layer(s)
        out = self.input_layer_norm(out)
        out = self.activation(out)
        out = self.hidden(torch.cat((out, a), dim=1))
        out = self.hidden_norm(out)
        out = self.activation(out)
        out = self.final_layer(out)
        # value = self.sequential(torch.cat((s, a), dim=-1))
        return out

    def set_weights_like(self, net):
        self.load_state_dict(net.state_dict())

    def set_weights(self, state_dict):
        self.load_state_dict(state_dict)

    def set_weighted_weights(self, net, tau):
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

        # self.sequential = nn.Sequential()

        self.input_layer = nn.Linear(input_shape, hidden_layer_size)
        self.input_layer_norm = nn.LayerNorm(hidden_layer_size)
        # nn.init.uniform_(self.input_layer.weight, -1/sqrt(input_shape),
        #                  1/sqrt(input_shape))
        # nn.init.uniform(self.input_layer.weight, -sqrt(2), sqrt(2))
        # self.sequential.add_module('input', self.input_layer)
        # self.sequential.add_module('activation0', self.activation)

        # for i in range(len(hidden_layers) - 1):
        #     self.hidden = nn.Linear(hidden_layers[i], hidden_layers[i + 1])
        #     nn.init.uniform(self.hidden.weight, -sqrt(2), sqrt(2))
            # nn.init.uniform_(self.hidden.weight,
            #                  -1/sqrt(hidden_layers[i]),
            #                  1/sqrt(hidden_layers[i]))
            # self.sequential.add_module('hidden'+str(i+1), self.hidden)
            # self.sequential.add_module('activation'+str(i+1), self.activation)

        self.hidden = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.hidden_norm = nn.LayerNorm(hidden_layer_size)

        self.final_layer = nn.Linear(hidden_layer_size, output_shape)
        nn.init.uniform_(self.final_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.final_layer.bias, -3e-3, 3e-3)

        self.tanh = nn.Tanh()
        # self.sequential.add_module('final', self.final_layer)
        # self.sequential.add_module('activation_final', nn.Tanh())

    def forward(self, s):
        out = self.input_layer(s)
        out = self.input_layer_norm(out)
        out = self.activation(out)
        out = self.hidden(out)
        out = self.hidden_norm(out)
        out = self.activation(out)
        out = self.final_layer(out)
        out = self.tanh(out)
        return out

    def set_weights_like(self, net):
        self.load_state_dict(net.state_dict())

    def set_weights(self, state_dict):
        self.load_state_dict(state_dict)

    def set_weighted_weights(self, net, tau):
        for (old_p, new_p) in zip(self.parameters(), net.parameters()):
            old_p.data.copy_(tau * new_p + (1 - tau) * old_p)
