import torch.nn as nn
import torch


class ProxyGenMLP(nn.Module):
    def __init__(self, sz_embed):
        super(ProxyGenMLP, self).__init__()
        self.proxy = nn.Sequential(
            nn.Linear(2 * sz_embed, 2*sz_embed),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(4*sz_embed, sz_embed),
            nn.ReLU()
        )

    def forward(self, embed1, embed2):
        embed = torch.cat(embed1, embed2)
        return self.proxy(embed)


class ProxyGenRNN(nn.Module):
    def __init__(self, sz_embed, num_layers):
        super(ProxyGenRNN, self).__init__()
        self.num_layers = num_layers
        h_dim = int(0.5 * sz_embed)
        self.w_h = [nn.Linear(h_dim, h_dim)]
        self.w_i = [nn.Linear(sz_embed, h_dim)]
        for i in range(num_layers-1):
            self.w_h.append(nn.Linear(h_dim, h_dim))
            self.w_i.append(nn.Linear(h_dim, h_dim))
        self.relu = [nn.ReLU() for _ in range(num_layers)]
        self.dropout = [nn.Dropout(0.2) for _ in range(num_layers)]

        self.final = nn.Linear(h_dim, sz_embed)

    def forward(self, h, x):
        h_new = self.relu1[0](self.dropout[0](self.w_h[0](h[0]) + self.w_i[0](x)))
        h_list = [h]
        for i in range(1, self.num_layers):
            sum = self.w_h[i](h[i]) + self.w_i[i](h_new)
            h_new = self.relu1[i](self.dropout[i](sum))
            h_list.append(h_new)
        x = self.final(h_new)

        return h, x