
import math
import torch
from torch import nn
from torch.nn import functional as F
import math
from .utils import *
from torch_scatter import scatter_add

class Matrix_Self_Attention_Block(nn.Module):
    def __init__(self, in_channels, inter_channels=None, nhead=1): #'attention_map', 'masked_image', with_residual''
        super(Matrix_Self_Attention_Block, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels

        self.nhead = nhead
        self.h_dim = int(self.inter_channels/self.nhead)

        conv_nd = nn.Conv2d
        
        # same as first reshaping and then applying MLP to the channel dimension
        self.q = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.k = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        
        self.v = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.out = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.sm = nn.Softmax(dim=1)


    def forward(self, x, x_g):
        '''
        :param x = new detections [x1, x1, x2, x2, x3, x3]
        :param x_g = tracklets (guidance) [x_g1, xg_2, x_g1, xg_2, x_g1, xg_2]
        :param attention = x: update feature map of x
        :param attention = x_g: update feature map of guidance 
        :return:
        '''

        batch_size, _, h, w = x.size()
        # [Comparisons (BS), C, H, W] --> [Comparisons (BS), HW, IC] same as sequence input now
        q = self.q(x_g).view(batch_size, -1, self.inter_channels) # BS X HW x IC
        k = self.k(x).view(batch_size, -1, self.inter_channels) # BS X HW x IC
        v = self.v(x).view(batch_size, -1, self.inter_channels) # BS X HW x InC

        q = q.view(batch_size, -1, self.nhead, self.h_dim).transpose(1, 2) # BS X HW X NHeads X HDim 
        k = k.view(batch_size, -1, self.nhead, self.h_dim).transpose(1, 2) # BS X HW X NHeads X HDim  
        v = v.view(batch_size, -1, self.nhead, self.h_dim).transpose(1, 2) # BS X HW X NHeads X InC  

        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attention = self.sm(torch.matmul(q, k.transpose(-2, -1))/math.sqrt(k.shape[2])) 
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        attended = torch.matmul(attention, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.inter_channels).transpose(1, 2)
        attended = attended.view(batch_size, -1, h, w)

        attended = self.out(attended)
        
        return attended


class TransformerLayer(nn.Module):
    # corresponds to Transformer decoder layer
    def __init__(self, in_channels, inter_channels=None, gnn_params=None, postprocessing=True, num_classes=751):
        super(TransformerLayer, self).__init__()
        self.attention = Matrix_Self_Attention_Block(in_channels=in_channels, 
                inter_channels=inter_channels)

        self.res1 = gnn_params['res1']
        self.res2 = gnn_params['res2']

        self.dropout1 = nn.Dropout(gnn_params['dropout_1'])
        # equivalent to mpl along embedding dim
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) if gnn_params['mlp'] else None 
        self.act = F.relu
        self.dropout = nn.Dropout(gnn_params['dropout_mlp'])
        # equivalent to mpl along embedding dim
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) if gnn_params['mlp'] else None

        self.norm = nn.LayerNorm([in_channels, 8, 4]) if gnn_params['norm1'] else None
        self.dropout2 = nn.Dropout(gnn_params['dropout_2'])
        self.norm2 = nn.LayerNorm([in_channels, 8, 4]) if gnn_params['norm2'] else None


    def forward(self, x, x_g=None, num_gallery=None, aggregate=True, classify=False):
        '''
        x would be detections
        x_g woule be tracklets
        '''
        if x_g is None:
            x_g = x

        # from x_guidance to x
        A = torch.where(torch.ones((x_g.shape[0], x.shape[0])) > 0)
        gs, qs = A[0], A[1]
        x_attended = self.attention(x[qs].cuda(x.get_device()), x_g[gs].cuda(x.get_device())) # x, x_guidance
        x_attended = scatter_add(x_attended, qs.cuda(x.get_device()), dim=0)

        # "post message processing"
        x2 = self.dropout1(x_attended)
        x = x2 + x if self.res1 else x2
        x = self.norm(x) if self.norm is not None else x
        x2 = self.conv2(self.dropout(self.act(self.conv1(x)))) if self.conv1 else x
        x2 = self.dropout2(x2)
        x = x + x2 if self.res2 else x2
        x = self.norm2(x) if self.norm2 is not None else x

        return x      


class GNNNetwork(nn.Module):
    def __init__(self, in_channels, gnn_params, num_layers, red=4):
        super(GNNNetwork, self).__init__()
        #print("Repeat same layer")
        #gnn = DotAttentionLayer(embed_dim, aggr, dev,
        #                            edge_dim, gnn_params)
        #layers = [gnn for _ in range(num_layers)]
        new_dim = int(in_channels/red)
        self.downsample = nn.Conv2d(in_channels, new_dim, kernel_size=1)
        layers = [TransformerLayer(new_dim, gnn_params=gnn_params) for _
                  in range(num_layers)]

        self.layers = Sequential(*layers)

    def forward(self, feats, edge_index, edge_attr):
        feats = self.downsample(feats)
        out = list()
        for layer in self.layers:
            feats = layer(feats) 
            out.append(feats)

        return out, edge_index, edge_attr


class SpatialGNNReIDTransformer(nn.Module):
    def __init__(self, dev, params: dict = None, embed_dim: int = 2048):
        super(SpatialGNNReIDTransformer, self).__init__()
        num_classes = params['classifier']['num_classes']
        self.dev = dev
        self.params = params
        self.gnn_params = params['gnn']

        red = 1
        logger.info("Reduce number of channels by {} (hardcoded)".format(red))
        
        self.gnn_model = GNNNetwork(embed_dim, self.gnn_params, self.gnn_params['num_layers'], red=red)
        embed_dim = int(embed_dim/red)

        print("Using avg pool")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #print("Using max pool")
        #self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        embed_dim = int(embed_dim/params['red'])
        self.dim_red = nn.Linear(embed_dim, int(embed_dim/params['red'])) if params['red'] != 1 else None

        # classifier
        self.neck = params['classifier']['neck']
        dim = self.gnn_params['num_layers'] * embed_dim if self.params['cat'] else embed_dim
        every = self.params['every']
        if self.neck:
            layers = [nn.BatchNorm1d(dim) for _ in range(self.gnn_params['num_layers'])] if every else [nn.BatchNorm1d(dim)]
            self.bottleneck = Sequential(*layers)
            for layer in self.bottleneck:
                layer.bias.requires_grad_(False)
                layer.apply(weights_init_kaiming)
            
            layers = [nn.Linear(dim, num_classes, bias=False) for _ in range(self.gnn_params['num_layers'])] if every else [nn.Linear(dim, num_classes, bias=False)]
            self.fc = Sequential(*layers)
            for layer in self.fc:
                layer.apply(weights_init_classifier)
        else:
            layers = [nn.Linear(dim, num_classes) for _ in range(self.gnn_params['num_layers'])] if every else [nn.Linear(dim, num_classes)]
            self.fc = Sequential(*layers)

    def forward(self, feats, edge_index, edge_attr=None, Y=None, output_option='norm', mode='test'):
        r, c = edge_index[:, 0], edge_index[:, 1]

        feats, _, _ = self.gnn_model(feats, edge_index, edge_attr)
        if not self.params['every']:
            feats = feats[-1]
            fc7 = self.avgpool(feats)
            fc7 = torch.flatten(fc7, 1)
            if self.dim_red is not None:
                fc7 = self.dim_red(fc7)
            fc7 = [fc7]
        else:
            fc7 = [torch.flatten(self.avgpool(f), 1) for f in feats]
            if self.dim_red is not None:
                fc7 = [self.dim_red(f) for f in fc7]

        if self.neck:
            features = [l(f) for l, f in zip(self.bottleneck, fc7)]
        else:
            features = [fc7] if type(fc7) is not list else fc7

        x = [fc(f) for fc, f in zip(self.fc, features)]

        if output_option == 'norm':
            return x, features, Y
        elif output_option == 'plain':
            return x, [F.normalize(f, p=2, dim=1) for f in features], Y
        elif output_option == 'neck' and self.neck:
            return x, features, Y
        elif output_option == 'neck' and not self.neck:
            print("Output option neck only avaiable if bottleneck (neck) is "
                  "enabeled - giving back x and fc7")
            return x, features, Y

        return x, feats, Y

