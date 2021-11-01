
import torch
from torch import nn
from torch.nn import functional as F
from time import time
from torch_scatter import scatter_add
from .utils import *
import torch.nn.functional as F

class _Query_Guided_Attention(nn.Module):
    def __init__(self, in_channels, inter_channels=None, norm='size', message='attention_map'): #'attention_map', 'masked_image', with_residual''
        super(_Query_Guided_Attention, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.norm = norm
        self.message = message
        print("Sending the following message {}".format(self.message))

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        self.max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.sm = nn.Softmax(dim=2)
        self.sigi = nn.Sigmoid()


    def forward(self, x_gallery, x_query, attention="x"):
        '''
        :param x = new detections
        :param x_query = tracklets (guidance)
        :param attention = x: update feature map of x
        :param attention = x_query: update feature map of guidance 
        :return:
        '''

        batch_size = x_gallery.size(0)
        # reduce channel dimension
        theta_x = self.theta(x_gallery) # --> BS x C' x H x W
        phi_x = self.phi(x_query) # --> BS x C' x H x W

        if attention == "x":
            theta_x = theta_x.view(batch_size, self.inter_channels, -1).permute(0, 2, 1) #x --> shape BS x HW x C'
            phi_x = phi_x.view(batch_size, self.inter_channels, -1) #x_query --> shape BS X C' x HW
            #phi_x = self.max_pool_layer(phi_x).view(batch_size, self.inter_channels, -1) #x_query (pyramid == s2) --> shape BS x C' x H'W'
            # get attention map
            f = torch.matmul(theta_x, phi_x) # BS x HW x HW (--> H'W' for s2)
          
            if self.norm == 'size': # from original paper
                N = f.size(-1) # --> dimension of guidance HW (--> H'W' for s2)
                f = f / N

            if self.norm == 'channels': # transformer style
                N = phi_x.shape[1]
                f = f / N

            f, _ = torch.max(f, 2) # GMP --> here we diverge from transformers
            f = f.view(batch_size, *x_gallery.size()[2:]).unsqueeze(1)
            
            if self.message == 'attention_map':
                return self.sigi(f)

            # mask image
            z = x_gallery * f 
            if self.message == 'masked_image':
                return z

            elif self.message == 'with_residual': # from original paper
                z = z + x_gallery
                return z


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


class Query_Guided_Attention_Layer(nn.Module):
    def __init__(self, in_channels, inter_channels=None,
            gnn_params=None, post_agg=False, num_classes=751, 
            post_dist=False, post_non_agg=False, aggregate=False, 
            class_agg=False, non_agg=False, class_non_agg=True, 
            distance=False, aggr=None, neck=False):
        super(Query_Guided_Attention_Layer, self).__init__()
        self.aggregate = aggregate
        self.class_agg = class_agg
        self.post_agg = post_agg
        
        self.non_agg = non_agg
        self.class_non_agg = class_non_agg
        self.post_non_agg = post_non_agg

        self.distance = distance
        self.post_dist = post_dist

        self.aggr = aggr
        self.neck = neck

        print('Aggregate branch', self.aggregate, self.class_agg, self.post_agg)
        print('Non-aggregate branch', self.non_agg, self.class_non_agg, self.post_non_agg)
        print('Distance branch', self.distance, self.post_dist)

        if inter_channels is None:
            inter_channels = in_channels

        self.query_guided_attention = _Query_Guided_Attention(in_channels=in_channels, 
                inter_channels=inter_channels)
        
        #self.query_guided_attention = Matrix_Self_Attention_Block(in_channels=in_channels,
        #        inter_channels=inter_channels)

        self.res1 = gnn_params['res1']
        self.res2 = gnn_params['res2']

        if self.aggr is None and self.aggregate:
            self.aggr = lambda out, row, dim, x_size: scatter_add(out,
                                                                   row,
                                                                   dim=dim,
                                                                   dim_size=x_size)
        if self.post_non_agg or self.post_agg or self.post_dist:
            self.dropout1 = nn.Dropout(gnn_params['dropout_1'])
            self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) if gnn_params['mlp'] else None #kernel_size 3 or 1? if 3: padding=1 
            self.act = F.relu
            self.dropout = nn.Dropout(gnn_params['dropout_mlp'])
            self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) if gnn_params['mlp'] else None

            self.norm = nn.LayerNorm([in_channels, 8, 4]) if gnn_params['norm1'] else None
            self.dropout2 = nn.Dropout(gnn_params['dropout_2'])
            self.norm2 = nn.LayerNorm([in_channels, 8, 4]) if gnn_params['norm2'] else None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.distance:
            self.final = nn.Sequential(
                nn.Linear(in_channels, int(in_channels/2)),
                nn.ReLU(),
                nn.Linear(int(in_channels/2), 1))
            self.sig = nn.Sigmoid()

        if self.non_agg:
            if self.neck:
                self.bottleneck = nn.BatchNorm1d(in_channels)
                self.bottleneck.bias.requires_grad_(False)  # no shift
                self.fc = nn.Linear(in_channels, num_classes, bias=False)

                self.bottleneck.apply(weights_init_kaiming)
                self.fc.apply(weights_init_classifier)

            else:
                self.fc = nn.Linear(in_channels, num_classes)

        if self.aggregate:
            self.fc_agg = nn.Linear(in_channels, num_classes)

    def forward(self, x, num_query=None, output_option='plain'):

        size = (x.shape[0], x.shape[0])
        if num_query is None:
            # let every sample attend over every sample but no "self-loop-attention"
            A = torch.where(torch.ones(size)-torch.diag(torch.ones(x.shape[0])) > 0)
            #A = torch.where(torch.ones(size) > 0)
            qs = A[0] # from query
            gs = A[1] # to gallery

            attended_gallery = self.query_guided_attention(x[gs].cuda(x.get_device()), x[qs].cuda(x.get_device())) # x, x_query
        else:
            A = torch.zeros(size)
            A[:num_query, num_query:] = 1

            # if with self-attention
            #for i in range(num_query):
            #    A[i, i] = 1
 
            A = torch.where(A > 0)
            qs = A[0] # from query
            gs = A[1] # to gallery
            attended_gallery = self.query_guided_attention(x[gs], x[qs])

        # idea: aggregate attention maps for Graph Neural Netowrk
        if self.aggregate:
            x2 = self.aggr(attended_gallery, gs.cuda(x.get_device()), 0, x.shape[0])
            if self.query_guided_attention.message == 'attention_map':
                x2 = x * x2  

            # transformer style
            if self.post_agg: 
                x2 = self.dropout1(x2)
                x = x2 + x if self.res1 else x2
                x = self.norm(x) if self.norm is not None else x
                x2 = self.conv2(self.dropout(self.act(self.conv1(x)))) if self.conv1 else x
                x2 = self.dropout2(x2)
                x = x + x2 if self.res2 else x2
                x = self.norm2(x) if self.norm2 is not None else x
            else:
                x = x2

            # classify aggregated feature maps and get embedding
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if self.class_agg:
                fc_x_agg = self.fc_agg(x)        
            aggregated = x

        else:
            aggregated = fc_x_agg = None

        # idea: compute distance directly from attention matrix
        if self.distance:

            # transformer style
            if self.post_dist:
                x = self.dropout1(x)
                x = self.norm(x) if self.norm is not None else x
                x = self.conv2(self.dropout(self.act(self.conv1(x)))) if self.conv1 else x
                x = self.dropout2(x)
                x = self.norm2(x) if self.norm2 is not None else x
            else:
                x = attended_gallery

            # get distance from attention matrix
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.final(x)
            x = self.sig(x)
            dist = torch.zeros(size).cuda(x.get_device())
            dist[qs, gs] = x.squeeze()
        else:
            dist = None

        # idea: classidy each gallery that was attendet over
        if self.non_agg:

            _x = x[gs].cuda(x.get_device())
            x2 = _x * attended_gallery

            # transformer style
            if self.post_non_agg:
                x2 = self.dropout1(x2)
                _x = x2 + _x if self.res1 else x2
                _x = self.norm(_x) if self.norm is not None else _x
                x2 = self.conv2(self.dropout(self.act(self.conv1(_x)))) if self.conv1 else _x
                x2 = self.dropout2(x2)
                _x = _x + x2 if self.res2 else x2
                _x = self.norm2(_x) if self.norm2 is not None else _x
            else:
                _x = x2 + _x

            # classify attended gallery images 
            _x = self.avgpool(_x)
            _x = torch.flatten(_x, 1)

            if self.neck:
                _x = self.bottleneck(_x)

            if self.class_non_agg:
                fc_x = self.fc(_x)
            
            if output_option == 'plain':
                _x = F.normalize(_x, p=2, dim=1)

        else:
            fc_x = _x = None

        return aggregated, fc_x_agg, dist, qs, gs, _x, fc_x, attended_gallery


class GNNNetwork(nn.Module):
    def __init__(self, in_channels, aggr, dev, gnn_params, num_layers, red=4):
        super(GNNNetwork, self).__init__()
        #print("Repeat same layer")
        #gnn = DotAttentionLayer(embed_dim, aggr, dev,
        #                            edge_dim, gnn_params)
        #layers = [gnn for _ in range(num_layers)]
        new_dim = int(in_channels/red)
        self.downsample = nn.Conv2d(in_channels, new_dim, kernel_size=1)
        print(self.downsample)
        layers = [Query_Guided_Attention_Layer(new_dim, aggr=aggr, gnn_params=gnn_params) for _
                  in range(num_layers)]

        self.layers = Sequential(*layers)

    def forward(self, feats, edge_index, edge_attr):
        feats = self.downsample(feats)
        out = list()
        for layer in self.layers:
            # feats = aggregated from QGAL
            feats, _, _, _, _, _, _ = layer(feats) 
            out.append(feats)
        
        return out, edge_index, edge_attr


class SpatialGNNReID(nn.Module):
    def __init__(self, dev, params: dict = None, embed_dim: int = 2048):
        super(SpatialGNNReID, self).__init__()
        num_classes = params['classifier']['num_classes']
        self.dev = dev
        self.params = params
        self.gnn_params = params['gnn']

        self.gnn_model = self._build_GNN_Net(embed_dim=embed_dim, red=params['red_before'])
        embed_dim = int(embed_dim/params['red_before']
        )
        print("Using avg pool")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #print("Using max pool")
        #self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        # if apply loss function to every layer of GNN
        every = self.params['every']

        # reduce output dimension of GNN
        range_int = self.gnn_params['num_layers'] if every else 1
        self.dim_red = [nn.Linear(embed_dim, int(embed_dim/params['red'])) \
            for _ in range_int] if params['red'] != 1 else None

        # classifier
        self.neck = params['classifier']['neck']

        # if concatenate output of all layers
        dim = self.gnn_params['num_layers'] * int(embed_dim/params['red']) \
            if self.params['cat'] else int(embed_dim/params['red'])

        if self.neck:
            # BN layers
            layers = [nn.BatchNorm1d(dim) for _ in range(self.gnn_params[\
                'num_layers'])] if every else [nn.BatchNorm1d(dim)]
            self.bottleneck = Sequential(*layers)
            for layer in self.bottleneck:
                layer.bias.requires_grad_(False)
                layer.apply(weights_init_kaiming)
            
            # fc layers
            layers = [nn.Linear(dim, num_classes, bias=False) for _ in range(self.gnn_params[\
                'num_layers'])] if every else [nn.Linear(dim, num_classes, bias=False)]
            self.fc = Sequential(*layers)
            for layer in self.fc:
                layer.apply(weights_init_classifier)
        else:
            # fc layers
            layers = [nn.Linear(dim, num_classes) for _ in range(self.gnn_params[\
                'num_layers'])] if every else [nn.Linear(dim, num_classes)]
            self.fc = Sequential(*layers)

    def _build_GNN_Net(self, embed_dim: int = 2048, red=4):
        # init aggregator
        if self.gnn_params['aggregator'] == "add":
            self.aggr = lambda out, row, dim, x_size: scatter_add(out, row,
                                                                  dim=dim,
                                                                  dim_size=x_size)
        if self.gnn_params['aggregator'] == "mean":
            self.aggr = lambda out, row, dim, x_size: scatter_mean(out,
                                                                   row,
                                                                   dim=dim,
                                                                   dim_size=x_size)
        if self.gnn_params['aggregator'] == "max":
            self.aggr = lambda out, row, dim, x_size: scatter_max(out, row,
                                                                  dim=dim,
                                                                  dim_size=x_size)
        
        return GNNNetwork(embed_dim, self.aggr, self.dev,
                                self.gnn_params, self.gnn_params['num_layers'], red=red)

    def forward(self, feats, edge_index, edge_attr=None, Y=None, output_option='norm', mode='test'):

        feats, _, _ = self.gnn_model(feats, edge_index, edge_attr)
        
        if not self.params['every']:
            fc7 = [torch.flatten(self.avgpool(feats[-1]), 1)]       
        else:
            fc7 = [torch.flatten(self.avgpool(f), 1) for f in feats]

        if self.dim_red is not None:
            fc7 = [layer(f) for layer, f in zip(self.dim_red, fc7)]
        #print(feats, feats[0].requires_grad)
        
        if self.neck:
            features = [layer(f) for layer, f in zip(self.bottleneck, fc7)]
        else:
            features = [feats] 

        x = [layer(f) for layer, f in zip(self.fc7, features)]

        if output_option == 'norm':
            return x, features
        elif output_option == 'plain':
            return x, [F.normalize(f, p=2, dim=1) for f in features]
        elif output_option == 'neck' and self.neck:
            return x, features
        elif output_option == 'neck' and not self.neck:
            print("Output option neck only avaiable if bottleneck (neck) is "
                  "enabeled - giving back x and fc7")
            return x, features

        return x, feats
