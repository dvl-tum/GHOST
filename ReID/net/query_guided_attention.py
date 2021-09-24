
import torch
from torch import nn
from torch.nn import functional as F
from time import time
from torch_scatter import scatter_add
from .utils import *

class _Query_Guided_Attention(nn.Module):
    def __init__(self, in_channels, inter_channels=None, norm='size', message='masked_image'): #'attention_map', 'masked_image', with_residual''
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


    def forward(self, x, x_g, gs=None, attention="x"):
        '''
        :param x = new detections
        :param x_g = tracklets (guidance)
        :param attention = x: update feature map of x
        :param attention = x_g: update feature map of guidance 
        :return:
        '''

        batch_size = x.size(0)
        theta_x = self.theta(x)
        phi_x = self.phi(x_g)

        if attention == "x":
            theta_x = theta_x.view(batch_size, self.inter_channels, -1).permute(0, 2, 1) #x
            # phi_x = phi_x.view(batch_size, self.inter_channels, -1) #x_g
            phi_x = self.max_pool_layer(phi_x).view(batch_size, self.inter_channels, -1) #x_g
            f = torch.matmul(theta_x, phi_x)
            if self.norm == 'size':
                N = f.size(-1)
                f = f / N
            if self.norm == 'channels':
                N = phi_x.shape[1]
                f = f / N
            elif self.norm == 'sm':
                f = self.sm(f)
            f, max_index = torch.max(f, 2)
            f = f.view(batch_size, *x.size()[2:]).unsqueeze(1)
            #torch.save(f.cpu(), 'attention_maps.pt')
            #quit()
            
            #torch.save(f, 'attention_maps.pt')
            #quit()
            
            if self.message == 'attention_map':
                return f
            z = x * f # mask image
            if self.message == 'masked_image':
                return z
            elif self.message == 'with_residual':
                z = z + x
                return z

        elif attention == "x_g":
            phi_x = phi_x.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
            theta_x = theta_x.view(batch_size, self.inter_channels, -1)
            f = torch.matmul(phi_x, theta_x)
            if self.norm == 'size':
                N = f.size(-1)
                f = f / N
            elif self.norm == 'sm':
                f = self.sm(f)
            f, max_index = torch.max(f, 2)
            f = f.view(batch_size, *x_g.size()[2:]).unsqueeze(1)

            z = x_g * f
            if self.final_res:
                z = z + x_g
            return z

class Query_Guided_Attention_Layer(nn.Module):
    def __init__(self, in_channels, inter_channels=None, aggr=None, gnn_params=None, postprocessing=True, num_classes=751):
        super(Query_Guided_Attention_Layer, self).__init__()
        self.query_guided_attention = _Query_Guided_Attention(in_channels=in_channels, 
                inter_channels=inter_channels)
        self.postprocessing = postprocessing
        print("Using postprocessing {}".format(postprocessing))
        self.res1 = gnn_params['res1']
        self.res2 = gnn_params['res2']
        self.aggr = aggr
        if self.aggr is None:
            self.aggr = lambda out, row, dim, x_size: scatter_add(out,
                                                                   row,
                                                                   dim=dim,
                                                                   dim_size=x_size)
        self.dropout1 = nn.Dropout(gnn_params['dropout_1'])
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) if gnn_params['mlp'] else None #kernel_size 3 or 1? if 3: padding=1 
        self.act = F.relu
        self.dropout = nn.Dropout(gnn_params['dropout_mlp'])
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) if gnn_params['mlp'] else None

        self.norm = nn.LayerNorm([in_channels, 12, 4]) if gnn_params['norm1'] else None
        self.dropout2 = nn.Dropout(gnn_params['dropout_2'])
        self.norm2 = nn.LayerNorm([in_channels, 12, 4]) if gnn_params['norm2'] else None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Sequential(
            nn.Linear(in_channels, int(in_channels/2)),
            nn.ReLU(),
            nn.Linear(int(in_channels/2), 1))
        self.sig = nn.Sigmoid()

        self.fc = nn.Linear(in_channels, num_classes)


    def forward(self, x, num_gallery=None, aggregate=True, classify=False):
        '''
        if not num_gallery: # x of form [query, gallery, query, gallery, ...]
            num_gallery = num_query = int(x.shape[0]/2)
            query_mask = torch.tensor([i%2==0 for i in range(x.shape[0])])
            gallery_mask = torch.tensor([i%2!=0 for i in range(x.shape[0])])
        else: # x of form [gallery, gallery, query, query]
            num_query = x.shape[0] - num_gallery
            query_mask = torch.tensor([i>=num_gallery for i in range(x.shape[0])])
            gallery_mask = torch.tensor([i<num_gallery for i in range(x.shape[0])])

        qs = torch.cat([torch.where(query_mask)[0].repeat(num_gallery), torch.where(query_mask)[0]]).cuda(x.get_device())
        gs = torch.cat([torch.where(gallery_mask)[0].unsqueeze(dim=1).repeat(1, num_query).flatten(), torch.where(query_mask)[0]]).cuda(x.get_device())
        '''
        A = torch.where(torch.ones((x.shape[0], x.shape[0])) > 0)
        qs = A[0]
        gs = A[1]
        attended_gallery = self.query_guided_attention(x[gs].cuda(x.get_device()), x[qs].cuda(x.get_device()), gs) # x, x_guidance

        if aggregate:
            x2 = self.aggr(attended_gallery, gs.cuda(x.get_device()), 0, x.shape[0])
            if self.query_guided_attention.message == 'attention_map':
                x2 = x * x2  
        
            if self.postprocessing:
                # "post message processing"
                x2 = self.dropout1(x2)
                x = x2 + x if self.res1 else x2
                x = self.norm(x) if self.norm is not None else x
                x2 = self.conv2(self.dropout(self.act(self.conv1(x)))) if self.conv1 else x
                x2 = self.dropout2(x2)
                x = x + x2 if self.res2 else x2
                x = self.norm2(x) if self.norm2 is not None else x
            else:
                x = x2
            if classify:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
            return x
        else:
            if self.postprocessing:
                x = self.dropout1(x)
                x = self.norm(x) if self.norm is not None else x
                x = self.conv2(self.dropout(self.act(self.conv1(x)))) if self.conv1 else x
                x = self.dropout2(x)
                x = self.norm2(x) if self.norm2 is not None else x
            else:
                x = attended_gallery
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.final(x)
            x = self.sig(x)

            return x, qs, gs        


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
            feats = layer(feats)
            out.append(feats)
        
        return out, edge_index, edge_attr


class SpatialGNNReID(nn.Module):
    def __init__(self, dev, params: dict = None, embed_dim: int = 2048):
        super(SpatialGNNReID, self).__init__()
        num_classes = params['classifier']['num_classes']
        self.dev = dev
        self.params = params
        self.gnn_params = params['gnn']

        red = 1
        logger.info("Reduce number of channels by {} (hardcoded)".format(red))
        
        self.gnn_model = self._build_GNN_Net(embed_dim=embed_dim, red=red)
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
        r, c = edge_index[:, 0], edge_index[:, 1]

        if self.params['use_edge_encoder']:
            edge_attr = torch.cat(
                [feats[r, :], feats[c, :], edge_attr.unsqueeze(dim=1)], dim=1)
            edge_attr = self.edge_encoder(edge_attr)

        if self.params['use_node_encoder']:
            feats = self.node_encoder(feats)

        feats, _, _ = self.gnn_model(feats, edge_index, edge_attr)
        feats = feats[-1]
        fc7 = self.avgpool(feats)
        fc7 = torch.flatten(fc7, 1)

        if self.dim_red is not None:
            fc7 = self.dim_red(fc7)
        #print(feats, feats[0].requires_grad)
        
        if self.neck:
            features = list()
            for i, layer in enumerate(self.bottleneck):
                f = layer(fc7)
                features.append(f)
        else:
            features = [feats] 

        #x = self.fc(features) 

        x = list()
        for i, layer in enumerate(self.fc):
            f = layer(features[i])
            x.append(f)

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
