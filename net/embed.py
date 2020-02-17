import torch
import torch.nn as nn


def make_embedding_layer(in_features, sz_embedding, weight_init = None):
    embedding_layer = torch.nn.Linear(in_features, sz_embedding)
    if weight_init != None:
        weight_init(embedding_layer.weight)
    return embedding_layer


def bn_inception_weight_init(weight):
    import scipy.stats as stats
    stddev = 0.001
    X = stats.truncnorm(-10, 10, scale=stddev)
    values = torch.Tensor(
        X.rvs(weight.data.numel())
    ).resize_(weight.size())
    weight.data.copy_(values)


def embed(model, sz_embedding, normalize_output = True):
    model.embedding_layer = make_embedding_layer(
        model.last_linear.in_features,
        sz_embedding,
        weight_init = bn_inception_weight_init
    )
    self.fc = nn.Linear(embed_features_size, num_classes)
    def forward(x):
        # split up original logits and forward methods
        x = model.features(x)
        x = model.global_pool(x)
        x = x.view(x.size(0), -1)
        x = model.embedding_layer(x)
        if normalize_output == True:
            x = torch.nn.functional.normalize(x, p = 2, dim = 1)
        return x
    model.forward = forward


# class embed(nn.Module):
#     def __init__(self, model, sz_embedding, normalize_output=True):
#         super(embed, self).__init__()
#         self.normalize_output = normalize_output
#         self.embedding_layer = torch.nn.Linear(1024, sz_embedding)
#         self.model = model
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#     def forward(self, x):
#         x = self.model(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.embedding_layer(x)
#         if self.normalize_output == True:
#             x = torch.nn.functional.normalize(x, p = 2, dim = 1)
#         return x


