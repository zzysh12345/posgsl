import torch
import torch.nn as nn
# from torch_geometric.utils import softmax
from torch_geometric.nn import inits
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn.pool.connect.filter_edges import filter_adj
from opengsl.module.encoder import MLPEncoder
from opengsl.module.encoder import GNNEncoder
from .utils import softmax


class SubgraphSelectModule(nn.Module):
    def __init__(self, n_feat, topk=0.8, min_score=None, conf_encoder=None, conf_estimator=None, select_flag=True, **kwargs):
        super(SubgraphSelectModule, self).__init__()
        assert ((topk is None) ^ (min_score is None))
        self.n_feat = n_feat
        self.topk = topk
        self.select_flag = select_flag
        self.min_score = min_score
        self.conf_encoder = conf_encoder
        self.conf_estimator = conf_estimator
        if self.select_flag:
            if self.conf_encoder['type'] == 'mlp':
                self.encoder = MLPEncoder(n_feat, **conf_encoder)
            elif self.conf_encoder['type'] == 'gnn':
                self.encoder = GNNEncoder(n_feat, **conf_encoder)
            else:
                raise NotImplementedError
            if self.conf_estimator['type'] == 'mlp':
                self.estimator = MLPEncoder(n_feat=conf_encoder['n_class'], n_class=1, **conf_estimator)
            elif self.conf_estimator['type'] == 'projection':
                self.estimator = ProjectionEstimator(n_feat=conf_encoder['n_class'], **conf_estimator)
            else:
                raise NotImplementedError

    def reset_parameters(self):
        if self.select_flag:
            self.encoder.reset_parameters()
            self.estimator.reset_parameters()

    def forward(self, batch_sub, belong, new_mapping):
        if self.select_flag:
            z_sub, z_node = self.encoder(x=batch_sub.x, edge_index=batch_sub.edge_index, batch=batch_sub.batch,
                                         return_before_pool=True)
            score = self.estimator(z_sub, belong).squeeze()   # (n_sub, )
            sub_index = topk(score, self.topk, belong, self.min_score).sort()[0]   # 保证node和subgraph变量顺序一致

            sub_mask = sub_index.new_full((batch_sub.num_graphs,), -1)
            sub_mask[sub_index] = torch.arange(sub_index.shape[0], device=sub_index.device)
            batch_after_select = sub_mask[batch_sub.batch]
            node_index = torch.where(batch_after_select >= 0)[0]
            new_edge_index, _ = filter_adj(batch_sub.edge_index, node_index=node_index, num_nodes=batch_sub.num_nodes, edge_attr=None)
            new_x = batch_sub.x[node_index]
            new_batch = batch_after_select[node_index]
            new_belong = belong[sub_index]
            if new_mapping is not None:
                new_mapping = new_mapping[node_index]   # 这个mapping对应一个batch里的实际位置（加了节点数的累计
            else:
                new_mapping = batch_sub.mapping[node_index]   # 这个mapping对应原来图里的实际位置
            if 'full_edge_index' in batch_sub:
                new_full_edge_index, _ = filter_adj(batch_sub.full_edge_index, node_index=node_index, num_nodes=batch_sub.num_nodes, edge_attr=None)
            else:
                new_full_edge_index = None
            return new_x, z_node[node_index], z_sub[sub_index], new_edge_index, new_batch, new_belong, new_mapping, score[sub_index], new_full_edge_index
        else:
            return batch_sub.x, 0, 0, batch_sub.edge_index, batch_sub.batch, belong, new_mapping, torch.ones(belong.shape, device=belong.device), batch_sub.full_edge_index if 'full_edge_index' in batch_sub else None


class ProjectionEstimator(nn.Module):
    def __init__(self, n_feat, act='sigmoid', **kwargs):
        super(ProjectionEstimator, self).__init__()
        self.n_feat = n_feat
        self.projection_vector = nn.Parameter(torch.rand(n_feat,))
        self.act = activation_resolver(act)
        self.act_name = act
        self.reset_parameters()

    def reset_parameters(self):
        inits.uniform(self.n_feat, self.projection_vector)

    def forward(self, z, batch):
        score = z@self.projection_vector
        if self.act_name == 'softmax':
            # 会导致reproducibility问题
            score = softmax(score, batch)
        else:
            score = self.act(score / self.projection_vector.norm(p=2, dim=-1))
        return score
