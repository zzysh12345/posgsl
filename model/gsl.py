from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.utils import cumsum, scatter, coalesce, to_dense_batch, unbatch_edge_index
# from torch_sparse import coalesce
from opengsl.module.encoder import MLPEncoder
from opengsl.module.metric import WeightedCosine
from opengsl.module.transform import EpsilonNN, RemoveSelfLoop, KNN
from opengsl.module.fuse import Interpolate
from opengsl.module.encoder import GNNEncoder
from .utils import dense_to_sparse


class BasicSubGraphLearner(nn.Module):
    def __init__(self, n_feat, conf_encoder, n_hidden=None, metric_type='weighted_cosine', conf_metric=None,
                 epsilon=0.5, lamb=0.5, gsl_one_by_one=False, **kwargs):
        super(BasicSubGraphLearner, self).__init__()
        self.n_feat = n_feat
        self.conf_encoder = conf_encoder
        self.encoder_type = conf_encoder['type']
        self.encoder = None
        if self.encoder_type == 'gnn':
            self.encoder = GNNEncoder(n_feat, **conf_encoder)
        elif self.encoder_type == 'mlp':
            self.encoder = MLPEncoder(n_feat, **conf_encoder)
        self.metric_type = metric_type
        self.n_hidden = conf_encoder['n_hidden'] if self.encoder_type else n_hidden
        self.conf_metric = conf_metric
        if self.metric_type == 'weighted_cosine':
            self.metric = WeightedCosine(d_in=self.n_hidden, **conf_metric)
        else:
            raise NotImplementedError
        self.postprocess = [EpsilonNN(epsilon=epsilon), RemoveSelfLoop()]   # 注意如果不one by one，不能使用knn
        self.lamb = lamb
        self.gsl_one_by_one = gsl_one_by_one

    def reset_parameters(self):
        if self.encoder is not None:
            self.encoder.reset_parameters()
        self.metric.reset_parameters()

    def forward(self, x, selected_edge_index, selected_batch, selected_mapping, selected_belong, selected_score,
                full_edge_index=None, recover_full_adj=False, raw_edge_index=None, **kwargs):
        raw_edge_attr = x.new_ones((raw_edge_index.shape[1], ))

        if self.encoder_type:
            z_node = self.encoder(x, edge_index=selected_edge_index, batch=selected_batch, return_before_pool=True)
            if isinstance(z_node, tuple):
                z_node = z_node[1]
        else:
            z_node = x
        # normalize score
        # score_sum = scatter(selected_score, selected_belong)
        # score_sum = score_sum[selected_belong]
        # selected_score = selected_score / score_sum

        if self.gsl_one_by_one:
            # 很慢，不会用
            c = 0
            edge_index_out = selected_edge_index.new_tensor([])
            edge_index_out_subs = selected_edge_index.new_tensor([])
            edge_attr_out = x.new_tensor([])
            edge_attr_out_subs = x.new_tensor([])
            n_nodes_sub = scatter(selected_batch.new_ones(selected_batch.shape[0]), selected_batch)
            for k in range(selected_batch.max()+1):
                z_node_sub = z_node[c:c+n_nodes_sub[k]]
                adj_sub = self.metric(z_node_sub)
                for p in self.postprocess:
                    adj_sub = p(adj_sub)
                edge_index, edge_attr = dense_to_sparse(adj_sub)
                edge_index_out_subs = torch.cat([edge_index_out_subs, edge_index+c], dim=1)   # 注意这里要加上累计节点数
                edge_attr_out_subs = torch.cat([edge_attr_out_subs, edge_attr])
                edge_attr = edge_attr * selected_score[k] * self.lamb
                edge_index = selected_mapping[edge_index+c]   # 注意这里要加上累计节点数
                c += n_nodes_sub[k]
                edge_index_out = torch.cat([edge_index_out, edge_index], dim=1)
                edge_attr_out = torch.cat([edge_attr_out, edge_attr])
            # fuse with raw adj
            edge_index_out = torch.cat([edge_index_out, raw_edge_index], dim=1)
            edge_attr_out = torch.cat([edge_attr_out, raw_edge_attr * (1 - self.lamb)])
            edge_index_out, edge_attr_out = coalesce(edge_index_out, edge_attr_out)
        else:
            adj_all = self.metric(z_node)
            mask = adj_all.new_zeros(adj_all.size())
            row, col = full_edge_index
            mask[row, col] = 1
            adj_all = adj_all * mask
            for p in self.postprocess:
                adj_all = p(adj_all)
            edge_index_out_subs, edge_attr_out_subs = dense_to_sparse(adj_all)
            if not recover_full_adj:
                return edge_index_out_subs, edge_attr_out_subs
            else:
                # ********** importance score相关 **********
                # weights = selected_score[selected_batch[edge_index_out_subs[0]]]
                # edge_attr_out = edge_attr_out_subs * weights * self.lamb1
                edge_attr_out = edge_attr_out_subs * self.lamb
                edge_index_out = selected_mapping[edge_index_out_subs]
                # fuse with raw adj
                edge_index_out = torch.cat([edge_index_out, raw_edge_index], dim=1)
                edge_attr_out = torch.cat([edge_attr_out, raw_edge_attr * (1 - self.lamb)])
                # coalesce
                edge_index_out, edge_attr_out = coalesce(edge_index_out, edge_attr_out)
            return edge_index_out_subs, edge_attr_out_subs, edge_index_out, edge_attr_out


class PadSubGraphLearner(nn.Module):
    def __init__(self, n_feat, conf_encoder, n_hidden=None, metric_type='weighted_cosine', conf_metric=None,
                 epsilon=0.5, lamb=0.5, **kwargs):
        super(PadSubGraphLearner, self).__init__()
        self.n_feat = n_feat
        self.conf_encoder = conf_encoder
        self.encoder_type = conf_encoder['type']
        self.encoder = None
        if self.encoder_type == 'gnn':
            self.encoder = GNNEncoder(n_feat, **conf_encoder)
        elif self.encoder_type == 'mlp':
            self.encoder = MLPEncoder(n_feat, **conf_encoder)
        self.metric_type = metric_type
        self.n_hidden = conf_encoder['n_hidden'] if self.encoder_type else n_hidden
        self.conf_metric = conf_metric
        if self.metric_type == 'weighted_cosine':
            self.metric = WeightedCosine(d_in=self.n_hidden, **conf_metric)
        else:
            raise NotImplementedError
        self.postprocess = [EpsilonNN(epsilon=epsilon), RemoveSelfLoop()]
        self.lamb = lamb

    def reset_parameters(self):
        if self.encoder is not None:
            self.encoder.reset_parameters()
        self.metric.reset_parameters()

    def forward(self, x, selected_edge_index, selected_batch, selected_mapping=None, selected_belong=None, selected_score=None,
                full_edge_index=None, raw_edge_index=None, recover_full_adj=False, **kwargs):

        raw_edge_attr = x.new_ones((raw_edge_index.shape[1],))

        # encode
        if self.encoder_type:
            z_node = self.encoder(x, edge_index=selected_edge_index, batch=selected_batch, return_before_pool=True)
            if isinstance(z_node, tuple):
                z_node = z_node[1]
        else:
            z_node = x

        # ********** importance score相关 **********
        # score_sum = scatter(selected_score, selected_belong)
        # score_sum = score_sum[selected_belong]
        # selected_score = selected_score / score_sum
        # ********** importance score相关 **********

        # generate graph using pad
        z_subgraph_pad, mask = to_dense_batch(z_node, batch=selected_batch)  # (n_sub, n_max_node, n_hidden), (n_sub, n_max_node)
        adj_all = self.metric(z_subgraph_pad)   # (n_sub, n_max_node, n_max_node)
        adj_all_mask = mask.unsqueeze(2).float() @ mask.unsqueeze(1).float()   # (n_sub, n_max_node, n_max_node)
        adj_all = adj_all * adj_all_mask

        # postprocess
        for p in self.postprocess:
            adj_all = p(adj_all)

        # get sparse adj
        edge_index_out_subs, edge_attr_out_subs = dense_to_sparse(adj_all, mask=mask)

        if recover_full_adj:
            # ********** importance score相关 **********
            # weights = selected_score[selected_batch[edge_index_out_subs[0]]]
            # edge_attr_out = edge_attr_out_subs * weights * self.lamb1
            # ********** importance score相关 **********
            edge_attr_out = edge_attr_out_subs * self.lamb
            edge_index_out = selected_mapping[edge_index_out_subs]
            edge_index_out = torch.cat([edge_index_out, raw_edge_index], dim=1)
            edge_attr_out = torch.cat([edge_attr_out, raw_edge_attr * (1 - self.lamb)])
            edge_index_out, edge_attr_out = coalesce(edge_index_out, edge_attr_out)
            return edge_index_out_subs, edge_attr_out_subs, edge_index_out, edge_attr_out
        else:
            return edge_index_out_subs, edge_attr_out_subs


class BasicGraphLearner(nn.Module):
    def __init__(self, n_feat, conf_encoder, n_hidden=None, metric_type='weighted_cosine', conf_metric=None,
                 epsilon=0.5, lamb1=0.5, gsl_one_by_one=False, **kwargs):
        super(BasicGraphLearner, self).__init__()
        self.n_feat = n_feat
        self.conf_encoder = conf_encoder
        self.encoder_type = conf_encoder['type']
        self.encoder = None
        if self.encoder_type == 'gnn':
            self.encoder = GNNEncoder(n_feat, **conf_encoder)
        elif self.encoder_type == 'mlp':
            self.encoder = MLPEncoder(n_feat, **conf_encoder)
        self.metric_type = metric_type
        self.n_hidden = conf_encoder['n_hidden'] if self.encoder_type else n_hidden
        self.conf_metric = conf_metric
        if self.metric_type == 'weighted_cosine':
            self.metric = WeightedCosine(d_in=self.n_hidden, **conf_metric)
        else:
            raise NotImplementedError
        self.postprocess = [EpsilonNN(epsilon=epsilon), RemoveSelfLoop()]   # 注意如果不one by one，不能使用knn
        self.fuse = Interpolate(lamb1=lamb1)
        self.gsl_one_by_one = gsl_one_by_one

    def reset_parameters(self):
        if self.encoder is not None:
            self.encoder.reset_parameters()
        self.metric.reset_parameters()

    def forward(self, x, edge_index, batch, **kwargs):
        # encode
        if self.encoder_type:
            z_node = self.encoder(x, edge_index=edge_index, batch=batch, return_before_pool=True)
            if isinstance(z_node, tuple):
                z_node = z_node[1]
        else:
            z_node = x
        # generage graph
        if self.gsl_one_by_one:
            c = 0
            edge_index_out = edge_index.new_tensor([])
            edge_attr_out = x.new_tensor([])
            n_nodes = scatter(batch.new_ones(batch.shape[0]), batch)
            for k in range(batch.max() + 1):
                z_node_g = z_node[c:c+n_nodes[k]]
                adj_g = self.metric(z_node_g)
                for p in self.postprocess:
                    adj_g = p(adj_g)
                edge_index_g, edge_attr_g = dense_to_sparse(adj_g)
                edge_index_out = torch.cat([edge_index_out, edge_index_g+c], dim=1)   # 注意这里要加上累计节点数
                edge_attr_out = torch.cat([edge_attr_out, edge_attr_g])
                c += n_nodes[k]
        else:
            full_edge_index = self.prepare_full_edge_index(batch)
            adj_all = self.metric(z_node)
            mask = adj_all.new_zeros(adj_all.size())
            row, col = full_edge_index
            mask[row, col] = 1
            adj_all = adj_all * mask
            for p in self.postprocess:
                adj_all = p(adj_all)
            edge_index_out, edge_attr_out = dense_to_sparse(adj_all)
        return edge_index_out, edge_attr_out

    @ torch.no_grad()
    def prepare_full_edge_index(self, batch):
        full_edge_index = batch.new_tensor([])
        n_nodes = scatter(batch.new_ones(batch.shape[0]), batch)
        cum_nodes = cumsum(n_nodes)
        for i in range(len(n_nodes)):
            tmp = torch.arange(cum_nodes[i], cum_nodes[i+1], device=batch.device)
            row, col = torch.meshgrid(tmp, tmp)
            full_edge_index = torch.cat([full_edge_index, torch.stack([row.flatten(), col.flatten()])], dim=1)
        return full_edge_index


class MotifVector(nn.Module):
    def __init__(self, n_hidden, n_motif_per_class, n_class, sim_type='euclidean_1', temperature=0.2,
                 update_type='topk', k2=1, tau=0.99, weighted_mean=False, device=torch.device('cuda'), hem=False, **kwargs):
        super(MotifVector, self).__init__()
        self.n_hidden = n_hidden
        self.n_motif_per_class = n_motif_per_class
        self.n_class = n_class
        self.n_motif = n_motif_per_class * n_class
        # self.Motif_Vector = nn.Parameter(torch.randn(n_motif_per_class * n_class, n_hidden))
        self.Motif_Vector = torch.randn(n_motif_per_class * n_class, n_hidden, device=device)
        self.sim_type = sim_type
        if self.sim_type == 'euclidean_1':
            self.g_sim = self.euclidean_1
        elif self.sim_type == 'euclidean_2':
            self.g_sim = self.euclidean_2
        elif self.sim_type == 'cosine':
            self.g_sim = self.cosine
        else:
            raise NotImplementedError
        self.temperature = temperature
        # mapping
        self.mapping = torch.zeros((self.n_motif, n_class), device=device)
        for j in range(self.n_motif):
            self.mapping[j, j // n_motif_per_class] = 1

        self.update_type = update_type
        self.k2 = k2
        self.tau = tau
        self.weighted_mean = weighted_mean
        self.hem = hem

    def reset_parameters(self):
        init.normal_(self.Motif_Vector)

    @ staticmethod
    def euclidean_1(X, M, epsilon=1e-4):
        # ProtGNN
        xp = X @ M.t()
        distance = -2 * xp + torch.sum(X ** 2, dim=1, keepdim=True) + torch.t(torch.sum(M ** 2, dim=1, keepdim=True))
        similarity = torch.log((distance + 1) / (distance + epsilon))
        return similarity, distance

    @staticmethod
    def euclidean_2(X, M):
        # TopExpert
        distance = torch.sum(torch.pow(X.unsqueeze(1) - M, 2), 2)
        similarity = 1.0 / (1.0 + distance)
        return similarity, distance

    @staticmethod
    def euclidean_3(X, M):
        distance = torch.sum(torch.pow(X.unsqueeze(1) - M, 2), 2)
        similarity = -distance
        return similarity, distance

    @ staticmethod
    def cosine(X, M):
        similarity = F.normalize(X, p=2, dim=-1) @ F.normalize(M, p=2, dim=-1).T
        distance = -1 * similarity
        return similarity, distance

    def loss_contrastive(self, z, y):
        # 用于引导子图接近所属类别的某个motif
        y = y.squeeze()
        device = z.device
        similarity, distance = self.g_sim(z, self.Motif_Vector)
        true_motifs = torch.t(self.mapping[:, y].bool())   # (n_sub, n_motif)
        similarities = torch.exp(similarity / self.temperature)
        # sim_pos = torch.max(similarities[true_motifs].reshape(-1, self.n_motif_per_class), dim=1)[0]
        sim_pos = torch.sum(similarities[true_motifs].reshape(-1, self.n_motif_per_class), dim=1)
        sim_neg = similarities[~true_motifs].reshape(-1, (self.n_class - 1) * self.n_motif_per_class)
        # hem
        if self.hem:
            sim_neg, _ = torch.sort(sim_neg, descending=True, dim=1)
            half_len = sim_neg.shape[1] // 2
            sim_neg = sim_neg[:, :half_len]
        loss = -torch.log(sim_pos / (sim_neg.sum(1) + sim_pos)).mean()
        return loss

    @ torch.no_grad()
    def update_motif(self, z, y):
        if self.update_type == 'kmeans':
            device = z.device
            new_M = torch.tensor([], device=device)
            z = z.detach().cpu().numpy()
            y = y.cpu().squeeze().numpy()
            for i in range(self.n_class):
                z_i = z[y == i]
                old_M = self.Motif_Vector[self.n_motif_per_class*i:self.n_motif_per_class*(i+1)].detach().cpu().numpy()
                if z_i.shape[0] < self.n_motif_per_class:
                    new_M = torch.cat([new_M, torch.tensor(old_M, device=device)], dim=0)
                else:
                    kmeans = KMeans(n_clusters=self.n_motif_per_class, random_state=0, init=old_M, n_init='auto').fit(z_i)
                    centroids = kmeans.cluster_centers_
                    new_M = torch.cat([new_M, torch.tensor(centroids, device=device)], dim=0)
            new_M = (1-self.tau) * new_M + self.tau * self.Motif_Vector
            self.Motif_Vector.data = new_M.detach().clone()
        elif self.update_type == 'kmeans_all':
            device = z.device
            z = z.detach().cpu().numpy()
            y = y.cpu().numpy()
            kmeans = KMeans(n_clusters=self.n_motif, random_state=0, init=self.Motif_Vector.detach().cpu().numpy(), n_init='auto').fit(z)
            centroids = kmeans.cluster_centers_
            new_M = torch.tensor(centroids, device=device)
            new_M = (1 - self.tau) * new_M + self.tau * self.Motif_Vector
            self.Motif_Vector.data = new_M.detach().clone()
        elif self.update_type == 'kmeans_noinit':
            device = z.device
            new_M = torch.tensor([], device=device)
            z = z.detach().cpu().numpy()
            y = y.cpu().numpy()
            for i in range(self.n_class):
                z_i = z[y == i]
                old_M = self.Motif_Vector[self.n_motif_per_class*i:self.n_motif_per_class*(i+1)].detach().cpu().numpy()
                if z_i.shape[0] < self.n_motif_per_class:
                    new_M = torch.cat([new_M, torch.tensor(old_M, device=device)], dim=0)
                else:
                    kmeans = KMeans(n_clusters=self.n_motif_per_class, random_state=0, n_init='auto').fit(z_i)
                    centroids = kmeans.cluster_centers_
                    new_M = torch.cat([new_M, torch.tensor(centroids, device=device)], dim=0)
            new_M = (1-self.tau) * new_M + self.tau * self.Motif_Vector
            self.Motif_Vector.data = new_M.detach().clone()
        else:
            similarity, distance = self.g_sim(z, self.Motif_Vector)
            index_select, sim_select = self.filter_similar_graph(distance, y)
            new_M = torch.tensor([], device=z.device)
            for j in range(self.n_motif):
                new_motif = z[index_select[j]]
                weights = torch.ones(new_motif.shape[0], device=z.device)
                if self.weighted_mean:
                    weights = weights * F.softmax(sim_select[j], dim=-1)
                else:
                    weights = weights / new_motif.shape[0]
                new_motif = weights.unsqueeze(0) @ new_motif
                updated_motif = self.tau * self.Motif_Vector[j].detach().clone() + (1 - self.tau) * new_motif
                new_M = torch.cat([new_M, updated_motif], dim=0)
            self.Motif_Vector.data = new_M.detach().clone()

    def filter_similar_graph(self, similarity, y):
        device = y.device
        true_motifs = self.mapping[:, y].bool().T
        similarity = similarity * true_motifs   # (n_graph, n_motif)
        index_select = []
        sim_select = []
        for j in range(self.n_motif):
            similarity_j = similarity[:, j]
            index = torch.where(similarity_j != 0)[0]
            if self.update_type == 'topk':
                rank = torch.argsort(similarity_j[index], descending=True)
                if self.k2 >= 1:
                    n_select = self.k2
                else:
                    n_select = int(len(rank) * self.k2)
                n_select = min(n_select, len(rank))
                index_select_j = index[rank[:n_select]]
            elif self.update_type == 'thresh':
                index_select_j = index[similarity_j[index] > self.k2]
            elif self.update_type == 'all':
                index_select_j = index
            else:
                raise NotImplementedError
            index_select.append(index_select_j)
            sim_select.append(similarity_j[index_select_j])
        return index_select, sim_select

    @torch.no_grad()
    def init_motif(self, z, y):
        device = z.device
        new_M = torch.tensor([], device=device)
        z = z.detach().cpu().numpy()
        y = y.cpu().squeeze().numpy()
        for i in range(self.n_class):
            z_i = z[y == i]
            kmeans = KMeans(n_clusters=self.n_motif_per_class, random_state=0, n_init='auto').fit(z_i)
            centroids = kmeans.cluster_centers_
            new_M = torch.cat([new_M, torch.tensor(centroids, device=device)], dim=0)
        self.Motif_Vector.data = new_M.detach().clone()

    def forward(self):
        pass