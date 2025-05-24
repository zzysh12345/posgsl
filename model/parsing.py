from torch_geometric.utils import to_dense_adj, subgraph, cumsum, scatter
from torch_geometric.utils import degree
from torch_geometric.data import Data, Batch
from torch_geometric.loader import ClusterData
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import os
import numpy as np


# ----------- BFS Parsing -----------
def bfs(adj, s, max_sub_size=10):
    # 从s出发BFS，限制最大节点数为sub_size，得到子图节点列表
    sub = []
    visited = [0 for _ in range(len(adj[0]))]
    queue = [s]
    visited[s] = 1
    node = queue.pop(0)
    sub.append(node)
    while True:
        for x in range(0, len(visited)):
            if adj[node][x] == 1 and visited[x] == 0:
                visited[x] = 1
                queue.append(x)
        if len(queue) == 0:
            break
        else:
            newnode = queue.pop(0)
            node = newnode
            sub.append(node)
            if len(sub) == max_sub_size:
                sub.sort()
                return sub
    sub.sort()
    return sub


def parse_bfs(g, max_n_sub=5, max_sub_size=10, **kwargs):
    # 首先计算度数并根据度数排序
    adj = to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes)[0]
    degrees = degree(g.edge_index[0], g.num_nodes)
    nodes = degrees.sort(descending=True)[1]
    tmp = {}
    subs = []
    for seed in nodes[:max_n_sub]:
        nodes_sub = bfs(adj, int(seed), max_sub_size)
        if tuple(nodes_sub) not in tmp.keys():
            # 防止完全重复的子图
            subs.append(nodes_sub)
            tmp[tuple(nodes_sub)] = 1
    return subs


def original(g, **kwargs):
    device = g.x.device
    n = g.num_nodes
    return [list(range(int(n)))]


def random_parse_bfs(g, max_n_sub=5, max_sub_size=10, **kwargs):
    adj = to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes)[0]
    degrees = degree(g.edge_index[0], g.num_nodes)
    nodes = degrees.sort(descending=True)[1]
    tmp = {}
    subs = []
    seeds = np.random.choice(nodes, size=min(max_n_sub, len(nodes)), replace=False)
    for seed in seeds:
        nodes_sub = bfs(adj, int(seed), max_sub_size)
        if tuple(nodes_sub) not in tmp.keys():
            # 防止完全重复的子图
            subs.append(nodes_sub)
            tmp[tuple(nodes_sub)] = 1
    return subs


def random_parse(g, max_n_sub=5, max_sub_size=10, **kwargs):
    adj = to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes)[0]
    n = adj.shape[0]
    nodes = torch.arange(0, n)
    tmp = {}
    subs = []
    for i in range(max_n_sub):
        nodes_sub = np.random.choice(nodes, size=min(max_sub_size, len(nodes)), replace=False)
        if tuple(nodes_sub) not in tmp.keys():
            subs.append(nodes_sub)
            tmp[tuple(nodes_sub)] = 1
    return subs


def metis_parse(g, n_sub=5, **kwargs):
    subs = []
    metis = ClusterData(g, num_parts=n_sub)
    partptr = metis.partition.partptr.unique() # 可能有极端情况，图节点数较小，分割出来会有空，需要unique
    node_perm = metis.partition.node_perm
    n_sub_real = len(partptr)-1
    for i in range(n_sub_real):
        nodes_sub = node_perm[partptr[i]:partptr[i+1]].numpy()
        subs.append(nodes_sub)
    return subs


def hash_parse(g, n_sub=5, **kwargs):
    subs = []
    n = g.num_nodes
    assert n > (n_sub * 3), 'Subgraph must have more than 3 nodes.'
    tmp = np.arange(n)
    np.random.shuffle(tmp)
    partition = tmp % n_sub
    subs = [np.where(partition==i)[0] for i in range(n_sub)]
    return subs


# ----------- Prepare Parsed Data -----------
def prepare_parsed_data(dataset, parsing='parse_bfs', parsing_params=None, require_full_edge=False):
    parsing = eval(parsing)
    graphs2subgraphs = {}
    for i in range(len(dataset)):
        graphs2subgraphs[i] = []
    for i, g in tqdm(enumerate(dataset)):
        subs_of_g = []
        subs = parsing(g, **parsing_params)
        for nodes_sub in subs:
            nodes_sub = torch.tensor(nodes_sub, dtype=torch.long)
            x_sub = g.x[nodes_sub]
            edge_sub = subgraph(nodes_sub, g.edge_index, relabel_nodes=True, num_nodes=g.num_nodes)[0]
            # mask
            if require_full_edge:
                tmp = torch.arange(nodes_sub.shape[0])
                row, col = torch.meshgrid(tmp, tmp)
                full_edge_index = torch.stack([row.flatten(), col.flatten()])
                g_sub = Data(x=x_sub, edge_index=edge_sub, y=g.y, belong=g.idx, mapping=nodes_sub, full_edge_index=full_edge_index)
            else:
                g_sub = Data(x=x_sub, edge_index=edge_sub, y=g.y, belong=g.idx, mapping=nodes_sub)
            subs_of_g.append(g_sub)
        graphs2subgraphs[i] = subs_of_g
    return graphs2subgraphs


class SimpleParsingModule(nn.Module):
    def __init__(self, parsing='parse_bfs', parsing_params=None, return_new_mapping=True, require_full_edge=False, reparsing=False, **kwargs):
        super(SimpleParsingModule, self).__init__()
        self.parsing = parsing
        self.parsing_params = parsing_params
        self.graphs2subgraphs = None
        self.return_new_mapping = return_new_mapping
        self.require_full_edge = require_full_edge
        self.reparsing = reparsing

    @ torch.no_grad()
    def forward(self, batch, only_idx=False):
        if only_idx:
            device = batch.x.device
            graphs_list = batch.to_data_list()
            batch_subgraphs_nodes_list = []
            batch_belong_list = torch.tensor([], dtype=torch.long, device=device)
            n_nodes_graph = scatter(batch.batch.new_ones(batch.batch.shape[0]), batch.batch)
            cum_n_nodes = cumsum(n_nodes_graph)
            for k, g in enumerate(graphs_list):
                subs_list = self.graphs2subgraphs[g.idx.item()]
                subs_belong = torch.full((len(subs_list),), k, dtype=torch.long, device=device)
                batch_belong_list = torch.cat([batch_belong_list, subs_belong])
                nodes_list = [tmp.mapping.to(device)+cum_n_nodes[k] for tmp in subs_list]   # TODO 这个to(device)是否有必要
                batch_subgraphs_nodes_list.extend(nodes_list)
            return batch_subgraphs_nodes_list, batch_belong_list

        else:
            device = batch.x.device
            # graphs_list = batch.to_data_list()
            graphs_list_idx = batch.idx
            batch_subgraphs_list = []
            batch_belong_list = torch.tensor([], dtype=torch.long, device=device)
            for k, t in enumerate(graphs_list_idx):
                subs_list = self.graphs2subgraphs[int(t)]
                subs_belong = torch.full((len(subs_list),), k, dtype=torch.long, device=device)
                batch_subgraphs_list.extend(subs_list)
                batch_belong_list = torch.cat([batch_belong_list, subs_belong])
            batch_subs = Batch.from_data_list(batch_subgraphs_list).to(device)   # TODO 这个to(device)是否有必要
            if self.return_new_mapping:
                n_nodes_graph = scatter(batch.batch.new_ones(batch.batch.shape[0]), batch.batch)
                cum_n_nodes = cumsum(n_nodes_graph)
                new_mapping = batch_subs.mapping + cum_n_nodes[batch_belong_list[batch_subs.batch]]
            else:
                new_mapping = None
            return batch_subs, batch_belong_list, new_mapping

    @torch.no_grad()
    def init_parsing(self, dataset):
        if not hasattr(dataset, 'attack_type'):
            path = os.path.join(dataset.path, dataset.name, '{}_{}_{}.pickle'.format(dataset.name, self.parsing , '_'.join(list(map(str, list(self.parsing_params.values()))))))
        else:
            path = os.path.join(dataset.path, dataset.name, '{}_{}_{}_{}.pickle'.format(dataset.name, dataset.attack_type, self.parsing, '_'.join(list(map(str, list(self.parsing_params.values()))))))
        if self.reparsing or (not os.path.exists(path)):
            self.graphs2subgraphs = prepare_parsed_data(dataset.data_raw, self.parsing, self.parsing_params, self.require_full_edge)
            with open(path, 'wb') as file:
                pickle.dump(self.graphs2subgraphs, file)
        else:
            with open(path, 'rb') as file:
                self.graphs2subgraphs = pickle.load(file)






