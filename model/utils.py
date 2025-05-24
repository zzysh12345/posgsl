from typing import Optional, Tuple

from torch import Tensor
import warnings
import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import scatter, segment, cumsum
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import segment_csr
import torch


def softmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
) -> Tensor:
    # 支持deterministic的softmax
    N = maybe_num_nodes(index, num_nodes)
    ones = torch.ones_like(index)
    ptr = cumsum(scatter(ones, index))
    # src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
    src_max = segment_csr(src.detach(), ptr, reduce='max')
    out = src - src_max.index_select(dim, index)
    out = out.exp()
    # out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
    out_sum = segment_csr(out, ptr, reduce='sum') + 1e-16
    out_sum = out_sum.index_select(dim, index)
    return out / out_sum


def dense_to_sparse(
    adj: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    # 支持梯度回传的dense_to_sparse
    if adj.dim() < 2 or adj.dim() > 3:
        raise ValueError(f"Dense adjacency matrix 'adj' must be two- or "
                         f"three-dimensional (got {adj.dim()} dimensions)")

    if mask is not None and adj.dim() == 2:
        warnings.warn("Mask should not be provided in case the dense "
                      "adjacency matrix is two-dimensional")
        mask = None

    if mask is not None and mask.dim() != 2:
        raise ValueError(f"Mask must be two-dimensional "
                         f"(got {mask.dim()} dimensions)")

    if mask is not None and adj.size(-2) != adj.size(-1):
        raise ValueError(f"Mask is only supported on quadratic adjacency "
                         f"matrices (got [*, {adj.size(-2)}, {adj.size(-1)}])")

    if adj.dim() == 2:
        edge_index = adj.nonzero().t()
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        flatten_adj = adj.view(-1, adj.size(-1))
        if mask is not None:
            flatten_adj = flatten_adj[mask.view(-1)]
        edge_index = flatten_adj.nonzero().t()
        edge_attr = flatten_adj[edge_index[0], edge_index[1]]

        if mask is None:
            offset = torch.arange(
                start=0,
                end=adj.size(0) * adj.size(2),
                step=adj.size(2),
                device=adj.device,
            )
            offset = offset.repeat_interleave(adj.size(1))
        else:
            count = mask.sum(dim=-1)
            offset = cumsum(count)[:-1]
            offset = offset.repeat_interleave(count)

        new_edge_index = edge_index.clone()
        new_edge_index[1] += offset[edge_index[0]]

        return new_edge_index, edge_attr