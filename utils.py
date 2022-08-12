# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import sys
import math
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)
import scipy.io as scio


def train_test_split_edges_direct(data, val_ratio: float = 0.05,
                           test_ratio: float = 0.1):
    r"""Splits the edges of a :class:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges.
    As such, it will replace the :obj:`edge_index` attribute with
    :obj:`train_pos_edge_index`, :obj:`train_pos_neg_adj_mask`,
    :obj:`val_pos_edge_index`, :obj:`val_neg_edge_index` and
    :obj:`test_pos_edge_index` attributes.
    If :obj:`data` has edge features named :obj:`edge_attr`, then
    :obj:`train_pos_edge_attr`, :obj:`val_pos_edge_attr` and
    :obj:`test_pos_edge_attr` will be added as well.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation edges.
            (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test edges.
            (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    data.edge_index = data.edge_attr = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    # if edge_attr is not None:
    #     out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
    #     data.train_pos_edge_index, data.train_pos_edge_attr = out
    # else:
    #     data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0, 
                   max_nodes_per_hop=None, node_features=None, 
                   y=1, directed=False, A_csc=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More 
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl'):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]
    
    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists)==0).to(torch.long)
    elif node_label == 'de':  # distance encoding
        z = de_node_labeling(adj, 0, 1)
    elif node_label == 'de+':
        z = de_plus_node_labeling(adj, 0, 1)
    elif node_label == 'degree':  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z>100] = 100  # limit the maximum label to 100
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z, 
                node_id=node_ids, num_nodes=num_nodes)
    return data

 
def extract_enclosing_subgraphs(link_index, A, x, y, num_hops, node_label='drnl', 
                                ratio_per_hop=1.0, max_nodes_per_hop=None, 
                                directed=False, A_csc=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []
    for src, dst in tqdm(link_index.t().tolist()):
        tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop, 
                             max_nodes_per_hop, node_features=x, y=y, 
                             directed=directed, A_csc=A_csc)
        data = construct_pyg_graph(*tmp, node_label)
        data_list.append(data)

    return data_list


def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    data = dataset[0]
    random.seed(234)
    torch.manual_seed(234)

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


def do_edge_split_direct(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    data = dataset.clone()
    random.seed(234)
    torch.manual_seed(234)

    if not fast_split:
        data = train_test_split_edges_direct(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


def do_edge_split_nc(edge_index, num_nodes, val_ratio=0.05, test_ratio=0.1):
    random.seed(234)
    torch.manual_seed(234)

    row, col = edge_index
    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]
    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))
    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    test_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    train_pos_edge_index = torch.stack([r, c], dim=0)
    # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
    neg_edge_index = negative_sampling(
        edge_index, num_nodes=num_nodes,
        num_neg_samples=row.size(0))
    test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]

    train_pos_edge = torch.cat([train_pos_edge_index, val_pos_edge_index], dim=1)

    return train_pos_edge.t(), test_pos_edge_index.t(), test_neg_edge_index.t()


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target), 
                                target_neg.view(-1)])
    return pos_edge, neg_edge


def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index


def AA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


def load_social_graphs(dataset_str="BlogCatalog"):
    data_file = 'dataset/{}/{}'.format(dataset_str, dataset_str) + '.mat'
    data = scio.loadmat(data_file)
    features = data['Attributes']
    labels = data['Label'].reshape(-1)
    adj = data['Network']

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1

    adj = adj - ssp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    adj = adj.tocoo()
    edge_index = np.vstack((adj.row, adj.col)).transpose()

    x = torch.from_numpy(features.todense()).to(torch.float)

    edge_index = torch.from_numpy(edge_index).t().to(torch.long).contiguous()
    y = torch.from_numpy(labels).to(torch.float)

    dataset = Data(x=x, edge_index=edge_index, y=y)
    return dataset


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, f=sys.stdout):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:', file=f)
            print(f'Highest Valid: {result[:, 0].max():.2f}', file=f)
            print(f'Highest Eval Point: {argmax + 1}', file=f)
            print(f'   Final Test: {result[argmax, 1]:.2f}', file=f)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:', file=f)
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}', file=f)
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}', file=f)


def evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true):
    train_auc = roc_auc_score(train_true, train_pred)
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    train_ap = average_precision_score(train_true, train_pred)
    valid_ap = average_precision_score(val_true, val_pred)
    test_ap = average_precision_score(test_true, test_pred)
    results = dict()
    results['AUC'] = (train_auc, valid_auc, test_auc)
    results['AP'] = (train_ap, valid_ap, test_ap)
    return results


class EdgeLoader(object):
    def __init__(self, train_edges, train_edge_false, batch_size, remain_delet=True, shuffle=True):
        self.shuffle = shuffle
        self.index = 0
        self.index_false = 0
        self.pos_edge = train_edges
        self.neg_edge = train_edge_false
        self.id_index = list(range(train_edges.shape[0]))
        self.data_len = len(self.id_index)
        self.remain_delet = remain_delet
        self.batch_size = batch_size
        if self.shuffle:
            self._shuffle()

    def __iter__(self):
        return self

    def _shuffle(self):
        random.shuffle(self.id_index)

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.remain_delet:
            if self.index + self.batch_size > self.data_len:
                self.index = 0
                self.index_false = 0
                self._shuffle()
                raise StopIteration
            batch_index = self.id_index[self.index: self.index + self.batch_size]
            batch_x = self.pos_edge[batch_index]
            batch_y = self.neg_edge[batch_index]
            self.index += self.batch_size

        else:
            if self.index >= self.data_len:
                self.index = 0
                self._shuffle()
                # raise StopIteration
            end_ = min(self.index + self.batch_size, self.data_len)
            batch_index = self.id_index[self.index: end_]
            batch_x = self.pos_edge[batch_index]
            batch_y = self.neg_edge[batch_index]
            self.index += self.batch_size
        return np.array(batch_x), np.array(batch_y)


class IndexLoader(object):
    def __init__(self, num_node, batch_size, drop_last=False, shuffle=True):
        self.shuffle = shuffle
        self.index = 0
        self.index_false = 0
        self.num_node = num_node
        data = np.array(range(num_node)).reshape(-1)
        self.data = torch.from_numpy(data)
        self.id_index = list(range(num_node))
        self.data_len = len(self.id_index)
        self.drop_last = drop_last
        self.batch_size = batch_size
        if self.shuffle:
            self._shuffle()

    def __iter__(self):
        return self

    def _shuffle(self):
        random.shuffle(self.id_index)

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.drop_last:
            if self.index + self.batch_size > self.data_len:
                self.index = 0
                self.index_false = 0
                self._shuffle()
                raise StopIteration
            batch_index = self.id_index[self.index: self.index + self.batch_size]
            batch_x = self.data[batch_index]
            self.index += self.batch_size

        else:
            if self.index >= self.data_len:
                self.index = 0
                self._shuffle()
                # raise StopIteration
            end_ = min(self.index + self.batch_size, self.data_len)
            batch_index = self.id_index[self.index: end_]
            batch_x = self.data[batch_index]
            self.index += self.batch_size
        return batch_x


def sparse_to_tuple(sparse_mx):
    if not ssp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def edgemask_um(mask_ratio, split_edge, device, num_nodes):
    if isinstance(split_edge, torch.Tensor):
        edge_index = split_edge
    else:
        edge_index = split_edge['train']['edge']
    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num])
    mask_index = torch.from_numpy(index[-mask_num:])
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].t()
    edge_index = to_undirected(edge_index_train)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index).t()
    return adj, edge_index, edge_index_mask.to(device)


def edgemask_dm(mask_ratio, split_edge, device, num_nodes):
    if isinstance(split_edge, torch.Tensor):
        edge_index = to_undirected(split_edge.t()).t()
    else:
        edge_index = torch.stack([split_edge['train']['edge'][:, 1], split_edge['train']['edge'][:, 0]], dim=1)
        edge_index = torch.cat([split_edge['train']['edge'], edge_index], dim=0)

    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num])
    mask_index = torch.from_numpy(index[-mask_num:])
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].to(device)

    edge_index = edge_index_train
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index).t()
    return adj, edge_index, edge_index_mask.to(device)
