import os.path as osp

from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T

from utils_lp import do_edge_split_direct as do_edge_split, load_social_graphs, do_edge_split_social_direct
from ogb.linkproppred import PygLinkPropPredDataset


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'ogbl-ddi', 'ogbl-collab', 'ogbl-ppa', 'BlogCatalog', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'Flickr']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    if name.startswith('ogbl'):
        dataset = PygLinkPropPredDataset(root=osp.join(root_path, 'OGB'), name=name)

        return dataset, dataset.get_edge_split()

    if name in ['BlogCatalog', 'Flickr']:
        data = load_social_graphs(name)
        split_edge = do_edge_split_social_direct(data)
        return [data], split_edge

    else:
        dataset = Planetoid(osp.join(root_path, 'Citation'), name)
        return dataset, do_edge_split(dataset)


def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return base_path
    else:
        return osp.join(base_path, name)
