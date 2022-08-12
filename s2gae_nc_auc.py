import argparse
import torch_geometric.transforms as T
import torch
from torch.utils.data import DataLoader
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
import time
import os
from torch_geometric.data import Data
from model import LPDecoder_ogb as LPDecoder
from model import GCN_mgaev3 as GCN
from model import SAGE_mgaev2 as SAGE
from model import GIN_mgaev2 as GIN
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score
from utils import edgemask_um, edgemask_dm, do_edge_split_nc
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import os.path as osp


def random_edge_mask(args, edge_index, device, num_nodes):
    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * args.mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num])
    mask_index = torch.from_numpy(index[-mask_num:])
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].to(device)

    edge_index_train, _ = add_self_loops(edge_index_train, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index_train).t()
    return adj, edge_index_train, edge_index_mask


def train(model, predictor, data, edge_index, optimizer, args):
    model.train()
    predictor.train()

    total_loss = total_examples = 0
    if args.mask_type == 'um':
        adj, _, pos_train_edge = edgemask_um(args.mask_ratio, edge_index, data.x.device, data.x.shape[0])
    else:
        adj, _, pos_train_edge = edgemask_dm(args.mask_ratio, edge_index, data.x.device, data.x.shape[0])
    adj = adj.to(data.x.device)

    for perm in DataLoader(range(pos_train_edge.size(0)), args.batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(data.x, adj)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h, edge)
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.x.shape[0], edge.size(), dtype=torch.long,
                             device=data.x.device)
        neg_out = predictor(h, edge)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, pos_test_edge, neg_test_edge, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.full_adj_t)

    pos_test_edge = pos_test_edge.to(data.x.device)
    neg_test_edge = neg_test_edge.to(data.x.device)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h, edge).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h, edge).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    test_true = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)
    test_auc = roc_auc_score(test_true, test_pred)
    return test_auc


def extract_feature_list_layer2(feature_list):
    xx_list = []
    xx_list.append(feature_list[-1])
    tmp_feat = torch.cat(feature_list, dim=-1)
    xx_list.append(tmp_feat)
    return xx_list


def accuracy(preds, labels):
    correct = (preds == labels).astype(float)
    correct = correct.sum()
    return correct / len(labels)


def test_classify(feature, labels):
    aucs = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(feature):
        train_X, train_y = feature[train_index], labels[train_index]
        test_X, test_y = feature[test_index], labels[test_index]
        clf = OneVsRestClassifier(LinearSVC(random_state=0))
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)

        rocauc_list = []

        for i in range(test_y.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(test_y[:, i] == 1) > 0 and np.sum(test_y[:, i] == 0) > 0:
                is_labeled = test_y[:, i] == test_y[:, i]
                rocauc_list.append(roc_auc_score(test_y[is_labeled, i], preds[is_labeled, i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        auc = sum(rocauc_list) / len(rocauc_list)

        aucs.append(auc)

    aucs = np.array(aucs)
    aucs = np.mean(aucs)
    print('Testing based on svm: ',
          'auc=%.4f' % aucs)
    return aucs


def main():
    parser = argparse.ArgumentParser(description='S2-GAE (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', type=str, default='GCN')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--use_valedges_as_input', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--decode_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--decode_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--mask_type', type=str, default='dm',
                        help='dm | um')  # whether to use mask features
    parser.add_argument('--patience', type=int, default=50,
                        help='Use attribute or not')
    parser.add_argument('--mask_ratio', type=float, default=0.8)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    path = osp.join('dataset/class')

    # edge_index = data.edge_index

    if args.dataset in {'arxiv', 'products', 'mag'}:
        print('loading ogb dataset...')
        dataset = PygNodePropPredDataset(root=path, name=f'ogbn-{args.dataset}')
        if args.dataset in ['mag']:
            rel_data = dataset[0]
            # We are only interested in paper <-> paper relations.
            data = Data(
                x=rel_data.x_dict['paper'],
                edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                y=rel_data.y_dict['paper'])
            # data = transform(data)
        else:
            data = dataset[0]
    elif args.dataset == 'proteins':
        dataset = PygNodePropPredDataset(
            name=f'ogbn-{args.dataset}', transform=T.ToSparseTensor(attr='edge_attr'))
        data = dataset[0]
        # Move edge features to node features.
        data.x = data.adj_t.mean(dim=1)

        data.edge_index = torch.stack([data.adj_t.storage.row(), data.adj_t.storage.col()],
                                      dim=0)  # undirected edge_index

        if data.is_undirected():
            edge_index = data.edge_index
        else:
            print('### Input graph {} is directed'.format(args.dataset))
            edge_index = to_undirected(data.edge_index)

        data.edge_index = edge_index

        data.y = data.y.to(torch.float)

    elif args.dataset in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(path, args.dataset)
        data = dataset[0]

    elif args.dataset == 'Reddit':
        dataset = Reddit(osp.join(path, args.dataset))
        data = dataset[0]
    elif args.dataset in {'Photo', 'Computers'}:
        dataset = Amazon(path, args.dataset)
        data = dataset[0]
    elif args.dataset in {'CS', 'Physics'}:
        dataset = Coauthor(path, args.dataset)
        data = dataset[0]
    else:
        raise ValueError(args.dataset)

    if data.is_undirected():
        edge_index = data.edge_index
    else:
        print('### Input graph {} is directed'.format(args.dataset))
        edge_index = to_undirected(data.edge_index)

    data.full_adj_t = SparseTensor.from_edge_index(edge_index).t()

    edge_index, test_edge, test_edge_neg = do_edge_split_nc(edge_index, data.x.shape[0])

    labels = data.y

    save_path_model = 'weight/s2gaesvm-' + args.use_sage + '_{}_{}'.format(args.dataset, args.mask_type) + '_{}'.format(
        args.num_layers) + '_hidd{}-{}-{}-{}'.format(args.hidden_channels, args.mask_ratio, args.decode_layers,
                                                     args.decode_channels) + '_model.pth'
    save_path_predictor = 'weight/s2gaesvm' + args.use_sage + '_{}_{}'.format(args.dataset,
                                                                          args.mask_type) + '_{}'.format(
        args.num_layers) + '_hidd{}-{}-{}-{}'.format(args.hidden_channels, args.mask_ratio, args.decode_layers,
                                                     args.decode_channels) + '_pred.pth'

    out2_dict = {0: 'last', 1: 'combine'}
    result_dict = out2_dict
    svm_result_final = np.zeros(shape=[args.runs, len(out2_dict)])
    # Use training + validation edges for inference on test set.

    data = data.to(device)

    if args.use_sage == 'SAGE':
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    elif args.use_sage == 'GIN':
        model = GIN(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    predictor = LPDecoder(args.hidden_channels, args.decode_channels, 1, args.num_layers,
                              args.decode_layers, args.dropout).to(device)

    print('Start training with mask ratio={} # optimization edges={} / {}'.format(args.mask_ratio,
                                                                             int(args.mask_ratio *
                                                                                 edge_index.shape[0]), edge_index.shape[0]))

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        best_valid = 0.0
        best_epoch = 0
        cnt_wait = 0
        for epoch in range(1, 1 + args.epochs):
            t1 = time.time()
            loss = train(model, predictor, data, edge_index, optimizer,
                         args)
            t2 = time.time()
            auc_test = test(model, predictor, data, test_edge, test_edge_neg,
                           args.batch_size)

            if auc_test > best_valid:
                best_valid = auc_test
                best_epoch = epoch
                torch.save(model.state_dict(), save_path_model)
                torch.save(predictor.state_dict(), save_path_predictor)
                cnt_wait = 0
            else:
                cnt_wait += 1

            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Best_epoch: {best_epoch:02d}, '
                  f'Best_valid: {100 * best_valid:.2f}%, '
                  f'Loss: {loss:.4f}, ')
            print('***************')
            if cnt_wait == 50:
                print('Early stop at {}'.format(epoch))
                break

        print('##### Testing on {}/{}'.format(run, args.runs))

        model.load_state_dict(torch.load(save_path_model))
        predictor.load_state_dict(torch.load(save_path_predictor))
        feature = model(data.x, data.full_adj_t)
        feature = [feature_.detach() for feature_ in feature]

        feature_list = extract_feature_list_layer2(feature)

        for i, feature_tmp in enumerate(feature_list):
            auc_test = test_classify(feature_tmp.data.cpu().numpy(), labels.data.cpu().numpy())
            svm_result_final[run, i] = auc_test
            print('**** SVM test acc on Run {}/{} for {} is acc={}'
                  .format(run + 1, args.runs, result_dict[i], auc_test))

    svm_result_final = np.array(svm_result_final)
    if osp.exists(save_path_model):
        os.remove(save_path_model)
        os.remove(save_path_predictor)
        print('Successfully delete the saved models')


    print('\n------- Print final result for SVM')
    for i in range(len(out2_dict)):
        temp_resullt = svm_result_final[:, i]
        print('#### Final svm test result on {} is mean={} std={}'.format(result_dict[i], np.mean(temp_resullt),
                                                                          np.std(temp_resullt)))


if __name__ == "__main__":
    main()
