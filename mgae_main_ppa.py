import argparse
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch_geometric.transforms as T
from linkmodel import PygLinkPropPredDataset, Evaluator
from logger import Logger
from model import LPDecoder_ogb as LPDecoder
from model import GCN_mgaev3 as GCN
from model import SAGE_mgaev2 as SAGE
from model import GIN_mgaev2 as GIN
from torch_geometric.utils import to_undirected, add_self_loops, negative_sampling
from torch_sparse import SparseTensor


def random_edge_mask(args, split_edge, device, num_nodes):
    # edge_index = split_edge['train']['edge']
    edge_index = torch.stack([split_edge['train']['edge'][:, 1], split_edge['train']['edge'][:, 0]], dim=1)
    edge_index = torch.cat([split_edge['train']['edge'], edge_index], dim=0)

    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * args.keep_prob)
    pre_index = torch.from_numpy(index[0:-mask_num])
    mask_index = torch.from_numpy(index[-mask_num:])
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].to(device)

    edge_index = edge_index_train
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index).t()
    return adj, edge_index, edge_index_mask


def train(model, predictor, data, split_edge, optimizer, args):
    model.train()
    predictor.train()

    # pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    adj, _, pos_train_edge = random_edge_mask(args, split_edge, data.x.device, data.x.size(0))
    adj = adj.to(data.x.device)

    for perm in DataLoader(range(pos_train_edge.size(0)), args.batch_size,
                           shuffle=True):

        optimizer.zero_grad()

        h = model(data.x, adj)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h, edge)
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
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
def test(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()

    h = model(data.x, data.adj_t)

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device)
    pos_test_edge = split_edge['test']['edge'].to(data.x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h, edge).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h, edge).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h, edge).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    h = model(data.x, data.full_adj_t)

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

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-PPA (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_sage', type=str, default='GCN')
    parser.add_argument('--dataset', type=str, default='ogbl-ppa')
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--decode_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--decode_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--decoder_mask', type=str, default='nmask',
                        help='mask | nmask')  # whether to use mask features
    parser.add_argument('--patience', type=int, default=50,
                        help='Use attribute or not')
    parser.add_argument('--keep_prob', type=float, default=0.8)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ppa',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.x = data.x.to(torch.float)
    if args.use_node_embedding:
        data.x = torch.cat([data.x, torch.load('embedding.pt')], dim=-1)
    data = data.to(device)

    split_edge = dataset.get_edge_split()

    edge_index = to_undirected(split_edge['train']['edge'].t())
    if args.use_sage == 'GCN':
        edge_index, _ = add_self_loops(edge_index)
    else:
        edge_index = edge_index
    val_edge_index = split_edge['valid']['edge'].t()
    val_edge_index = to_undirected(val_edge_index)
    data.adj_t = SparseTensor.from_edge_index(edge_index).t()
    edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
    data.full_adj_t = SparseTensor.from_edge_index(edge_index).t()

    data = data.to(device)

    save_path_model = 'weight/mgaev2udict-' + args.use_sage + '_{}_{}'.format(args.dataset, args.decoder_mask) + '_{}'.format(
        args.num_layers) + '_hidd{}-{}-{}-{}-{}-{}'.format(args.hidden_channels, args.keep_prob, args.decode_layers,
                                                           args.decode_channels,args.lr, args.dropout) + '_model.pth'
    save_path_predictor = 'weight/mgaev2udict' + args.use_sage + '_{}_{}'.format(args.dataset, args.decoder_mask) + '_{}'.format(
        args.num_layers) + '_hidd{}-{}-{}-{}-{}-{}'.format(args.hidden_channels, args.keep_prob, args.decode_layers,
                                                           args.decode_channels,args.lr, args.dropout) + '_pred.pth'

    metric = 'Hits@100'

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

    evaluator = Evaluator(name='ogbl-ppa')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    print('Start training with mask ratio={} # optimization edges={}'.format(args.keep_prob,
                                                                             int(args.keep_prob *
                                                                                 split_edge['train']['edge'].shape[0])))

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        best_valid = 0.0
        best_epoch = 0
        wait_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge, optimizer,
                         args)

            results = test(model, predictor, data, split_edge, evaluator,
                           args.batch_size)

            valid_hits = results[metric][1]
            if valid_hits > best_valid:
                best_valid = valid_hits
                best_epoch = epoch
                torch.save(model.state_dict(), save_path_model)
                torch.save(predictor.state_dict(), save_path_predictor)
                wait_cnt = 0
            else:
                wait_cnt += 1

            for key, result in results.items():
                train_hits, valid_hits, test_hits = result
                print(key)
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Best_epoch: {best_epoch:02d}, '
                      f'Best_valid: {100 * best_valid:.2f}%, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_hits:.2f}%, '
                      f'Valid: {100 * valid_hits:.2f}%, '
                      f'Test: {100 * test_hits:.2f}%')
            print('***************')
            if wait_cnt == args.patience:
                print('Early stop at epoch={}'.format(epoch))
                break

        print('##### Testing on {}/{}'.format(run, args.runs))

        model.load_state_dict(torch.load(save_path_model))
        predictor.load_state_dict(torch.load(save_path_predictor))
        results = test(model, predictor, data, split_edge, evaluator,
                       args.batch_size)

        for key, result in results.items():
            train_hits, valid_hits, test_hits = result
            print(key)
            print(f'**** Testing on Run: {run + 1:02d}, '
                  f'Epoch: {best_epoch:02d}, '
                  f'Train: {100 * train_hits:.2f}%, '
                  f'Valid: {100 * valid_hits:.2f}%, '
                  f'Test: {100 * test_hits:.2f}%')

        for key, result in results.items():
            loggers[key].add_result(run, result)

    print('##### Final Testing result')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
