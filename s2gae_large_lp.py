import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from model import GCN_mgaev3 as GCN
from model import SAGE_mgaev2 as SAGE
from model import GIN_mgaev2 as GIN
from model import LPDecoder
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import time
from logger import Logger
from utils import edgemask_dm, edgemask_um
from torch_geometric.utils import to_undirected, add_self_loops, negative_sampling
from torch_sparse import SparseTensor


def train(model, predictor, x, data, split_edge, optimizer, args):
    if args.dataset == 'ogbl-ddi':
        row, col, _ = data.full_adj_t.coo()
        edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    total_loss = total_examples = 0

    if args.mask_type == 'um':
        adj, _, pos_train_edge = edgemask_um(args.mask_ratio, split_edge, x.device, x.shape[0])
    else:
        adj, _, pos_train_edge = edgemask_dm(args.mask_ratio, split_edge, x.device, x.shape[0])

    adj = adj.to(x.device)

    for perm in DataLoader(range(pos_train_edge.size(0)), args.batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(x, adj)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h, edge)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        if args.dataset == 'ogbl-ddi':
            edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                     num_neg_samples=perm.size(0), method='dense')
        else:
            edge = torch.randint(0, x.shape[0], edge.size(), dtype=torch.long,
                                 device=data.x.device)

        neg_out = predictor(h, edge)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        if args.dataset == 'ogbl-ddi':
            torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, data, split_edge, evaluator, batch_size, data_name):
    model.eval()
    predictor.eval()

    h = model(x, data.adj_t)

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

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

    h = model(x, data.full_adj_t)

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
    if data_name == 'ogbl-ddi':
        tops = [10, 20, 30]
    else:
        tops = [10, 50, 100]
    for K in tops:
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
    parser = argparse.ArgumentParser(description='S2GAE-Large (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', type=str, default='GCN')
    parser.add_argument('--dataset', type=str, default='ogbl-ddi')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--decode_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--decode_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024, help='64 * 1024')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=3)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--mask_type', type=str, default='dm',
                        help='dm | um')  # whether to use mask features
    parser.add_argument('--mask_ratio', type=float, default=0.7)
    parser.add_argument('--patience', type=int, default=50,
                        help='Use attribute or not')
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.dataset == 'ogbl-ddi':
        dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                         transform=T.ToSparseTensor())
        data = dataset[0]
        num_nodes = data.num_nodes
    elif args.dataset == 'ogbl-collab':
        dataset = PygLinkPropPredDataset(name='ogbl-collab')
        data = dataset[0]
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
        data = T.ToSparseTensor()(data)
        num_nodes = data.x.shape[0]
    else:
        dataset = PygLinkPropPredDataset(name='ogbl-ppa',
                                         transform=T.ToSparseTensor())

        data = dataset[0]
        data.x = data.x.to(torch.float)
        num_nodes = data.x.shape[0]

    split_edge = dataset.get_edge_split()

    edge_index = to_undirected(split_edge['train']['edge'].t())
    if args.use_sage == 'GCN':
        edge_index, _ = add_self_loops(edge_index)
    else:
        edge_index = edge_index
    val_edge_index = split_edge['valid']['edge'].t()
    val_edge_index = to_undirected(val_edge_index)
    adj_t = SparseTensor.from_edge_index(edge_index).t()
    edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
    adj_test = SparseTensor.from_edge_index(edge_index).t()

    data.adj_t = adj_t
    data.full_adj_t = adj_test
    data = data.to(device)

    save_path_model = 'weight/s2gae-' + args.use_sage + '_{}_{}'.format(args.dataset, args.mask_type) + '_{}'.format(
        args.num_layers) + '_hidd{}-{}-{}-{}-{}-{}'.format(args.hidden_channels, args.mask_ratio, args.decode_layers,
                                                     args.decode_channels, args.lr, args.dropout) + '_model.pth'
    save_path_emb = 'weight/s2gae-' + args.use_sage + '_{}_{}'.format(args.dataset,
                                                                         args.mask_type) + '_{}'.format(
        args.num_layers) + '_hidd{}-{}-{}-{}-{}-{}'.format(args.hidden_channels, args.mask_ratio, args.decode_layers,
                                                     args.decode_channels, args.lr, args.dropout) + '_emb.pth'
    save_path_predictor = 'weight/s2gae-' + args.use_sage + '_{}_{}'.format(args.dataset,
                                                                          args.mask_type) + '_{}'.format(
        args.num_layers) + '_hidd{}-{}-{}-{}-{}-{}'.format(args.hidden_channels, args.mask_ratio, args.decode_layers,
                                                     args.decode_channels, args.lr, args.dropout) + '_pred.pth'

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
    print('Start training with mask ratio={} # optimization edges={} / {}'.format(args.mask_ratio,
                                                                             int(args.mask_ratio * split_edge['train']['edge'].shape[0]), split_edge['train']['edge'].shape[0]))

    if args.dataset == 'ogbl-ddi':
        emb = torch.nn.Embedding(num_nodes, args.hidden_channels).to(device)
        input_feature = args.hidden_channels
    else:
        emb = data.x.to(device)
        input_feature = data.x.shape[1]

    if args.use_sage == 'SAGE':
        model = SAGE(input_feature, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    elif args.use_sage == 'GIN':
        model = GIN(input_feature, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(input_feature, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    predictor = LPDecoder(args.hidden_channels, args.decode_channels, 1, args.num_layers,
                              args.decode_layers, args.dropout).to(device)

    if args.dataset == 'ogbl-ddi':
        metric = 'Hits@20'
        loggers = {
            'Hits@10': Logger(args.runs, args),
            'Hits@20': Logger(args.runs, args),
            'Hits@30': Logger(args.runs, args),
        }
    elif args.dataset == 'ogbl-collab':
        metric = 'Hits@50'
        loggers = {
            'Hits@10': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
        }
    else:
        metric = 'Hits@50'
        loggers = {
            'Hits@10': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
        }
    evaluator = Evaluator(name=args.dataset)

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()

        if args.dataset == 'ogbl-ddi':
            torch.nn.init.xavier_uniform_(emb.weight)
            optimizer = torch.optim.Adam(list(model.parameters()) + list(emb.parameters()) +
            list(predictor.parameters()), lr=args.lr)
            x = emb.weight
        else:
            optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()),
                                         lr=args.lr)
            x = emb

        best_valid = 0.0
        best_epoch = 0
        wait_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            t1 = time.time()
            loss = train(model, predictor, x, data, split_edge,
                         optimizer, args)
            t2 = time.time()

            results = test(model, predictor, x, data, split_edge,
                           evaluator, args.batch_size, args.dataset)

            valid_hits = results[metric][1]
            if valid_hits > best_valid:
                best_valid = valid_hits
                best_epoch = epoch
                torch.save(model.state_dict(), save_path_model)
                torch.save(predictor.state_dict(), save_path_predictor)
                if args.dataset == 'ogbl-ddi':
                    torch.save(emb.state_dict(), save_path_emb)
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
                      f'Test: {100 * test_hits:.2f}%',
                      f'Time: {t2-t1:.2f}'
                      )
            print('***************')
            if wait_cnt == args.patience:
                print('Early stop at epoch={}'.format(epoch))
                break

        print('##### Testing on {}/{}'.format(run, args.runs))

        model.load_state_dict(torch.load(save_path_model))
        predictor.load_state_dict(torch.load(save_path_predictor))
        if args.dataset == 'ogbl-ddi':
            emb.load_state_dict(torch.load(save_path_emb))
        results = test(model, predictor, x, data, split_edge,
                       evaluator, args.batch_size, args.dataset)
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

    if os.path.exists(save_path_model):
        os.remove(save_path_model)
        os.remove(save_path_predictor)
        if args.dataset == 'ogbl-ddi':
            os.remove(save_path_emb)
        print('Successfully delete the saved models')

    print('##### Final Testing result')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
