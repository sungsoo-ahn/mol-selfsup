import random
import os
from tqdm import tqdm
import argparse
import numpy as np
from collections import defaultdict
import networkx as nx

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

import neptune.new as neptune

from module.conv import GNN_node

### automatic dataloading and splitting
atom_feature_dims = get_atom_feature_dims()
bond_feature_dims = get_bond_feature_dims()


def reset_idxes(G):
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


def graph_data_obj_to_nx_simple(data):
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        G.add_node(i, atom_features=atom_features[i])
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, edge_attr=edge_attr[j])

    return G


def nx_to_graph_data_obj_simple(G):
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_features_list.append(node["atom_features"])
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_atom_features = len(atom_feature_dims)  # atom type,  chirality tag
    num_bond_features = len(bond_feature_dims)  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = edge["edge_attr"]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

criterion = torch.nn.BCEWithLogitsLoss()


def compute_categorical_accuracy(pred, target):
    return torch.sum(torch.max(pred, dim=1)[1] == target) / pred.size(0)

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

def train(model, model_context, device, loader, optimizer, optimizer_context, neg_samples=1):
    statistics = defaultdict(list)

    model.train()
    for batch in tqdm(loader):
        batch = batch.to(device)

        # creating substructure representation
        substruct_rep = model.forward_with_elems(
            batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct,
            )[batch.center_substruct_idx]

        ### creating context representations
        overlapped_node_rep = model_context.forward_with_elems(
            batch.x_context, batch.edge_index_context, batch.edge_attr_context
        )[batch.overlap_context_substruct_idx]

        context_rep = global_mean_pool(overlapped_node_rep, batch.batch_overlapped_context)
        neg_context_rep = torch.cat(
            [context_rep[cycle_index(len(context_rep), i + 1)] for i in range(neg_samples)],
            dim=0,
        )

        pred_pos = torch.sum(substruct_rep * context_rep, dim=1)
        pred_neg = torch.sum(substruct_rep.repeat((neg_samples, 1)) * neg_context_rep, dim=1)

        loss_pos = criterion(
            pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double()
        )
        loss_neg = criterion(
            pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double()
        )
        loss = loss_pos + neg_samples * loss_neg
        
        optimizer.zero_grad()
        optimizer_context.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_context.step()

        with torch.no_grad():
            acc = 0.5 * (
                torch.sum(pred_pos > 0) / len(pred_pos) + torch.sum(pred_neg < 0) / len(pred_neg)
            )

        statistics["contextpred/loss"].append(loss.detach().cpu().item())
        statistics["contextpred/acc"].append(acc.cpu().item())

    for key in statistics:
        statistics[key] = np.mean(statistics[key])

    return statistics

def collate(data_list):
    batch = Data()
    keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]

    for key in keys:
        batch[key] = []

    batch.batch_overlapped_context = []
    batch.overlapped_context_size = []

    cumsum_main = 0
    cumsum_substruct = 0
    cumsum_context = 0

    i = 0
    
    for data in data_list:
        if hasattr(data, "x_context"):
            num_nodes = data.num_nodes
            num_nodes_substruct = len(data.x_substruct)
            num_nodes_context = len(data.x_context)

            batch.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx), ), i, dtype=torch.long))
            batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

            ###batching for the substructure graph
            for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                item = data[key]
                cumsum = key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"]
                item = item + cumsum_substruct if cumsum else item
                batch[key].append(item)
            

            ###batching for the context graph
            for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                item = data[key]
                cumsum = key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"]
                item = item + cumsum_context if cumsum else item
                batch[key].append(item)

            cumsum_main += num_nodes
            cumsum_substruct += num_nodes_substruct   
            cumsum_context += num_nodes_context
            i += 1

    for key in keys:
        cat_dim = (
            -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0
        )
        batch[key] = torch.cat(batch[key], dim=cat_dim)
        
    batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
    batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

    return batch.contiguous()

def main():
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbgmol* data with Pytorch Geometrics"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--l1", type=int, default=4)
    parser.add_argument("--l2", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tag", type=str, default="contextpred")
    args = parser.parse_args()

    neptune_project = "sungsahn0215/mol-selfsup"
    neptune_name = "pretrain_contextpred"
    run = neptune.init(
        project=neptune_project,
        name=neptune_name,
        source_files=["*.py", "**/*.py"],
        mode=("offline" if args.offline else "async"),
    )
    run["parameters"] = vars(args)
    neptune_run_id = None if args.offline else run["sys/id"].fetch()
    checkpoint_dir = f"../resource/checkpoint/{args.tag}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    def transform(data):
        num_atoms = data.x.size()[0]
        root_idx = random.sample(range(num_atoms), 1)[0]

        G = graph_data_obj_to_nx_simple(data)

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(G, root_idx, args.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)
            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[root_idx]])

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx, args.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx, args.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)
            context_data = nx_to_graph_data_obj_simple(context_G)
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index
        context_substruct_overlap_idxes = list(
            set(context_node_idxes).intersection(set(substruct_node_idxes))
        )
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [
                context_node_map[old_idx] for old_idx in context_substruct_overlap_idxes
            ]
            data.overlap_context_substruct_idx = torch.tensor(
                context_substruct_overlap_idxes_reorder
            )

        return data

    train_dataset = PygGraphPropPredDataset(
        name=args.dataset, root="../resource/dataset", transform=transform
    )

    split_idx = train_dataset.get_idx_split()
    train_loader = DataLoader(
        train_dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate
    )

    model = GNN_node().to(device)
    model_context = GNN_node(num_layer=args.l2 - args.l1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer_context = optim.Adam(model_context.parameters(), lr=0.001)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train_statistics = train(
            model, model_context, device, train_loader, optimizer, optimizer_context
        )
        print(train_statistics)
        for key, val in train_statistics.items():
            run[f"pretrain/train/{key}"].log(val)

        state_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "model_context": model_context.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_context": optimizer_context.state_dict(),
            "gnn_node": model.state_dict(),
            "neptune": {
                "project": neptune_project,
                "name": neptune_name,
                "run_id": neptune_run_id,
            },
        }

        torch.save(state_dict, f"{checkpoint_dir}/checkpoint{epoch:03d}.pth")

    torch.save(state_dict, f"{checkpoint_dir}/checkpoint.pth")

    print("Finished training!")


if __name__ == "__main__":
    main()
