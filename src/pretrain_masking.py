import random
import os
from tqdm import tqdm
import argparse
import numpy as np
from collections import defaultdict

import torch
import torch.optim as optim

from torch_geometric.data import DataLoader

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.utils.features import get_atom_feature_dims

import neptune.new as neptune

from module.conv import GNN_node


class GNN(torch.nn.Module):
    def __init__(self, emb_dim=300):
        super(GNN, self).__init__()
        atom_feature_dims = get_atom_feature_dims()
        self.gnn_node = GNN_node()
        self.node_pred = torch.nn.Linear(emb_dim, atom_feature_dims[0])

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_node = h_node[batched_data.node_mask]
        h_node = self.node_pred(h_node)
        return h_node


criterion = torch.nn.CrossEntropyLoss()


def compute_categorical_accuracy(pred, target):
    return torch.sum(torch.max(pred, dim=1)[1] == target) / pred.size(0)


def train(model, device, loader, optimizer):
    statistics = defaultdict(list)

    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            continue

        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.masked_x0)
        with torch.no_grad():
            acc = compute_categorical_accuracy(pred, batch.masked_x0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        statistics["masking/loss"].append(loss.detach().cpu().item())
        statistics["masking/acc"].append(acc.cpu().item())

    for key in statistics:
        statistics[key] = np.mean(statistics[key])

    return statistics


def main():
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbgmol* data with Pytorch Geometrics"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tag", type=str, default="masking")
    args = parser.parse_args()

    neptune_project = "sungsahn0215/mol-selfsup"
    neptune_name = "pretrain_masking"
    run = neptune.init(
        project=neptune_project,
        name=neptune_name,
        source_files=["*.py", "**/*.py"],
        mode=("offline" if args.offline else "async")
    )
    run["parameters"] = vars(args)
    neptune_run_id = run["sys/id"].fetch()
    checkpoint_dir = f"../resource/checkpoint/{args.tag}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    ### automatic dataloading and splitting
    atom_feature_dims = get_atom_feature_dims()

    def transform(data):
        num_nodes = data.x.size(0)
        masked_nodes = random.sample(range(num_nodes), int(args.mask_rate * num_nodes))
        data.node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.node_mask[masked_nodes] = True
        data.masked_x0 = data.x[data.node_mask, 0]

        x = data.x.clone()
        for idx, dim in enumerate(atom_feature_dims):
            x[masked_nodes, idx] = dim

        data.x = x

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
    )

    model = GNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train_statistics = train(model, device, train_loader, optimizer)
        for key, val in train_statistics.items():
            run[f"pretrain/train/{key}"].log(val)

        state_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "gnn_node": model.gnn_node.state_dict(),
            "neptune": {
                    "project": neptune_project,
                    "name": neptune_name,
                    "run_id": neptune_run_id,
                }
        }


        torch.save(state_dict, f"{checkpoint_dir}/checkpoint{epoch:03d}.pth")

    torch.save(state_dict, f"{checkpoint_dir}/checkpoint.pth")
    
    print("Finished training!")


if __name__ == "__main__":
    main()
