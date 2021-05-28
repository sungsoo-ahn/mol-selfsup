import os
import neptune.new as neptune
from tqdm import tqdm
import argparse
import numpy as np

import torch
from torch.utils.data import Subset, DataLoader, TensorDataset
import torch.optim as optim

from torch_geometric.data import DataLoader as PygDataLoader
from torch_geometric.nn import global_mean_pool

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from module.conv import GNN_node

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.gnn_node = GNN_node()
        
    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = global_mean_pool(h_node, batched_data.batch)
        return h_graph


def train(linear_model, device, loader, optimizer, task_type):
    statistics = {"loss": []}
    
    linear_model.train()
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)

        pred = linear_model(x)
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = (y == y)
        if "classification" in task_type:
            loss = cls_criterion(
                pred.to(torch.float32)[is_labeled], y.to(torch.float32)[is_labeled]
            )
        else:
            loss = reg_criterion(
                pred.to(torch.float32)[is_labeled], y.to(torch.float32)[is_labeled]
            )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        statistics["loss"].append(loss.detach().cpu().item())
    
    for key in statistics:    
        statistics[key] = np.mean(statistics[key])
    
    return statistics


def eval(linear_model, device, loader, evaluator):
    linear_model.eval()
    y_true = []
    y_pred = []

    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            pred = linear_model(x)

        y_true.append(y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def create_tsr_dataset(model, dataset, device, batch_size=1024, num_workers=0):
    loader = PygDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    x_list = []
    y_list = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        with torch.no_grad():
            x = model(batch)
            y = batch.y
            
            x_list.append(x.cpu())
            y_list.append(y.cpu())
    
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    tsr_dataset = TensorDataset(x, y)
    return tsr_dataset

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if args.checkpoint_path != "":
        state_dict = torch.load(args.checkpoint_path)
    else:
        state_dict = dict()
        
    if "neptune" in state_dict:
        run = neptune.init(
            project=state_dict["neptune"]["project"],
            name=state_dict["neptune"]["name"],
            run=state_dict["neptune"]["run_id"],
            mode=("offline" if args.offline else "async")
            )
    else:
        run = neptune.init(
            project="sungsahn0215/mol-selfsup",
            name="tune_linear",
            source_files=["*.py", "**/*.py"],
            mode=("offline" if args.offline else "async")
            )
    
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = GNN().to(device)
    if "gnn_node" in state_dict:
        model.gnn_node.load_state_dict(state_dict["gnn_node"])

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset, root="../resource/dataset")
    evaluator = Evaluator(args.dataset)
    tsr_dataset = create_tsr_dataset(model, dataset, device)
    
    ### Define loaders
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(
        Subset(tsr_dataset, split_idx["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        Subset(tsr_dataset, split_idx["valid"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        Subset(tsr_dataset, split_idx["test"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ### Begin experiment
    valid_curves = []
    test_curves = []
    train_curves = []
    best_valids = []
    best_tests = []

    for repeat in range(args.repeats):
        ### Define linear model
        emb_dim = tsr_dataset.tensors[0].size(1)
        linear_model = torch.nn.Linear(emb_dim, dataset.num_tasks).to(device)
        optimizer = optim.Adam(linear_model.parameters(), lr=0.001)
    
        valid_curve = []
        test_curve = []
        train_curve = []    

        for epoch in range(args.epochs):
            print("=====Epoch {}".format(epoch))
            print("Training...")
            train_statistics = train(linear_model, device, train_loader, optimizer, dataset.task_type)
            print({"Train": train_statistics})

            for key, val in train_statistics.items():
                run[f"tune_linear/repeat_{repeat}/train/{key}"].log(val)
            
            print("Evaluating...")
            train_perf = eval(linear_model, device, train_loader, evaluator)
            valid_perf = eval(linear_model, device, valid_loader, evaluator)        
            test_perf = eval(linear_model, device, test_loader, evaluator)
            
            print({"Train": train_perf, "Validation": valid_perf, "Test": test_perf})

            for key, val in train_perf.items():
                run[f"tune_linear/repeat_{repeat}/train/{key}"].log(val)

            for key, val in valid_perf.items():
                run[f"tune_linear/repeat_{repeat}/valid/{key}"].log(val)

            for key, val in test_perf.items():
                run[f"tune_linear/repeat_{repeat}/test/{key}"].log(val)
            
            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

            if "classification" in dataset.task_type:
                best_val_epoch = np.argmax(np.array(valid_curve))
            else:
                best_val_epoch = np.argmin(np.array(valid_curve))

            run[f"tune_linear/repeat_{repeat}/best_valid/{dataset.eval_metric}"] = valid_curve[best_val_epoch]
            run[f"tune_linear/repeat_{repeat}/best_test/{dataset.eval_metric}"] = test_curve[best_val_epoch]
        
        valid_curves.append(valid_curve)
        test_curves.append(test_curve)
        train_curves.append(train_curve)
        
        avg_valid_curve = np.array(valid_curves).mean(axis=0).tolist()
        avg_test_curve = np.array(test_curves).mean(axis=0).tolist()
        avg_train_curve = np.array(train_curves).mean(axis=0).tolist()
        
        if repeat > 0:
            run.pop(f"tune_linear/repeat_avg/valid/{dataset.eval_metric}")
            run.pop(f"tune_linear/repeat_avg/test/{dataset.eval_metric}")
            run.pop(f"tune_linear/repeat_avg/train/{dataset.eval_metric}")
        
        for val in avg_valid_curve:
            run[f"tune_linear/repeat_avg/valid/{dataset.eval_metric}"].log(val)

        for val in avg_test_curve:
            run[f"tune_linear/repeat_avg/test/{dataset.eval_metric}"].log(val)

        for val in avg_train_curve:
            run[f"tune_linear/repeat_avg/train/{dataset.eval_metric}"].log(val)

        best_valids.append(valid_curve[best_val_epoch])
        best_tests.append(test_curve[best_val_epoch])
        
        run[f"tune_linear/repeat_avg/best_valid/{dataset.eval_metric}"] = np.mean(best_valids)
        run[f"tune_linear/repeat_avg/best_test/{dataset.eval_metric}"] = np.mean(best_tests)
        


if __name__ == "__main__":
    main()
