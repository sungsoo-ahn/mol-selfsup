import os
import argparse

import torch
import neptune.new as neptune
from module.conv import GNN_node

def main():
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbgmol* data with Pytorch Geometrics"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tag", type=str, default="nopretrain")
    args = parser.parse_args()

    neptune_project = "sungsahn0215/mol-selfsup"
    neptune_name = "pretrain_nopretrain"
    run = neptune.init(
        project=neptune_project,
        name=neptune_name,
        source_files=["*.py", "**/*.py"],
        mode=("offline" if args.offline else "async"),
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

    gnn_node = GNN_node().to(device)

    state_dict = {
        "gnn_node": gnn_node.state_dict(),
        "neptune": {
            "project": neptune_project,
            "name": neptune_name,
            "run_id": neptune_run_id,
        },
    }

    torch.save(state_dict, f"{checkpoint_dir}/checkpoint.pth")


if __name__ == "__main__":
    main()
