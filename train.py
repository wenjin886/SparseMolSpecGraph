import os
os.environ.pop("SLURM_NTASKS", None)
os.environ["WANDB_DIR"] = "./exp"

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader
import json
from argparse import ArgumentParser
from datetime import datetime

from model import PeakGraphModule


def get_formatted_exp_name(exp_name, resume=False):
    formatted_time = datetime.now().strftime("%H-%M-%d-%m-%Y")
    if resume and ("resume" not in exp_name):
        formatted_exp_name = f"resume_{exp_name}_{formatted_time}"
    else:
        formatted_exp_name = f"{exp_name}_{formatted_time}"
    return formatted_exp_name

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    torch.set_float32_matmul_precision(args.precision)

    print(f"Loading dataset: {args.dataset_path}")
    dataset = torch.load(args.dataset_path)
    num_data = len(dataset)

    train_set = dataset[:int(0.85*num_data)]
    val_set = dataset[int(0.85*num_data):int(0.9*num_data)]
    test_set = dataset[int(0.9*num_data):]

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=17)
    val_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)

    with open(args.dataset_info_path, "r") as f:
        dataset_info = json.load(f)
    if arg.spec_type:
        MULTIPLETS = dataset_info["MULTIPLETS"]
        NUM_H = dataset_info["NUM_H"]
        model = PeakGraphModule(mult_class_num=len(MULTIPLETS), nH_class_num=len(NUM_H), 
                                in_dim=7, hidden_dim=32)
        print(model)
code_test = True
# code_test = False
if args.code_test:
    wandb_logger = None
    fast_dev_run = 2
else:
    exp_name = "test_nmr_graph_nHcls"
    exp_save_path = "./exp/exp_example"

    wandb_logger = WandbLogger(
                project=args.wandb_project,
                name=exp_name,
                save_dir=exp_save_path
            )
    fast_dev_run = False

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=50,
    logger=wandb_logger,
    fast_dev_run=fast_dev_run
)
    
    
trainer.fit(model, train_dataloader, val_dataloader)
trainer.test(dataloaders=test_dataloader)

if __name__ == "__main___":
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_info_path', type=str, required=True)
    parser.add_argument('--spec_type', type=str, choices=['h_nmr', 'c_nmr', 'ms'], default='h_nmr')
    parser.add_argument('--code_test', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='NMR-Graph')
    parser.add_argument('--precision', type=str, default='medium',choices=['medium', 'high'])