import os
os.environ.pop("SLURM_NTASKS", None)
os.environ["WANDB_DIR"] = "./exp"
import os.path as osp

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch_geometric.loader import DataLoader
import json
from argparse import ArgumentParser
from datetime import datetime
import random
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
    assert osp.exists(args.dataset_path), f"Dataset path does not exist: {args.dataset_path}"
    if osp.isfile(args.dataset_path):
        dataset = torch.load(args.dataset_path)
        num_data = len(dataset)
        random.seed(args.seed)
        random.shuffle(dataset)
        train_set = dataset[:int(0.85*num_data)]
        val_set = dataset[int(0.85*num_data):int(0.9*num_data)]
        test_set = dataset[int(0.9*num_data):]

        save_dir = osp.join(osp.pardir(args.dataset_path), "train_val_test_set")
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(train_set, osp.join(save_dir, "train_set.pt"))
        torch.save(val_set, osp.join(save_dir, "val_set.pt"))
        torch.save(test_set, osp.join(save_dir, "test_set.pt"))

        with open(osp.join(save_dir, "dataset_info.json"), "w") as f:
            info = {
                "split_dataset_from": args.dataset_path,
                "train_set": (len(train_set), 0.85),
                "val_set": (len(val_set), 0.05),
                "test_set": (len(test_set), 0.1),
            }
            json.dump(info, f)

    elif osp.isdir(args.dataset_path):
        train_set = torch.load(osp.join(args.dataset_path, "train_set.pt"))
        val_set = torch.load(osp.join(args.dataset_path, "val_set.pt"))
        test_set = torch.load(osp.join(args.dataset_path, "test_set.pt"))


    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=17)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    with open(args.dataset_info_path, "r") as f:
        dataset_info = json.load(f)
    if args.spec_type:
        MULTIPLETS = dataset_info["MULTIPLETS"]
        NUM_H = dataset_info["NUM_H"]
        model = PeakGraphModule(mult_class_num=len(MULTIPLETS), nH_class_num=len(NUM_H), 
                                in_node_dim=16, hidden_node_dim=64, graph_dim=32, 
                                num_layers=4, num_heads=4,
                                warm_up_step=args.warm_up_step, lr=args.lr)
        print(model)


    if args.code_test:
        wandb_logger = None
        fast_dev_run = 2
    else:
        wandb_logger = WandbLogger(
                    project=args.wandb_project,
                    name=get_formatted_exp_name(args.exp_name),
                    save_dir=args.exp_save_path
                )
        fast_dev_run = False

    if device == torch.device("cuda"):
        device_num = torch.cuda.device_count()
        accelerator = "gpu"
    else:
        device_num = "auto"
        accelerator = "auto"

    checkpoint_callback = ModelCheckpoint(dirpath=args.exp_save_path, 
                                          save_top_k=args.save_top_k, 
                                          every_n_epochs=args.save_every_n_epochs,
                                          monitor=args.loss_monitor,
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=device_num,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks
    )
        
        
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # 在code_test模式下直接测试，不使用checkpoint
    if args.code_test:
        trainer.test(ckpt_path='last', dataloaders=test_dataloader)
    else:
        # 正常模式下使用最佳checkpoint测试
        trainer.test(ckpt_path="best", dataloaders=test_dataloader)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--exp_save_path', type=str, default='../exp')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_info_path', type=str, required=True)

    parser.add_argument('--code_test', action='store_true')
    parser.add_argument('--warm_up_step', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_top_k', type=int, default=3)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--spec_type', type=str, choices=['h_nmr', 'c_nmr', 'ms'], default='h_nmr')
    parser.add_argument('--wandb_project', type=str, default='NMR-Graph')
    parser.add_argument('--precision', type=str, default='medium',choices=['medium', 'high'])
    parser.add_argument('--loss_monitor', type=str, default='val_loss')
    

    args = parser.parse_args()
    print(args)
    main(args)