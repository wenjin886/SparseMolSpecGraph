import os
os.environ.pop("SLURM_NTASKS", None)
os.environ["WANDB_DIR"] = "./exp"
import os.path as osp

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch_geometric.loader import DataLoader

import json
from argparse import ArgumentParser
from datetime import datetime
import random
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

from model import PeakGraphModule

def get_formatted_exp_name(exp_name, resume=False):
    formatted_time = datetime.now().strftime("%H-%M-%d-%m-%Y")
    if resume and ("resume" not in exp_name):
        formatted_exp_name = f"resume_{exp_name}_{formatted_time}"
    else:
        formatted_exp_name = f"{exp_name}_{formatted_time}"
    return formatted_exp_name

def split_dataset(dataset_path, seed, save_dir_name):
    save_dir = osp.join(osp.dirname(dataset_path), save_dir_name)
    if not osp.exists(save_dir): os.makedir(save_dir)
    print(f"Will Save splitted dataset in {save_dir}")

    dataset = torch.load(dataset_path)
    num_data = len(dataset)

    print(f"Shuffling dataset with seed ({seed})...")
    random.seed(seed)
    random.shuffle(dataset)
    print("Splitting...")
    train_set = dataset[:int(0.85*num_data)]
    val_set = dataset[int(0.85*num_data):int(0.9*num_data)]
    test_set = dataset[int(0.9*num_data):]

    
    torch.save(train_set, osp.join(save_dir, "train_set.pt"))
    torch.save(val_set, osp.join(save_dir, "val_set.pt"))
    torch.save(test_set, osp.join(save_dir, "test_set.pt"))

    with open(osp.join(save_dir, "splitted_set_info.json"), "w") as f:
        info = {
            "split_dataset_from": args.dataset_path,
            "train_set": (len(train_set), 0.85),
            "val_set": (len(val_set), 0.05),
            "test_set": (len(test_set), 0.1),
        }
        json.dump(info, f)
    return train_set, val_set, test_set

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    torch.set_float32_matmul_precision(args.precision)
  
    
    if args.spec_type == "h_nmr":
        if args.label_type == "origin":
            with open(args.dataset_info_path, "r") as f:
                dataset_info = json.load(f)
            MULTIPLETS = dataset_info["MULTIPLETS"]
            NUM_H = dataset_info["NUM_H"]
            
        elif args.label_type == "mapped":
            mult_map_info_path = "../Dataset/h_nmr/multiplet_mapping.json"
            nh_map_info_path = "../Dataset/h_nmr/nh_mapping.json"
            with open(mult_map_info_path, "r") as f:
                mult_map_info = json.load(f)
                MULTIPLETS = mult_map_info['multiplet_label_to_id']
            with open(nh_map_info_path, "r") as f:
                nh_map_info = json.load(f)
                NUM_H = nh_map_info['nh_label_to_id']
        
        print(f"lable num | MULTIPLETS: {len(MULTIPLETS)} | NUM_H: {len(NUM_H)}")

        # # train_set是一个包含Data对象的list
        # all_multiplets = []
        # print("compute class weight")
        # for data in tqdm(train_set,desc="train set"):
        #     # 假设每个data.multiplets是1D tensor
        #     all_multiplets.append(data.multiplets)
        # all_multiplets = torch.cat(all_multiplets)
        # mult_class_counts = torch.bincount(all_multiplets, minlength=len(MULTIPLETS))  # 保证长度为len(MULTIPLETS)
        # print(mult_class_counts)
        # mult_class_weights = 1.0 / (mult_class_counts.float() + 1e-6)
        # mult_class_weights[mult_class_counts == 0] = 0.0
        # # mult_class_weights = mult_class_weights * (len(mult_class_counts) / mult_class_weights.sum())
        # print(mult_class_weights)

        model = PeakGraphModule(mult_class_num=len(MULTIPLETS), nH_class_num=len(NUM_H), 
                                mult_embed_dim=args.mult_embed_dim, nH_embed_dim=args.nH_embed_dim, c_w_embed_dim=args.c_w_embed_dim,
                                num_layers=args.num_layers, num_heads=args.num_heads,
                                warm_up_step=args.warm_up_step, lr=args.lr)
        print(model)

    exp_name = get_formatted_exp_name(args.exp_name)
    save_dirpath = osp.join(args.exp_save_path, exp_name)
    print(f"Will save training results in {save_dirpath}")

    
    
    if args.code_test:
        wandb_logger = None
        fast_dev_run = 2
    else:
        wandb_logger = WandbLogger(
                    project=args.wandb_project,
                    name=exp_name,
                    save_dir=args.exp_save_path
                )
        fast_dev_run = False
        
        

    if device == torch.device("cuda"):
        device_num = torch.cuda.device_count()
        accelerator = "gpu"
    else:
        device_num = "auto"
        accelerator = "auto"

    
    checkpoint_callback = ModelCheckpoint(dirpath=save_dirpath, 
                                          save_top_k=args.save_top_k, 
                                          every_n_epochs=args.save_every_n_epochs,
                                          monitor=args.loss_monitor,
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(
            monitor='val_loss',      # 监控指标，例如验证集损失
            patience=5,             # 如果10个epoch内指标未改善则停止训练
            mode='min',              # 监控指标越小越好（如损失函数）
            verbose=True             # 是否打印信息
        )
    callbacks = [checkpoint_callback, lr_monitor, early_stop_callback]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=device_num,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks
    )

    if not args.code_test:
        if not os.path.exists(save_dirpath):
            print(f"Making dir: {save_dirpath}")
            os.makedirs(save_dirpath)
        with open(osp.join(save_dirpath, 'config.json'), 'w') as f: # 保存为 JSON 文件
            args_dict = vars(args)
            json.dump(args_dict, f, indent=4)
        
    print(f"Loading dataset: {args.dataset_path}")
    assert osp.exists(args.dataset_path), f"Dataset path does not exist: {args.dataset_path}"
    if osp.isfile(args.dataset_path):
        train_set, val_set, test_set = split_dataset(args.dataset_path, args.seed, args.splitted_set_save_dir_name)
    elif osp.isdir(args.dataset_path):
        if args.code_test:
            print("Only load val set due to code test...")
            train_set = torch.load(osp.join(args.dataset_path, "val_set.pt"))
            val_set = train_set
            test_set = train_set
        else:
            train_set = torch.load(osp.join(args.dataset_path, "train_set.pt"))
            val_set = torch.load(osp.join(args.dataset_path, "val_set.pt"))
            test_set = torch.load(osp.join(args.dataset_path, "test_set.pt"))

    

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

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
    parser.add_argument('--splitted_set_save_dir_name', type=str, default="train_val_test_set")
    parser.add_argument('--dataset_info_path', type=str)
    parser.add_argument('--spec_type', type=str, choices=['h_nmr', 'c_nmr', 'ms'], default='h_nmr')
    parser.add_argument('--label_type', type=str, choices=['origin', 'mapped'], required=True)

    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--mult_embed_dim', type=int, default=128)
    parser.add_argument('--nH_embed_dim', type=int, default=64)
    parser.add_argument('--c_w_embed_dim', type=int, default=32)

    parser.add_argument('--code_test', action='store_true')
    parser.add_argument('--warm_up_step', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_top_k', type=int, default=3)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    
    parser.add_argument('--wandb_project', type=str, default='NMR-Graph')
    parser.add_argument('--precision', type=str, default='medium',choices=['medium', 'high'])
    parser.add_argument('--loss_monitor', type=str, default='val_loss')
    

    args = parser.parse_args()
    print(args)
    main(args)