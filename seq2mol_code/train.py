import os
os.environ.pop("SLURM_NTASKS", None)
os.environ["WANDB_DIR"] = "../exp"
import os.path as osp

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch_geometric.loader import DataLoader
from transformers import PreTrainedTokenizerFast
import json
from argparse import ArgumentParser
from datetime import datetime
import random
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

from model_seq2mol import NMRSeq2MolGenerator
from loader import create_nmr2mol_dataloader, NMR2MolDataset

def get_formatted_exp_name(exp_name, resume=False):
    formatted_time = datetime.now().strftime("%H-%M_%d-%m-%Y")
    if resume and ("resume" not in exp_name):
        formatted_exp_name = f"resume_{exp_name}_{formatted_time}"
    else:
        formatted_exp_name = f"{exp_name}_{formatted_time}"
    return formatted_exp_name

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    torch.set_float32_matmul_precision(args.precision)
  
    
    src_tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.src_tokenizer_path)
    tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tgt_tokenizer_path)

    if args.resume:
        exp_name = get_formatted_exp_name(args.exp_name, resume=True)
        model = NMRSeq2MolGenerator.load_from_checkpoint(args.checkpoint_path)
    else:
        model = NMRSeq2MolGenerator(smiles_tokenizer_path=args.tgt_tokenizer_path,
                                    src_vocab_size=len(src_tokenizer.get_vocab()), 
                                    use_formula=args.use_formula, nmr_bos_token_id=src_tokenizer.vocab["1HNMR"],
                                    spec_formula_encoder_head=args.spec_formula_encoder_head, 
                                    spec_formula_encoder_layer=args.spec_formula_encoder_layer,
                                    d_model=args.d_model, d_ff=args.d_ff, 
                                    decoder_head=args.decoder_head, N_decoder_layer=args.N_decoder_layer, 
                                    dropout=args.dropout,
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
                    save_dir=save_dirpath
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
                                          monitor=args.monitor,
                                          mode=args.monitor_mode,
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if args.early_stop:
        early_stop_callback = EarlyStopping(
                monitor=args.monitor,      # 监控指标，例如验证集损失
                patience=5,             # 如果10个epoch内指标未改善则停止训练
                mode='min',              # 监控指标越小越好（如损失函数）
                verbose=True             # 是否打印信息
            )
        callbacks = [checkpoint_callback, lr_monitor, early_stop_callback]
    else:
        callbacks = [checkpoint_callback, lr_monitor]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=device_num,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad_batches
    )

    if not args.code_test:
        if not os.path.exists(save_dirpath):
            print(f"Making dir: {save_dirpath}")
            os.makedirs(save_dirpath)
        with open(osp.join(save_dirpath, 'config.json'), 'w') as f: # 保存为 JSON 文件
            args_dict = vars(args)
            json.dump(args_dict, f, indent=4)
        
    print(f"Loading dataset: {args.data_dir_path}")

    train_set = NMR2MolDataset(src_data=osp.join(args.data_dir_path, "src-train.txt"), 
                               tgt_data=osp.join(args.data_dir_path, "tgt-train.txt"), 
                               src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
    val_set = NMR2MolDataset(src_data=osp.join(args.data_dir_path, "src-val.txt"), 
                             tgt_data=osp.join(args.data_dir_path, "tgt-val.txt"), 
                             src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
    test_set = NMR2MolDataset(src_data=osp.join(args.data_dir_path, "src-test.txt"), 
                              tgt_data=osp.join(args.data_dir_path, "tgt-test.txt"), 
                              src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        
    

    train_dataloader = create_nmr2mol_dataloader(train_set, batch_size=args.batch_size,  num_workers=17)
    val_dataloader = create_nmr2mol_dataloader(val_set, batch_size=args.batch_size, num_workers=17)
    test_dataloader = create_nmr2mol_dataloader(test_set, batch_size=args.batch_size, num_workers=17)

    
    if not args.code_test:
        trainer.fit(model, train_dataloader, val_dataloader,
                ckpt_path=args.checkpoint_path if args.resume else None
                )
        trainer.test(ckpt_path="best", dataloaders=test_dataloader)
    else:
        trainer.fit(model, train_dataloader,
                    ckpt_path=args.checkpoint_path if args.resume else None
                    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--exp_save_path', type=str, default='../exp')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--data_dir_path', type=str, required=True)
    parser.add_argument('--src_tokenizer_path', type=str, required=True)
    parser.add_argument('--tgt_tokenizer_path', type=str, required=True)

    parser.add_argument('--use_formula', type=eval, default=True)
    parser.add_argument('--spec_formula_encoder_head', type=int, default=8)
    parser.add_argument('--spec_formula_encoder_layer', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--decoder_head', type=int, default=8)
    parser.add_argument('--N_decoder_layer', type=int, default=4)
    parser.add_argument('--dropout', type=int, default=0.1)

    parser.add_argument('--code_test', action='store_true')
    parser.add_argument('--warm_up_step', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--save_top_k', type=int, default=3)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    
    parser.add_argument('--wandb_project', type=str, default='NMR-Graph')
    parser.add_argument('--precision', type=str, default='medium',choices=['medium', 'high'])
    parser.add_argument('--monitor', type=str, default='val_acc')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'])
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--early_stop', action='store_true')
    

    args = parser.parse_args()
    print(args)
    main(args)