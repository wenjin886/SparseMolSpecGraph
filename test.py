import torch
from torch_geometric.loader import DataLoader
from torchmetrics.classification import MulticlassAUROC
import json
import argparse
import os.path as osp
from tqdm import tqdm
from model import PeakGraphModule

def compute_auc(pred, target, nH_auc_metric, mult_auc_metric):
    # 转为概率（softmax）
    nH_probs = torch.softmax(pred["nH"], dim=-1)
    print(torch.bincount(torch.argmax(nH_probs, dim=-1)))
    
    mult_probs = torch.softmax(pred["multiplet_logits"], dim=-1)
    print(torch.bincount(torch.argmax(mult_probs, dim=-1)))


    # 获取标签
    nH_target = target["nH"]
    mult_target = target["multiplet"]

    # 分别更新metric对象
    nH_auc = nH_auc_metric(nH_probs, nH_target)
    mult_auc = mult_auc_metric(mult_probs, mult_target)

    return nH_auc, mult_auc


def main(args):
    # device = torch.device("cuda")
    device = torch.device("cpu")
    print("Loading model checkpoint...")
    model = PeakGraphModule.load_from_checkpoint(args.checkpoint_path)
    model.to(device)

    print("Loading test data...")
    test_set = torch.load(osp.join(args.dataset_path, "test_set.pt"))
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    with open(args.dataset_info_path, "r") as f:
        dataset_info = json.load(f)

    if args.spec_type == "h_nmr":
        MULTIPLETS = dataset_info["MULTIPLETS"]
        NUM_H = dataset_info["NUM_H"]
        mult_auroc = MulticlassAUROC(len(MULTIPLETS), average="macro")
        nH_auroc = MulticlassAUROC(len(NUM_H), average="macro")

    mult_auroc_list = []
    nH_auroc_list = []
    for batch in tqdm(test_dataloader, desc="Testing", total=len(test_dataloader)):
        pred = model(batch.to(device))
        nH_auc, mult_auc = compute_auc(pred, batch.masked_node_target, nH_auroc, mult_auroc)
        mult_auroc_list.append(mult_auc)
        nH_auroc_list.append(nH_auc)
        break

    print(f"Mult AUC: {torch.mean(torch.tensor(mult_auroc_list)):.4f}")
    print(f"NH AUC: {torch.mean(torch.tensor(nH_auroc_list)):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument('--dataset_info_path', type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument('--spec_type', type=str, choices=['h_nmr', 'c_nmr', 'ms'], default='h_nmr')
    args = parser.parse_args()
    main(args)