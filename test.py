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
    mult_probs = torch.softmax(pred["multiplet_logits"], dim=-1)

    # 获取标签
    nH_target = target["nH"]
    mult_target = target["multiplet"]

    # 分别更新metric对象
    nH_auc = nH_auc_metric(nH_probs, nH_target)
    mult_auc = mult_auc_metric(mult_probs, mult_target)

    return nH_auc, mult_auc


def main(args):
    model = PeakGraphModule.load_from_checkpoint(args.checkpoint_path)
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
        pred = model(batch)
        nH_auc, mult_auc = compute_auc(pred, batch.masked_node_target, nH_auroc, mult_auroc)
        mult_auroc_list.append(mult_auc)
        nH_auroc_list.append(nH_auc)

    print(f"Mult AUC: {torch.mean(mult_auroc_list):.4f}")
    print(f"NH AUC: {torch.mean(nH_auroc_list):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    main(args)