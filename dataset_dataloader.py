import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
import os
import re
import json
from tqdm import tqdm

import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors
from transformers import PreTrainedTokenizerFast
from smiles_tokenizer import split_smiles
from formula_tokenizer import split_formula

def fully_connected_edge_index(num_nodes):
    # 生成 [0, 0, ..., 1, 1, ..., n-1, n-1]
    row = torch.arange(num_nodes).repeat_interleave(num_nodes)
    # 生成 [0, 1, ..., n-1, 0, 1, ..., n-1]
    col = torch.arange(num_nodes).repeat(num_nodes)
    # 去除 self-loop（如果不想保留的话）
    mask = row != col
    return torch.stack([row[mask], col[mask]], dim=0)

def single_spectrum_to_graph(peaks, id, smiles):
    """
    将一个 NMR 谱图（多个峰）转换为 PyG 图
    peaks: list[dict]，每个峰的信息
    category_list: 所有可能的 category（用于one-hot）
    j_threshold: 耦合值相等判断的容差
    """
    num_peaks = len(peaks)
    

    centroids = []
    peak_widths = []
    multiplets = []
    nH = []
    # 1. 构造节点特征
    for peak in peaks:
        centroids.append(float(peak['centroid']))
        peak_widths.append(float(peak['rangeMax'])-float(peak['rangeMin']))
        multiplets.append(MULTIPLETS[peak['category']])
        nH.append(NUM_H[peak['nH']])

    # 2. 构造边（基于是否有相同的 J 耦合值）
    edge_index = fully_connected_edge_index(num_peaks)
    # print(edge_index)
    edge_attr = [ float(centroids[i] - centroids[j]) for i, j in zip(edge_index[0], edge_index[1])] # 有向图
    # print(edge_attr)

    data = Data(id=id, smiles=smiles, num_nodes=num_peaks,
                centroids=torch.tensor(centroids), 
                peak_widths=torch.tensor(peak_widths), 
                multiplets=torch.tensor(multiplets), 
                nH=torch.tensor(nH),
                edge_index=edge_index, 
                edge_attr=torch.tensor(edge_attr))
    return data

def create_masked_data(data, mask_ratio=0.1):
    """
    为每个图随机mask掉一些节点(保存了边的信息)，用于自监督学习
    Args:
        data: PyG Data对象
        mask_ratio: mask比例，默认10%
    Returns:
        masked_data: 包含mask信息的Data对象
    """
    num_nodes = data.num_nodes
    
    # 随机选择要mask的节点
    num_masked = max(1, int(num_nodes * mask_ratio))  # 至少mask 1个节点
    masked_indices = torch.randperm(num_nodes)[:num_masked]
    
    # 创建mask标记
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[masked_indices] = True
    
    # 保存原始特征作为目标
    masked_target = {
        "centroid": torch.tensor([data.centroids[i] for i in masked_indices], dtype=torch.float),
        "width": torch.tensor([data.peak_widths[i] for i in masked_indices], dtype=torch.float),
        "nH": torch.tensor([data.nH[i] for i in masked_indices], dtype=torch.long),
        "multiplet": torch.tensor([data.multiplets[i] for i in masked_indices], dtype=torch.long)
    }
    
    # 将masked节点的特征替换为特殊值（比如0）
    masked_centroids = data.centroids.clone()
    # print('masked_centroids', masked_centroids)
    masked_widths = data.peak_widths.clone()
    masked_nH = data.nH.clone()
    masked_multiplets = data.multiplets.clone()
    
    for idx in masked_indices:
        masked_centroids[idx] = 0.0  # 或者用其他特殊值
        masked_widths[idx] = 0.0
        masked_nH[idx] = 0.0
        masked_multiplets[idx] = 0  # 假设0是特殊类别
    
    # print('masked_centroids', type(masked_centroids))
    # 创建新的Data对象
    masked_data = Data(
        id=data.id,
        smiles=data.smiles,
        num_nodes=data.num_nodes,
        centroids=masked_centroids,
        peak_widths=masked_widths,
        nH=masked_nH,
        multiplets=masked_multiplets,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        masked_node_index=masked_indices,  # 被mask的节点索引
        masked_node_target=masked_target,  # 被mask节点的真实值
        mask=mask  # 整个mask标记
    )
    
    return masked_data

def process_origin_data(data_dir, save_dir, dataset_name):
    """
    Now just for H NMR spectra
    """
    file_list = os.listdir(data_dir)
    file_list.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    # file_list = file_list[:2]
    
    data_df = pd.DataFrame()
    # get dataset info
    MULTIPLETS_LIST = []
    NUM_H_LIST = []
    
    all_centroids = []
    all_peak_widths = []
    num_peaks = []

    for file in tqdm(file_list):
        data_i = pd.read_parquet(os.path.join(data_dir, file))
        print(data_i)
        data_df = pd.concat([data_df, data_i[['h_nmr_peaks', 'smiles']]])
        

        for id, row in tqdm(data_i.iterrows(), total=len(data_i)):
            num_peaks.append(len(row['h_nmr_peaks']))
            for peak in row['h_nmr_peaks']:
                if peak['category'] not in MULTIPLETS_LIST:
                    MULTIPLETS_LIST.append(peak['category'])
                if peak['nH'] not in NUM_H_LIST:
                    NUM_H_LIST.append(peak['nH'])
                all_centroids.append(float(peak['centroid']))
                all_peak_widths.append(float(peak['rangeMax'])-float(peak['rangeMin']))
    print(data_df)

    MULTIPLETS_LIST = sorted(MULTIPLETS_LIST, key=lambda x: (len(x), x))
    global MULTIPLETS
    MULTIPLETS = {k: i for i, k in enumerate(MULTIPLETS_LIST)}

    NUM_H_LIST = sorted(NUM_H_LIST, key=lambda x: x)
    global NUM_H
    NUM_H = {k: i for i, k in enumerate(NUM_H_LIST)}
    
    print(MULTIPLETS)

    centroid_mean, centroid_std = np.mean(all_centroids), np.std(all_centroids)
    width_mean, width_std = np.mean(all_peak_widths), np.std(all_peak_widths)
    num_peaks_mean, num_peaks_std = np.mean(num_peaks), np.std(num_peaks)
    dataset_info = {
        "MULTIPLETS": MULTIPLETS,
        "NUM_H": NUM_H,
        "centroid": {"max": float(max(all_centroids)), "min": float(min(all_centroids)), "mean": float(centroid_mean), "std": float(centroid_std)},
        "width": {"max": float(max(all_peak_widths)), "min": float(min(all_peak_widths)), "mean": float(width_mean), "std": float(width_std)},
        "num_peaks": {"max": float(max(num_peaks)), "min": float(min(num_peaks)), "mean": float(num_peaks_mean), "std": float(num_peaks_std)}
    }
    with open(os.path.join(save_dir, dataset_name+'.json'), "w") as f:
        json.dump(dataset_info, f, indent=4)

    h_nmr_dataset = []
    masked_hnmr_dataset = []
    for id, row in data_df.iterrows():
        data_i = single_spectrum_to_graph(row['h_nmr_peaks'], id, row['smiles'])
        h_nmr_dataset.append(data_i)
        masked_hnmr_dataset.append(create_masked_data(data_i))
    print(len(h_nmr_dataset))
    torch.save(h_nmr_dataset, os.path.join(save_dir, dataset_name+'.pt'))
    torch.save(masked_hnmr_dataset, os.path.join(save_dir, dataset_name+'_masked.pt'))



def processing_hnmr_cls_label(dataset_path, dataset_info_path,
                              mult_map_info_path, nh_map_info_path, save_path):
    
    print(dataset_info_path)
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
    MULTIPLETS_id2type, NUM_H_id2type = {}, {}  
    for type, id in dataset_info['MULTIPLETS'].items():
        MULTIPLETS_id2type[int(id)] = type
    for type, id in dataset_info['NUM_H'].items():
        NUM_H_id2type[int(id)] = type
    print("MULTIPLETS_id2type", MULTIPLETS_id2type)
    print("NUM_H_id2type", NUM_H_id2type)

    with open(mult_map_info_path, "r") as f:
        mult_map_info = json.load(f)
        multiplet_mapping = mult_map_info["multiplet_mapping"]
        mult_label_to_id = mult_map_info['multiplet_label_to_id']

    with open(nh_map_info_path, "r") as f:
        nh_map_info = json.load(f)
        nh_mapping = nh_map_info["nh_mapping"]
        nh_label_to_id = nh_map_info['nh_label_to_id']
    print("mult_label_to_id", mult_label_to_id)
    print("nh_label_to_id", nh_label_to_id)

    print("Loading dataset")
    dataset = torch.load(dataset_path) 
    mapped_dataset = []
    # Process multiplet label
    for data in tqdm(dataset, total=len(dataset)):
        try:
            data.multiplets = torch.tensor([mult_label_to_id[multiplet_mapping[MULTIPLETS_id2type[int(id)]]] for id in data.multiplets])
            data.nH = torch.tensor([nh_label_to_id[nh_mapping[NUM_H_id2type[int(id)]]] for id in data.nH])
        except Exception as e:
            print(e)
            print("data.multiplets", data.multiplets)
            print("data.nH", data.nH)
            break
        mapped_dataset.append(data)
    
    torch.save(mapped_dataset, os.path.join(save_path, "h_nmr_label_mapper.pt"))
    
def generate_masked_node_dataset(unmasked_dataset_path, save_name):
    unmaksed_dataset = torch.load(unmasked_dataset_path)
    masked_dataset = []
    for data_i in tqdm(unmaksed_dataset, total=len(unmaksed_dataset)):
        masked_dataset.append(create_masked_data(data_i))
    
    torch.save(masked_dataset, os.path.join(os.path.dirname(unmasked_dataset_path), save_name))

def generate_formula_dataset(dataset_path, save_name):
    dataset = torch.load(dataset_path)
    formula_dataset = []
    for data_i in tqdm(dataset, total=len(dataset)):
        formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(data_i.smiles))
        data_i.formula = formula
        formula_dataset.append(data_i)
    torch.save(formula_dataset, os.path.join(os.path.dirname(dataset_path), save_name))
    

def generate_smiles_ids_and_formula_ids_dataset(unmasked_dataset_path, smiles_tokenizer_path, formula_tokenizer_path, save_name):
    unmaksed_dataset = torch.load(unmasked_dataset_path)
    smiles_tokenizer = PreTrainedTokenizerFast(tokenizer_file=smiles_tokenizer_path,
                                        bos_token="[BOS]",
                                        eos_token="[EOS]",
                                        pad_token="[PAD]",
                                        unk_token="[UNK]",
                                        padding_side="right" )
    formula_tokenizer = PreTrainedTokenizerFast(tokenizer_file=formula_tokenizer_path,
                                        bos_token="[BOS]",
                                        eos_token="[EOS]",
                                        pad_token="[PAD]",
                                        unk_token="[UNK]",
                                        padding_side="right" )
    smiles_dataset = []
    for data_i in tqdm(unmaksed_dataset, total=len(unmaksed_dataset)):
        
        splitted_smiles = " ".join(split_smiles(data_i.smiles))
        splitted_formula = " ".join(split_formula(data_i.formula))

        data_i.smiles_ids = smiles_tokenizer.encode(splitted_smiles)
        data_i.smiles_len = len(data_i.smiles_ids)

        data_i.formula_ids = formula_tokenizer.encode(splitted_formula)
        data_i.formula_len = len(data_i.formula_ids)

        smiles_dataset.append(data_i)

    torch.save(smiles_dataset, os.path.join(os.path.dirname(unmasked_dataset_path), save_name))


if __name__ == "__main__":

    # data_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/Dataset/multimodal_spectroscopic_dataset"
    # save_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/Dataset/h_nmr"
    # process_origin_data(data_dir, save_dir, 'h_nmr')

    # dir_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/Dataset/h_nmr"
    # dataset_path = os.path.join(dir_path, "h_nmr.pt")
    # dataset_info_path = os.path.join(dir_path, "h_nmr.json")
    # mult_map_info_path = os.path.join(dir_path, "multiplet_mapping.json")
    # nh_map_info_path = os.path.join(dir_path, "nh_mapping.json")
    # processing_hnmr_cls_label(dataset_path, dataset_info_path,
    #                           mult_map_info_path, nh_map_info_path, save_path=dir_path)

    # dataset_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/Dataset/h_nmr/h_nmr_label_mapped.pt"
    # generate_masked_node_dataset(dataset_path, "masked_h_nmr_label_mapped.pt")

    # dataset_path = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/example_hnmr.pt"
    # generate_formula_dataset(dataset_path, "example_hnmr_with_formula.pt")
    
    smiles_tokenizer_path = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/smiles_tokenizer_fast/tokenizer.json"
    formula_tokenizer_path = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/formula_tokenizer_fast/tokenizer.json"
    dataset_path = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/example_hnmr_with_formula.pt"
    generate_smiles_ids_and_formula_ids_dataset(dataset_path, smiles_tokenizer_path, formula_tokenizer_path, "hnmr_with_smiles_and_formula_ids.pt")
    

