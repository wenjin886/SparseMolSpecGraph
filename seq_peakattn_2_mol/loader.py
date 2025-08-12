import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast

class NMR2MolDataset(torch.utils.data.Dataset):
    def __init__(self, src_data, tgt_data, src_tokenizer, tgt_tokenizer):
        
        with open(src_data, 'r') as f:
            self.src_data = f.readlines()
            self.src_data = [line.strip() for line in self.src_data]
        
        with open(tgt_data, 'r') as f:
            self.tgt_data = f.readlines()
            self.tgt_data = [line.strip() for line in self.tgt_data]
        
        self.tgt_tokenizer = tgt_tokenizer
        self.src_tokenizer = src_tokenizer
        
        # 检查特殊token是否存在，如果不存在则抛出有意义的错误
        if "1HNMR" not in self.src_tokenizer.vocab:
            raise ValueError("Token '1HNMR' not found in source tokenizer vocabulary")
        if "|" not in self.src_tokenizer.vocab:
            raise ValueError("Token '|' not found in source tokenizer vocabulary")
        self.nmr_split_token = self.src_tokenizer.vocab["1HNMR"]
        self.peak_split_token = self.src_tokenizer.vocab["|"]

    def __len__(self):
        return len(self.tgt_data)
    
    def __getitem__(self, idx):
        tgt_ids = torch.tensor(self.tgt_tokenizer.encode(self.tgt_data[idx]))

        src_ids = torch.tensor(self.src_tokenizer.encode(self.src_data[idx]))
        # process src by splitting peaks
        nmr_split_token_idx = torch.where(src_ids == self.nmr_split_token)[0]
        formula_ids = src_ids[1:nmr_split_token_idx] # remove <bos>

        peaks_ids = src_ids[nmr_split_token_idx+1:]
        peak_split_token_idx = torch.where(peaks_ids == self.peak_split_token)[0]
        peaks_list = []
        for i, idx in enumerate(peak_split_token_idx):
            if i == 0:
                print(i, idx)
                peaks_list.append(peaks_ids[:idx])
            else:
                peaks_list.append(peaks_ids[peak_split_token_idx[i-1]+1:idx]) # already remove <eos>
        peaks_ids = pad_sequence(peaks_list, batch_first=True, padding_value=0)
        
        return {"tgt_ids": tgt_ids, "formula_ids": formula_ids,  "peaks_ids": peaks_ids, "peak_num":peaks_ids.shape[0]}

def collate_fn(batch):
    
    print("processing tgt")
    tgt_ids = [torch.tensor(item["tgt_ids"], dtype=torch.long) for item in batch]
    tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=0)
    tgt_mask = (tgt_ids != 0).float()
    
    print("processing formula")
    formula_ids = [torch.tensor(item["formula_ids"], dtype=torch.long) for item in batch]
    formula_ids = pad_sequence(formula_ids, batch_first=True, padding_value=0)
    formula_mask = (formula_ids != 0).float()


    print("processing peaks")
    peaks_ids = [torch.tensor(item["peaks_ids"], dtype=torch.long) for item in batch]
    peaks_ids = pad_sequence(peaks_ids, batch_first=True, padding_value=0)
    peaks_mask = (peaks_ids != 0).float()
    
    return {
        "tgt_ids": tgt_ids,
        "tgt_mask": tgt_mask,

        "formula_ids": formula_ids,
        "formula_mask": formula_mask,

        "peaks_ids": peaks_ids,
        "peaks_mask": peaks_mask,
        
    }

def create_nmr2mol_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )



if __name__ == "__main__":
    src_data = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/h_nmr/data/src-val.txt"
    tgt_data = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/h_nmr/data/tgt-val.txt"
    src_tokenizer = PreTrainedTokenizerFast(tokenizer_file="../seq2mol_code/tokenizer/src_tokenizer/nmr_formula_tokenizer_fast/tokenizer.json")
    tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file="../seq2mol_code/tokenizer/tgt_tokenizer/smiles_tokenizer_fast/tokenizer.json")
    dataset = NMR2MolDataset(src_data, tgt_data, src_tokenizer, tgt_tokenizer)
    # print(dataset[0])
    print('--------------------------------')
    dataloader = create_nmr2mol_dataloader(dataset, batch_size=2, shuffle=False)
    for batch in dataloader:
        print(batch)
        break