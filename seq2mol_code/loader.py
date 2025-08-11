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
        
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.tgt_data)
    
    def __getitem__(self, idx):
        src_ids = self.src_tokenizer.encode(self.src_data[idx])
        tgt_ids = self.tgt_tokenizer.encode(self.tgt_data[idx])
        return {"src_ids": src_ids, "tgt_ids": tgt_ids}

def create_nmr2mol_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    def collate_fn(batch):
        src_ids = [torch.tensor(item["src_ids"], dtype=torch.long) for item in batch]
        tgt_ids = [torch.tensor(item["tgt_ids"], dtype=torch.long) for item in batch]
        src_ids = pad_sequence(src_ids, batch_first=True, padding_value=0)
        tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=0)
        src_mask = (src_ids != 0).float()
        tgt_mask = (tgt_ids != 0).float()
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask
        }

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

class NMR2MolGraphDataset(torch.utils.data.Dataset):
    def __init__(self, src_data, tgt_data, src_tokenizer, tgt_tokenizer):
        
        with open(src_data, 'r') as f:
            self.src_data = f.readlines()
            self.src_data = [line.strip() for line in self.src_data]
        
        with open(tgt_data, 'r') as f:
            self.tgt_data = f.readlines()
            self.tgt_data = [line.strip() for line in self.tgt_data]
        
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.tgt_data)
    
    def __getitem__(self, idx):
        src_ids = self.src_tokenizer.encode(self.src_data[idx])
        tgt_ids = self.tgt_tokenizer.encode(self.tgt_data[idx])
        return {"src_ids": src_ids, "tgt_ids": tgt_ids}

if __name__ == "__main__":
    src_data = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/h_nmr/data/src-val.txt"
    tgt_data = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/h_nmr/data/tgt-val.txt"
    src_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/src_tokenizer/smiles_tokenizer_fast/tokenizer.json")
    tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/tgt_tokenizer/smiles_tokenizer_fast/tokenizer.json")
    dataset = NMR2MolDataset(src_data, tgt_data, src_tokenizer, tgt_tokenizer)
    print(dataset[0])
    print('--------------------------------')
    dataloader = create_nmr2mol_dataloader(dataset, batch_size=2, shuffle=False)
    for batch in dataloader:
        print(batch)
        break