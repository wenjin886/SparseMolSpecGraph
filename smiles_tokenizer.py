#Experimental Class for Smiles Enumeration, Iterator and SmilesIterator adapted from Keras 1.2.2
from rdkit import Chem
import numpy as np
import threading
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast
import re
import os.path as osp
import torch






def split_smiles(smile: str) :
    pattern_full = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    regex = re.compile(pattern_full)
    tokens = [token for token in regex.findall(smile)]

    if smile != "".join(tokens):
        print(smile)
        raise ValueError(
            "Tokenised smiles does not match original: {} {}".format(tokens, smile)
        )

    return tokens


def get_training_corpus(smiles_data):
    for i in range(len(smiles_data)):
        smiles = smiles_data[i]
        
        smiles = " ".join(split_smiles(smiles))
        yield smiles

def load_smiles_data(dataset_path):
    data = torch.load(dataset_path)
    smiles_data = [data_i.smiles for data_i in data]
    return smiles_data

def main(dataset_path, tokenizer_save_path):      
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    special_tokens = [ "[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    trainer = trainers.WordLevelTrainer(special_tokens=special_tokens)
    smiles_data = load_smiles_data(dataset_path)
    tokenizer.train_from_iterator(get_training_corpus(smiles_data),trainer=trainer)

    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    print(bos_token_id, eos_token_id)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[BOS]:0 $A:0 [EOS]:0",
        # pair=f"[BOS]:0 $A:0 [EOS]:0 [BOS]:1 $B:1 [EOS]:1",
        special_tokens=[("[BOS]", bos_token_id), ("[EOS]", eos_token_id)]
    )

    tokenizer.save(osp.join(tokenizer_save_path, "smiles_tokenizer.json"))

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token = "[UNK]",
        pad_token = "[PAD]",
        bos_token = "[BOS]",
        eos_token = "[EOS]"
    )
    wrapped_tokenizer.save_pretrained(osp.join(tokenizer_save_path, 'smiles_tokenizer_fast'))

if __name__ == "__main__":
    dataset_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/Dataset/h_nmr/h_nmr.pt"
    main(dataset_path=dataset_path, tokenizer_save_path=osp.dirname(dataset_path))
