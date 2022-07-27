# Milestone tagger using MLP-CRF

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from collections import OrderedDict
from typing import List, Union, Tuple
import pickle
import tarfile
from tqdm import tqdm

from milestone.src.model.transformer_crf import TransformersCRF
from milestone.src.config import context_models
from milestone.src.data import TransformersNERDataset


class Milestone_Tagger():
    
    def __init__(self, model_archived_file:str, cuda_device: str = "cpu"):
        """
        model_archived_file: ends with "tar.gz"
        OR
        directly use the model folder patth
        """
        device = torch.device(cuda_device)
        if model_archived_file.endswith("tar.gz"):
            tar = tarfile.open(model_archived_file)
            self.conf = pickle.load(tar.extractfile(tar.getnames()[1])) ## config file
            self.model = TransformersCRF(self.conf)
            self.model.load_state_dict(torch.load(tar.extractfile(tar.getnames()[2]), map_location=device)) ## model file
        else:
            folder_name = model_archived_file
            assert os.path.isdir(folder_name)
            import sys
            sys.path.append('path/to/whiteboard')
            f = open(folder_name + "/config.conf", 'rb')
            self.conf = pickle.load(f)
            f.close()
            self.model = TransformersCRF(self.conf)
            self.model.load_state_dict(torch.load(f"{folder_name}/lstm_crf.m", map_location=device))
        self.conf.device = device
        self.model.to(device)
        self.model.eval()

        print(f"[Data Info] Tokenizing the instances using '{self.conf.embedder_type}' tokenizer")
        self.tokenizer = context_models[self.conf.embedder_type]["tokenizer"].from_pretrained(self.conf.embedder_type)


    def predict(self, sents: List[List[str]], batch_size = -1):
        '''
        Input: [batch_size * sentence(natural language)]
        Output: [Targets for each sentence]
        '''

        batch_size = len(sents) if batch_size == -1 else batch_size

        dataset = TransformersNERDataset(file=None, sents=sents, tokenizer=self.tokenizer, label2idx=self.conf.label2idx, is_train=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

        all_predictions = []
        for batch_id, batch in tqdm(enumerate(loader, 0), desc="--evaluating batch", total=len(loader)):
            one_batch_insts = dataset.insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = self.model.decode(words= batch.input_ids.to(self.conf.device),
                    word_seq_lens = batch.word_seq_len.to(self.conf.device),
                    orig_to_tok_index = batch.orig_to_tok_index.to(self.conf.device),
                    input_mask = batch.attention_mask.to(self.conf.device))

            for idx in range(len(batch_max_ids)):
                length = batch.word_seq_len[idx]
                prediction = batch_max_ids[idx][:length].tolist()
                prediction = prediction[::-1]
                prediction = [self.conf.idx2labels[l] for l in prediction]
                one_batch_insts[idx].prediction = prediction
                all_predictions.append(prediction)
        
        # Now format prediction to have {target_type: word}

        batch_targets = []
        for sent, pred in zip(sents, all_predictions):
            targets = OrderedDict()
            for word, tag in zip(sent, pred):
                if tag != "O":
                    if tag == "S-nav":
                        tag = "nav"
                    elif tag == "S-inter":
                        tag = "inter"

                    if tag not in targets:
                        targets[tag] = [word]
                    else:
                        targets[tag].append(word)
            batch_targets.append(targets)

        return batch_targets
    

if __name__ == '__main__':
    '''
    Debugging purpose
    '''

    test_path = "milestone/test.txt"
    sents = []
    with open(test_path, "r") as f:
        for line in f:
            sents.append(line.split())
    
    model_path = "milestone/bert_frozen_models"
    device = "cpu" # cpu, cuda:0, cuda:1
    ## or model_path = "english_model.tar.gz"
    predictor = Milestone_Tagger(model_path, cuda_device=device)
    predictions = predictor.predict(sents, batch_size=60)
    print(predictions)
