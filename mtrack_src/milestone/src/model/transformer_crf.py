
import torch
import torch.nn as nn

from milestone.src.model.module.linear_crf_inferencer import LinearCRF
from milestone.src.model.module.linear_encoder import LinearEncoder
from milestone.src.model.module.transformers_embedder import TransformersEmbedder

from typing import Tuple
from overrides import overrides

from milestone.src.config.utils import START_TAG, STOP_TAG, PAD

class TransformersCRF(nn.Module):

    def __init__(self, config):
        super(TransformersCRF, self).__init__()
        self.embedder = TransformersEmbedder(transformer_model_name=config.embedder_type,
                                             parallel_embedder=config.parallel_embedder)
        self.encoder = LinearEncoder(label_size=config.label_size, input_dim=self.embedder.get_output_dim())
        self.inferencer = LinearCRF(label_size=config.label_size, label2idx=config.label2idx, add_iobes_constraint=config.add_iobes_constraint,
                                    idx2labels=config.idx2labels)

    def decode(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    orig_to_tok_index: torch.Tensor,
                    input_mask,
                    **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Decode from words
        '''
        word_rep = self.embedder(words, orig_to_tok_index, input_mask)
        features = self.encoder(word_rep, word_seq_lens)
        best_scores, decode_idx = self.inferencer.decode(features, word_seq_lens)
        return best_scores, decode_idx

    def decode_emb(self, word_emb, word_seq_lens):
        '''
        Decoding with embedding provided from transformer
        '''
        features = self.encoder(word_emb, word_seq_lens)
        best_scores, decode_idx = self.inferencer.decode(features, word_seq_lens)
        return best_scores, decode_idx
