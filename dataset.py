import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    """Some Information about BilingualDataset"""

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super(BilingualDataset, self).__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor(
            [self.tokenizer_src.token_to_ids(["[SOS]"])], dtype=torch.int64
        )
        self.eos_token = torch.Tensor(
            [self.tokenizer_src.token_to_ids(["[EOS]"])], dtype=torch.int64
        )
        self.pad_token = torch.Tensor(
            [self.tokenizer_src.token_to_ids(["[PAD]"])], dtype=torch.int64
        )

    def __getitem__(self, index):
        src_target_pair = self.df[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        src_pad_size = self.seq_len - len(src_tokens) - 2
        tgt_pad_size = self.seq_len - len(tgt_tokens) - 1

        if src_pad_size < 0 or tgt_pad_size < 0:
            raise ValueError("sentence is too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(src_tokens, dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token] * src_pad_size, dtype=torch.int64),
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(tgt_tokens, dtype=torch.int64),
                torch.Tensor([self.pad_token] * tgt_pad_size, dtype=torch.int64),
            ]
        )

        label = torch.cat(
            [
                torch.Tensor(tgt_tokens, dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token] * tgt_pad_size, dtype=torch.int64),
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & casual_mask(
                decoder_input.size(0)
            ),  # (1,1,seq_len) & (1, seq_len, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

    def __len__(self):
        return len(self.ds)


def casual_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
    return mask == 0
