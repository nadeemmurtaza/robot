"""
Tokenization module for Python code.
Supports:
1. Fast, zero-dependency Byte-Level Code Tokenizer (exact UTF-8 byte reproduction, vocab size 260)
2. HuggingFace BPE Tokenizer (e.g., GPT-2 or CodeGen)
"""
from typing import List, Union
import torch

class ByteCodeTokenizer:
    """
    Byte-Level Tokenizer for code.
    Maps UTF-8 bytes (0-255) to token IDs (4-259).
    IDs 0-3 are reserved for special tokens:
        0: <pad>
        1: <bos>
        2: <eos>
        3: <unk>
    """
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    NUM_SPECIAL = 4
    VOCAB_SIZE = 256 + NUM_SPECIAL # 260
    
    def __init__(self):
        self.vocab_size = self.VOCAB_SIZE
        self.pad_token_id = self.PAD_ID
        self.bos_token_id = self.BOS_ID
        self.eos_token_id = self.EOS_ID
        self.unk_token_id = self.UNK_ID

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        raw_bytes = list(text.encode("utf-8", errors="replace"))
        ids = [b + self.NUM_SPECIAL for b in raw_bytes]
        if add_special_tokens:
            ids = [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        
        filtered_bytes = []
        for token_id in ids:
            if token_id < self.NUM_SPECIAL:
                if not skip_special_tokens:
                    if token_id == self.PAD_ID:
                        filtered_bytes.extend(self.PAD_TOKEN.encode('utf-8'))
                    elif token_id == self.BOS_ID:
                        filtered_bytes.extend(self.BOS_TOKEN.encode('utf-8'))
                    elif token_id == self.EOS_ID:
                        filtered_bytes.extend(self.EOS_TOKEN.encode('utf-8'))
                    else:
                        filtered_bytes.extend(self.UNK_TOKEN.encode('utf-8'))
            else:
                byte_val = token_id - self.NUM_SPECIAL
                if 0 <= byte_val <= 255:
                    filtered_bytes.append(byte_val)
                    
        return bytes(filtered_bytes).decode("utf-8", errors="replace")


class HFCodeTokenizer:
    """
    Wrapper around HuggingFace AutoTokenizer (e.g. gpt2) for code.
    """
    def __init__(self, model_id: str = "gpt2"):
        from transformers import AutoTokenizer
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        self.vocab_size = len(self.hf_tokenizer)
        self.pad_token_id = self.hf_tokenizer.pad_token_id
        self.bos_token_id = self.hf_tokenizer.bos_token_id or self.hf_tokenizer.eos_token_id
        self.eos_token_id = self.hf_tokenizer.eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return self.hf_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
