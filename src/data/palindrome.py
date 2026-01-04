import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# --- Constants ---
VOCAB_SIZE = 10
PAD_TOKEN = 10
DEV_SIZE = 30_000 
TEST_SIZE = 10_000

# Sequence lengths
DEV_SEQ_RANGE = (50, 100)
TEST_SEQ_RANGE = (300, 500) # Extrapolation

class PalindromeDataset(Dataset):
    """
    A synthetic PyTorch Dataset for binary classification of palindromes.
    
    This dataset generates a balanced mix (50/50) of palindromic (label 1) 
    and non-palindromic (label 0) sequences of integers.
    """
    def __init__(self, size: int, seq_range: tuple, seed: int):
        """
        Args:
            size (int): Total number of samples to generate.
            seq_range (tuple[int, int]): A tuple containing (min_len, max_len) 
                                         for the generated sequences.
            seed (int): Random seed for reproducibility.
        """
        self.min_len, self.max_len = seq_range
        self.rng = np.random.default_rng(seed)
        self.data = []
        
        n_pos = size // 2
        n_neg = size - n_pos
        
        # --- Generate Positives ---
        for _ in range(n_pos):
            length = self.rng.integers(self.min_len, self.max_len + 1)
            is_odd = (length % 2 == 1)
            half_len = length // 2
            half_seq = self.rng.integers(0, VOCAB_SIZE, size=half_len)
            
            if is_odd:
                mid_char = self.rng.integers(0, VOCAB_SIZE, size=1)
                seq = np.concatenate([half_seq, mid_char, half_seq[::-1]])
            else:
                seq = np.concatenate([half_seq, half_seq[::-1]])
            
            self.data.append((torch.from_numpy(seq).long(), torch.tensor(1, dtype=torch.long)))

        # --- Generate Negatives ---
        count = 0
        while count < n_neg:
            length = self.rng.integers(self.min_len, self.max_len + 1)
            cand = self.rng.integers(0, VOCAB_SIZE, size=length)
            if np.array_equal(cand, cand[::-1]): continue
            self.data.append((torch.from_numpy(cand).long(), torch.tensor(0, dtype=torch.long)))
            count += 1
            
        self.rng.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (sequence, label)
                - sequence (torch.LongTensor): Shape (seq_len,) containing integers [0, VOCAB_SIZE).
                - label (torch.LongTensor): Scalar tensor, 1 for palindrome, 0 otherwise.
        """
        return self.data[idx]

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in a batch.
    
    Args:
        batch (list): A list of tuples (sequence, label) from the dataset.

    Returns:
        tuple: (inputs_padded, targets, lengths)
            - inputs_padded (torch.LongTensor): Shape (batch_size, max_seq_len). 
              Padded with PAD_TOKEN.
            - targets (torch.LongTensor): Shape (batch_size,). Class labels.
            - lengths (torch.Tensor): Shape (batch_size,). Original lengths of the sequences 
              before padding (useful for masking or packing sequences).
    """
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in inputs])
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN)
    targets = torch.stack(targets)
    return inputs_padded, targets, lengths

def get_base_datasets(seed=100):
    """
    Factory function to generate the standard Development and Test datasets.
    
    Uses global constants DEV_SIZE/DEV_SEQ_RANGE and TEST_SIZE/TEST_SEQ_RANGE.
    
    Args:
        seed (int): Base random seed. The test set uses seed + 100.
        
    Returns:
        tuple: (dev_ds, test_ds) where both are instances of PalindromeDataset.
    """
    dev_ds = PalindromeDataset(DEV_SIZE, DEV_SEQ_RANGE, seed=seed)
    test_ds = PalindromeDataset(TEST_SIZE, TEST_SEQ_RANGE, seed=seed+100)
    return dev_ds, test_ds