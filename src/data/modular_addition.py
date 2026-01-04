import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# --- Constants ---
MODULUS = 31
VOCAB_SIZE = 32 # 0-30 are numbers, 31 is PAD
PAD_TOKEN = 31

# Sizes
DEV_SIZE = 30_000 
TEST_SIZE = 10_000

# Sequence lengths
DEV_SEQ_RANGE = (2, 20)
TEST_SEQ_RANGE = (21, 40) # Extrapolation

class ModularDataset(Dataset):
    """
    A sequence-to-sequence dataset for modular arithmetic.
    
    Task: Given a sequence of integers, predict the cumulative sum modulo N 
    at every step.
    
    This dataset ensures a balanced distribution of sequence lengths
    across the specified range.
    """
    def __init__(self, size: int, seq_range: tuple, modulus: int, seed: int):
        """
        Args:
            size (int): Total number of samples.
            seq_range (tuple[int, int]): (min_len, max_len).
            modulus (int): The modulus N for the arithmetic (e.g., 31).
            seed (int): Random seed.
        """
        self.modulus = modulus
        self.min_len, self.max_len = seq_range
        self.rng = np.random.default_rng(seed)
        self.data = []
        
        # Calculate samples per length to ensure balance
        n_lengths = self.max_len - self.min_len + 1
        base_count = size // n_lengths
        remainder = size % n_lengths
        
        for i, length in enumerate(range(self.min_len, self.max_len + 1)):
            # Distribute remainder among the first few lengths
            count = base_count + (1 if i < remainder else 0)
            
            if count > 0:
                # Generate random inputs: (count, length)
                nums_matrix = self.rng.integers(0, self.modulus, size=(count, length))
                
                # Calculate targets: Cumulative sum modulo N
                # axis=1 performs the sum across the sequence dimension
                targets = np.cumsum(nums_matrix, axis=1) % self.modulus
                
                for r in range(count):
                    self.data.append((
                        torch.from_numpy(nums_matrix[r]).long(), 
                        torch.from_numpy(targets[r]).long()
                    ))
        
        self.rng.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (sequence, target)
                - sequence (torch.LongTensor): Shape (seq_len,). Input integers.
                - target (torch.LongTensor): Shape (seq_len,). Cumulative sums.
        """
        return self.data[idx]

def collate_fn(batch):
    """
    Prepares a batch for sequence-to-sequence training.
    
    Args:
        batch (list): List of tuples (input_seq, target_seq).

    Returns:
        tuple: (inputs_padded, targets_padded, lengths)
            - inputs_padded (torch.LongTensor): Shape (batch_size, max_seq_len). 
              Padded with PAD_TOKEN.
            - targets_padded (torch.LongTensor): Shape (batch_size, max_seq_len). 
              Padded with -100 (standard PyTorch ignore_index).
            - lengths (torch.Tensor): Shape (batch_size,). Original lengths.
    """
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in inputs])
    
    # Pad inputs with the specific PAD_TOKEN
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN)
    
    # Pad targets with -100 (standard PyTorch ignore_index) so we don't compute loss on pads
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-100)
    
    return inputs_padded, targets_padded, lengths

def get_base_datasets(seed=100):
    """
    Factory function to generate the Modular arithmetic datasets.
    
    Returns:
        tuple: (dev_ds, test_ds)
            - dev_ds: Standard length range (2, 20).
            - test_ds: Extrapolation length range (21, 40).
    """
    dev_ds = ModularDataset(
        size=DEV_SIZE, 
        seq_range=DEV_SEQ_RANGE, 
        modulus=MODULUS, 
        seed=seed
    )
    
    test_ds = ModularDataset(
        size=TEST_SIZE, 
        seq_range=TEST_SEQ_RANGE, 
        modulus=MODULUS, 
        seed=seed + 100
    )
    
    return dev_ds, test_ds