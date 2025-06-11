import os
from tqdm import tqdm
import numpy as np
import pickle
from itertools import product
from datasets import Dataset, DatasetDict

# Configuration
num_proc = 4
chunk_size = 1024
vocab_type = 'codon'  # 'nucleotide' or 'codon'

class BioTokenizer:
    def __init__(self, vocab_type='codon'):
        self.vocab_type = vocab_type
        self.nucleotides = ['A', 'U', 'C', 'G']
        
        # Base vocabulary
        if vocab_type == 'nucleotide':
            self.vocab = self.nucleotides
        else:
            self.vocab = [''.join(p) for p in product(self.nucleotides, repeat=3)]
        
        # Special tokens (must include all 5 requested)
        self.special_tokens = {
            '<SOS>': "Start of sequence",
            '<EOS>': "End of sequence",
            '<PAD>': "Padding token",
            '<UKN>': "Unknown nucleotide/codon",
            '<MSK>': "Mask token for pretraining"
        }
        
        # Build mappings
        self._build_vocab_mappings()
    
    def _build_vocab_mappings(self):
        """Create token to index and index to token mappings"""
        self.stoi = {s:i for i,s in enumerate(self.vocab)}
        self.itos = {i:s for i,s in enumerate(self.vocab)}
        
        # Add special tokens after regular vocabulary
        offset = len(self.vocab)
        for i, (token, _) in enumerate(self.special_tokens.items(), start=offset):
            self.stoi[token] = i
            self.itos[i] = token
        
        # Set token properties for easy access
        self.sos_token = self.stoi['<SOS>']
        self.eos_token = self.stoi['<EOS>']
        self.pad_token = self.stoi['<PAD>']
        self.ukn_token = self.stoi['<UKN>']
        self.msk_token = self.stoi['<MSK>']
        self.vocab_size = len(self.vocab) + len(self.special_tokens)
    
    def encode(self, sequence, add_special_tokens=True):
        """Encode a biological sequence with proper special tokens"""
        # Split into tokens based on vocabulary type
        if self.vocab_type == 'nucleotide':
            tokens = list(sequence)
        else:
            # Split into codons, handling incomplete ends
            tokens = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
            # Mark incomplete codons as unknown
            tokens = [t if len(t) == 3 else '<UKN>' for t in tokens]
        
        # Convert to IDs, handling unknown tokens
        ids = [self.stoi.get(t, self.ukn_token) for t in tokens]
        
        # Add special tokens if requested
        if add_special_tokens:
            ids = [self.sos_token] + ids + [self.eos_token]
        
        return ids
    
    def decode(self, ids, remove_special_tokens=True):
        """Decode IDs back to biological sequence"""
        tokens = []
        for id in ids:
            token = self.itos.get(id, '<UKN>')
            if remove_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token if token not in self.special_tokens else '')
        return ''.join(tokens)

def load_mrna_and_species(filepath):
    """Load mRNA and species ID data from text file"""
    mrna_list = []
    species_list = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                mrna_list.append(parts[0])
                species_list.append(parts[1])
    return mrna_list, species_list

def build_species_vocab(species_ids):
    unique_species = sorted(set(species_ids))
    species_vocab = {'<UKN>': 0}
    for i, sid in enumerate(unique_species, start=1):
        species_vocab[sid] = i
    return species_vocab
def encode_species_ids(species_ids, species_vocab):
    return [species_vocab.get(sid, species_vocab['<UKN>']) for sid in species_ids]

if __name__ == '__main__':
    # 1. Initialize tokenizer
    tokenizer = BioTokenizer(vocab_type=vocab_type)
    print(f"Initialized tokenizer with {tokenizer.vocab_size} tokens")
    print(f"Special tokens: {tokenizer.special_tokens.keys()}")
    

    # 2. Load mRNA and species data (replace with your data loading logic)
    mrna_sequences,species_ids = load_mrna_and_species('../../../virus_rna_sequence_with_taxonomy.txt')
    print(f"Loaded {len(mrna_sequences)} mRNA sequences")
    # 添加这行来只保留1%的数据
    np.random.seed(42)  # 可以使用任何数字作为种子
    mrna_sequences, species_ids = zip(*[(seq, sid) for seq, sid in zip(mrna_sequences, species_ids) if np.random.random() < 0.01])
    mrna_sequences, species_ids = list(mrna_sequences), list(species_ids)
    print(f"After sampling 1%: {len(mrna_sequences)} mRNA sequences")
    
    
    # Build species vocabulary
    species_vocab = build_species_vocab(species_ids)
    print(f"Species vocabulary size: {len(species_vocab)}")
    # Encode species IDs
    species_encoded = encode_species_ids(species_ids, species_vocab)
        
    # Build dataset
    dataset = Dataset.from_dict({"text": mrna_sequences, "species": species_encoded})

    # Split dataset
    split_dataset = dataset.train_test_split(test_size=0.1, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    # Tokenization function
    def process(example):
        ids = tokenizer.encode(example['text'], add_special_tokens=True)  # (L,)
        species_id = example['species']  # 一个整数
        species_ids = [species_id] * len(ids)
        return {'ids': ids, 'len': len(ids), 'species_ids': species_ids}

    tokenized = split_dataset.map(
        process,
        desc="Tokenizing",
        num_proc=num_proc,
    )

    # Save token IDs and species IDs
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        species_filename = f'species_token_{split}.bin'
        dtype = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32

        print(f"\nWriting {filename} and {species_filename} with {arr_len:,} tokens")

        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        species_arr = np.memmap(species_filename, dtype=np.uint16, mode='w+', shape=(arr_len,))

        total_batches = min(1024, len(dset))  # 分批写入
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {split}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            ids_batch = np.concatenate(batch['ids'])
            species_batch = np.concatenate(batch['species_ids'])

            assert len(ids_batch) == len(species_batch), "token 和 species 对齐错误"
            arr[idx:idx+len(ids_batch)] = ids_batch
            species_arr[idx:idx+len(species_batch)] = species_batch
            idx += len(ids_batch)

        arr.flush()
        species_arr.flush()

    # 8. Save metadata (enhanced version)
    meta = {
        'vocab_size': tokenizer.vocab_size,
        'vocab_type': vocab_type,
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos,
        'special_tokens': {
            'sos': tokenizer.sos_token,
            'eos': tokenizer.eos_token,
            'pad': tokenizer.pad_token,
            'ukn': tokenizer.ukn_token,
            'msk': tokenizer.msk_token,
        },
        'num_sequences': {
            'train': len(split_dataset['train']),
            'val': len(split_dataset['val']),
        },
        'data_source': 'mRNA viral sequences',
        'tokenizer_config': {
            'add_special_tokens': True,
            'handle_incomplete': 'mark_with_ukn'
        },
        'species_vocab': species_vocab,
        'species_vocab_size': len(species_vocab),
        'has_token_level_species': True,

    }
    
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    print("\nProcessing complete. Files created:")
    print(f"- train.bin: Training tokens")
    print(f"- val.bin: Validation tokens")
    print(f"- meta.pkl: Full metadata")
    
    # Verification
    train_tokens = np.sum(tokenized['train']['len'])
    val_tokens = np.sum(tokenized['val']['len'])
    print(f"\nTotal tokens:")
    print(f"Train: {train_tokens:,}")
    print(f"Val: {val_tokens:,}")
    print(f"Vocab size: {tokenizer.vocab_size} ({vocab_type} level)")