import numpy as np
import os

def read_bin_file(filepath):
    """读取 .bin 文件并返回其长度"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件 {filepath} 不存在")
    
    # 根据文件大小推断 dtype（与 prepare.py 中的逻辑一致）
    vocab_size = 64  # 假设这是你的词汇表大小（codon模式是64+5=69）
    dtype = np.uint16 if vocab_size < 65536 else np.uint32
    
    # 使用 memmap 读取文件（与写入方式一致）
    arr = np.memmap(filepath, dtype=dtype, mode='r')
    return len(arr)

if __name__ == '__main__':
    # 选择要读取的文件（可以是 'train.bin' 或 'val.bin'）
    bin_file = 'val.bin'  # 修改为你想要读取的文件
    
    try:
        length = read_bin_file(bin_file)
        print(f"文件 {bin_file} 包含 {length:,} 个 tokens")
    except Exception as e:
        print(f"读取文件时出错: {e}")