#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: basecall.py
@Description: description of this file
@Datatime: 2026/01/08 13:46:08
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

import re
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from ont_fast5_api.fast5_interface import get_fast5_file
# import nanopore_signal_tokenizer as nst
import torch
from typing import Dict, Optional, Tuple, Union, Literal, List
from nanopore_signal_tokenizer import VQTokenizer
from nanopore_signal_tokenizer import KmeansTokenizer



def get_overlap_hat(
    chunk_size: int, 
    window_len: int, 
    stride: int, 
    suppose_overlap_len: int = 500
) -> Tuple[int, int, int]:
    """
    Calculate the overlap needed for chunking.
    给定chunk_size, window_len, stride, 计算实际的overlap长度，以及前后chunk交界处的window索引。
    这里实际的overlap长度在suppose_overlap_len附近波动，以保证前后两个chunk重叠的区域的window是对齐的。
    Args:
        chunk_size (int): Size of each chunk.
        window_len (int): Length of each window.
        stride (int): Stride between windows.
        suppose_overlap_len (int, optional): Expected overlap length. Defaults to 500.
    Returns:
        Tuple[int, int, int]: Actual overlap length, previous chunk end window index, next chunk start window index.
    """
    tail_len = (chunk_size - window_len) % stride
    n_window_hat = (suppose_overlap_len - tail_len - window_len) // stride + 1 # 两个相邻chunk重叠区域的window数量
    overlap_hat = tail_len + window_len + stride * (n_window_hat - 1)

    total_window_num_in_a_chunk = (chunk_size - window_len) // stride + 1
    assert n_window_hat < total_window_num_in_a_chunk, "Overlap is too large for the given chunk size."

    #前一个chunk的结尾所在的位置 1-based
    pre_end = chunk_size - overlap_hat + window_len + stride * (n_window_hat//2 - 1)
    #前一个chunk的结尾所在的window编号 0-based
    pre_end_w_idx = (pre_end - window_len) // stride
    #后一个chunk的开头所在的window编号  0-based
    post_start_w_idx = n_window_hat // 2

    return total_window_num_in_a_chunk, overlap_hat, pre_end_w_idx, post_start_w_idx




def fast5_to_word(
    model_path: str,
    fast5_files_dir: str,
    method: Literal['kmeans', 'vq'] = 'kmeans',
    kwargs_kmeans: Optional[Dict] = None,
    kwargs_vq: Optional[Dict] = None,
    output_dir: str = './',
):
    """
    Convert raw signals from fast5 files to word sequences using a trained tokenizer model.
    
    Args:
        window_size (int): Size of the sliding window.
        stride (int): Stride of the sliding window.
        model_path (str): Path to the trained tokenizer model.
        fast5_files_dir (str): Directory containing fast5 files.
        method (Literal['kmeans', 'vq'], optional): Tokenization method. Defaults to 'kmeans'.
    """
    if method == 'kmeans':
        required = {'window_size', 'stride'}
        if not required.issubset(kwargs_kmeans):
            raise ValueError("kwargs_kmeans must contain 'window_size' and 'stride'.")
        tokenizer = KmeansTokenizer(
            window_size=window_size, 
            stride=stride,
            centroids_path=model_path
        )
    elif method == 'vq':
        tokenizer = VQTokenizer(
            model_ckpt=model_path,  # 必填：预训练模型路径
            device="cuda",                                # 可选：设备，默认自动选 cuda/cpu
            token_batch_size=100                          # 可选：批处理 token 的内部 batch size, 也就是每次同时转换多少个token
        )
    else:
        raise ValueError("Method must be either 'kmeans' or 'vq'.")

    fast5_files = glob.glob(fast5_files_dir + '/*.fast5')
    for fast5_file in tqdm(fast5_files, desc="Processing fast5 files"):
        
        tokenizer.tokenize_fast5(
            fast5_path=fast5_file,
            output_path=f"{output_dir}/sample.json.gz"
    ) 