import json
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from aistudio_common.utils import env_utils
from FlagEmbedding import BGEM3FlagModel, FlagModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 假设 embeddings 是形状为 (n, d) 的torch张量，其中 n 是句子数量，d 是嵌入维度
# embeddings = [embedding1, embedding2, ..., embedding100]


def calculate_similarity_matrix_fast(embeddings: List[float]):
    embeddings = torch.tensor(embeddings).to("cuda")  # 将数据移到 GPU 上

    # 设置批处理大小和逐块计算的块大小
    batch_size = 512  # 适当调整批处理大小
    n = embeddings.size(0)
    similarity_matrix = torch.zeros((n, n)).to(
        embeddings.device
    )  # 在 GPU 上创建相似度矩阵

    # 使用 DataLoader 进行批处理计算
    dataset = TensorDataset(embeddings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 批处理计算余弦相似度
    start_idx = 0
    for (batch_embeddings,) in loader:
        batch_size = batch_embeddings.size(0)

        # 计算整个批次中的余弦相似度
        similarity_batch = torch.matmul(
            batch_embeddings, embeddings.transpose(0, 1)
        )

        # 更新相似度矩阵的部分
        end_idx = start_idx + batch_size
        similarity_matrix[start_idx:end_idx, :] = similarity_batch

        start_idx = end_idx

    return similarity_matrix


# 计算相似度矩阵
def calculate_similarity_matrix(embeddings):
    embeddings = torch.tensor(embeddings).to("cuda")  # 将数据移到 GPU 上

    # 设置批处理大小和逐块计算的块大小
    batch_size = 512  # 适当调整批处理大小
    block_size = 2048  # 适当调整块大小

    n = embeddings.size(0)
    similarity_matrix = torch.zeros((n, n)).to(
        embeddings.device
    )  # 在 GPU 上创建相似度矩阵

    # 逐块计算余弦相似度矩阵
    for start in tqdm(range(0, n, block_size)):
        end = min(start + block_size, n)
        block_embeddings = embeddings[start:end]

        # 计算余弦相似度矩阵块
        similarity_block = torch.matmul(block_embeddings, embeddings.t())

        # 更新相似度矩阵
        similarity_matrix[start:end, :] = similarity_block

    return similarity_matrix


# 用于调试UserHandler类的功能
if __name__ == "__main__":
    import pandas as pd

    df1 = pd.read_csv("/mntnlp/xingyi/label_data/1030_to_tag.csv")

    chunks_list = df1["title"].tolist()
    emb_model = FlagModel("/mntnlp/common_base_model/bge-m3")
    embeddings = emb_model.encode(chunks_list)
    similarity_matrix = calculate_similarity_matrix_fast(embeddings)

    # 相似度阈值
    threshold = 0.95

    filter_hash = {}
    similarity_matrix = similarity_matrix.cpu().numpy()

    indices = np.where(similarity_matrix > threshold)

    for i in range(indices[0].shape[0]):
        if indices[0][i] < indices[1][i]:
            filter_hash[indices[0][i]] = filter_hash.get(indices[0][i], []) + [
                indices[1][i]
            ]

    filtered_indices = set()
    for k in filter_hash:
        if k not in filtered_indices:
            for delete_idx in filter_hash[k]:
                filtered_indices.add(delete_idx)

    final_res = []
    for i, val in tqdm(enumerate(chunks_list)):
        if i not in filtered_indices:
            if val:
                final_res.append(i)
    df1.iloc[final_res].to_csv(
        "1030_to_tag.csv", encoding="utf-8-sig", index=False
    )
