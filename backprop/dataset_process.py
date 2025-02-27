import numpy as np
from omegaconf import OmegaConf
import jax
from jax import numpy as jnp
from backprop import sl
from args import get_args
from evosax import NetworkMapper, ParameterReshaper, FitnessShaper
from tqdm import tqdm
import os
import random

# 新增迪利克雷数据划分类
class DirichletDataSplitter:
    def __init__(self, dataset, labels, n_clients, alpha=0.1, is_iid=True):
        self.dataset = dataset
        self.labels = labels
        self.n_clients = n_clients
        self.alpha = alpha
        self.is_iid = is_iid
        
        if not is_iid and alpha <= 0:
            raise ValueError("Non-IID requires alpha > 0")
        
        # 创建客户端数据划分
        self.partitions = self._split_data(dataset, labels, n_clients, alpha, is_iid)
    
    def _split_data(self, data, labels, n_clients, alpha, is_iid):
        # 使用迪利克雷分布生成非均匀划分
        label_indices = {}
        for idx, label in enumerate(labels):
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(idx)
        
        partitions = [[] for _ in range(n_clients)]
        if is_iid:
            # IID划分（简单随机采样）
            indices = list(range(len(data)))
            random.shuffle(indices)
            for i in range(n_clients):
                start = i * len(indices) // n_clients
                end = (i+1) * len(indices) // n_clients
                partitions[i] = indices[start:end]
        else:
            # 基于标签的迪利克雷分布划分
            for client_id in range(n_clients):
                # 每个类别的样本按迪利克雷分布分配
                client_samples = []
                for label, indices in label_indices.items():
                    if not indices:
                        continue
                    # 生成该类别样本在当前客户端的数量比例
                    probs = np.random.dirichlet([self.alpha] * n_clients)
                    total = sum(probs)
                    probs /= total
                    num_samples = int(len(indices) * probs[client_id])
                    selected = indices[:num_samples]
                    client_samples.extend(selected)
                    indices = indices[num_samples:]
                partitions[client_id] = client_samples
        
        return partitions
    
    def get_client_data(self, client_id):
        """获取指定客户端的数据和标签"""
        indices = self.partitions[client_id]
        X = self.dataset[indices]
        y = self.labels[indices]
        return X, y

# 修改数据加载逻辑
def get_fed_datasets(dataset_name, n_clients, num_samples_per_label, is_iid=True, alpha=0.1):
    # 加载原始数据集（假设此函数已存在）
    data, labels = sl.load_dataset(dataset_name)
    
    # 使用迪利克雷划分创建联邦数据集
    splitter = DirichletDataSplitter(data, labels, n_clients, alpha=alpha, is_iid=is_iid)
    fed_datasets = []
    for client_id in range(n_clients):
        X_client, y_client = splitter.get_client_data(client_id)
        fed_datasets.append((X_client, y_client))
    
    return fed_datasets