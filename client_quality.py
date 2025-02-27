import numpy as np
from scipy.stats import entropy

def calculate_entropy(data, labels):
    """
    计算给定数据集的熵值（类别分布均匀性指标）
    
    Args:
        data (np.ndarray or jax.numpy.ndarray): 客户端数据
        labels (np.ndarray or jax.numpy.ndarray): 对应标签
        
    Returns:
        float: 熵值（范围0-1，1表示完全不平衡）
    """
    # 转换为NumPy数组以确保兼容性
    labels_np = np.array(labels)
    unique_labels = np.unique(labels_np)
    
    # 处理空数据情况
    if len(unique_labels) == 0:
        return 0.0
    
    # 计算每个类别的样本数量
    label_counts = {}
    for label in labels_np:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    # 计算概率分布
    probabilities = list(label_counts.values()) / len(labels_np)
    
    # 计算熵值（使用自然对数底数为e）
    entropy_value = entropy(probabilities, base=np.e)
    
    # 转换为以2为底的对数（可选）
    entropy_value = entropy_value / np.log(2)
    
    return entropy_value