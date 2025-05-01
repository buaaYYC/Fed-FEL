# Fitness-Driven Evolutionary Federated Learning: A Novel Communication-Efficient Framework for Heterogeneous Data Environments(FD-EFL)

## Project Overview 
FD-EFL is a lightweight framework designed for heterogeneous federated learning environments, integrating three core technologies: **evolutionary strategy optimization**, **dynamic client selection**, and **critical learning period-based dynamic population adjustment**. It aims to reduce communication overhead while maintaining model accuracy. It is suitable for image classification tasks such as CIFAR-10 and FMNIST.

FD-EFL 是一个面向异构联邦学习环境的轻量级框架，集成了 ​**进化策略优化**、**动态客户端选择**​ 和 ​**关键学习期动态种群调整**​ 三大核心技术，旨在降低通信开销的同时保持模型精度。适用于 CIFAR10、FMNIST 等图像分类任务。

---

## 目录结构

### 1. ​**核心模块**
```plaintext
├─ backprop                # 模型训练与优化逻辑
├─ configs                # 不同数据集的配置文件（CIFAR10/FMNIST/MNIST）
│  ├─ Vision-CIFAR10       # CIFAR10 数据集配置
│  ├─ Vision-FMNIST        # FMNIST 数据集配置
│  └─ Vision-MNIST        # MNIST 数据集配置
├─ evosax                # 进化策略实现（CMA-ES 等）
│  ├─ networks             # 神经网络定义
│  ├─ problems            # 优化问题定义
│  ├─ restarts            # 进化过程重试机制
│  ├─ strategies          # 进化策略实现
│  └─ utils               # 工具函数
└─ utils                 # 公共工具模块

├─ args.py                # 参数解析与配置管理
├─ bp.py                  # 基础训练逻辑
├─ client_quality.py      # 客户端质量评估（基于数据熵）
├─ fed-fel.py             # 主程序入口（联邦学习训练）
├─ fedavg.py              # 传统 FedAvg 实现（对比基准）
└─ utils.py               # 工具函数库
```
## 快速入门指南
### 1. ​环境准备
```pip install -r requirements.txt```

### 2. ​启动训练
```
执行下面命令<!--  -->
python fed-fel.py --config configs/Vision-CIFAR10/config.yaml
```

## 数据集准备

为了能够顺利运行此项目，您需要准备并下载以下数据集：

- **CIFAR10**（vggNet）
- **FMNIST**（AlexNet）
- **MNIST***（AlexNet)

下载方式请参考各自数据集的官方网站或相关链接。


## 实验结果

经过实验验证，FD-EFL显著减少了在异构数据环境下的通信开销，同时保证了与传统联邦学习方法（如FedAvg）相当的模型准确度。
