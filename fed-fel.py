import chex
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import wandb
from backprop import sl
from args import get_args
from utils import helpers, evo, CPCheck
from evosax import NetworkMapper, ParameterReshaper, FitnessShaper
from tqdm import tqdm
import os
from client_quality import calculate_entropy
from backprop.dataset_process import get_fed_datasets

# 环境配置
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# 辅助函数保持不变...
# l2 distance
def l2(x, y):
    return -1 * jnp.sqrt(jnp.sum((x - y) ** 2))

def l1(x, y):
    return -1 * jnp.sum(jnp.abs(x - y))

class TaskManager:
    def __init__(self, rng: chex.PRNGKey, args: OmegaConf):
        wandb.init(config=args)
        self.args = args

                # 使用改进的数据加载函数
        self.train_ds = get_fed_datasets(
            args.dataset,
            args.n_clients,
            num_samples_per_label=20,
            is_iid=args.is_iid,
            alpha=args.alpha
        )
        self.cp_check = CPCheck.CPCheck(args.n_clients, window=10, threshold=0.01)
        
        # 数据集加载
        self.train_ds, self.test_ds = sl.get_fed_datasets(
            args.dataset, args.n_clients, 20, args.dist == 'IID'
        )
        
        # 网络初始化
        network = NetworkMapper[args.network_name](**args.network_config)
        self.state = sl.create_train_state(
            jax.random.PRNGKey(args.seed),
            network,
            learning_rate=args.lr,
            momentum=args.momentum
        )
        
        # 参数工具类
        self.param_reshaper = ParameterReshaper(self.state.params, n_devices=1)
        self.test_param_reshaper = ParameterReshaper(self.state.params, n_devices=1)
        
        # 初始化进化策略
        self.es_params = {
            'population_size': args.pop_size,
            'total_params': sum(p.size for p in self.state.params)
        }
        self.strategy = evo.get_strategy_and_params_cma(**self.es_params)
        
        # 其他成员变量...
        
    def run_epoch(self, epoch, rng):
        # 客户端训练与梯度收集
        grad_norms = []
        clients_info = []
        for client_idx in range(self.args.n_clients):
            input_rng = jax.random.split(rng)[0]
            client_state, grad_norm = sl.train_epoch(
                self.state,
                self.train_ds[client_idx]['images'],
                self.train_ds[client_idx]['labels'],
                self.args.batch_size,
                input_rng
            )
            grad_norms.append(grad_norm)
            clients_info.append(client_state)
        
        # CLP检测与pop_size调整
        if self.args.enable_clp:
            self.cp_check.recv_info(grad_norms)
            new_pop_size, is_clp = self.cp_check.win_check(self.args.n_clients)
            
            # 强制约束范围
            adjusted_pop_size = max(16, min(128, new_pop_size))
            self.args.pop_size = adjusted_pop_size
            
            # 动态调整进化策略
            if is_clp and self.es_params['population_size'] != adjusted_pop_size:
                self.es_params['population_size'] = adjusted_pop_size
                self.strategy, _ = evo.get_strategy_and_params_cma(**self.es_params)
                # 重新初始化参数服务器
                self.server = self.strategy.initialize(
                    jax.random.PRNGKey(self.args.seed),
                    self.es_params
                ).replace(mean=self.test_param_reshaper.network_to_flat(self.state.params))
        
        # 参数聚合与进化策略交互
        target_server = jax.vmap(self.param_reshaper.network_to_flat)(clients_info[0].params)
        x, self.server = self.strategy.ask(jax.random.split(rng)[0], self.server, self.es_params)
        fitness = jax.vmap(lambda a, b: l2(a - b.mean, b - b.mean))(x, target_server)
        fitness = self.fit_shaper.apply(fitness)
        self.server = self.strategy.tell(x, fitness, self.server, self.es_params)
        self.state = self.state.replace(params=self.test_param_reshaper.reshape_single_net(self.server.mean))

        # 测试评估与日志记录
        test_loss, test_accuracy = sl.eval_model(self.state.params, self.test_ds, jax.random.split(rng)[0])
        wandb.log({
            'Round': epoch,
            'Test Loss': test_loss,
            'Global Accuracy': test_accuracy,
            'Population Size': self.args.pop_size
        })

def main():
    args = get_args()
    wandb.init(project='fedfel-new', config=args)
    
    # 使用狄利克雷分布划分成128份异构数据集
    fed_datasets = sl.get_diric_datasets(
        args.dataset,
        n_clients=128,          # 新增：划分为128份
        num_samples_per_label=20,
        is_iid=args.is_iid,      # 控制是否使用IID
        alpha=args.alpha         # 可通过配置文件调节异构性强度 0.1 ,0.3 , 0.5
    )
    
    # 计算客户端质量并排序
    quality_scores = []
    for data, labels in fed_datasets:
        # 将JAX数组转换为NumPy数组
        np_data = data.numpy()
        np_labels = labels.numpy()
        entropy = calculate_entropy(np_data, np_labels)
        quality_scores.append(entropy)
    
    # 根据质量降序排序客户端
    sorted_clients = sorted(zip(fed_datasets, quality_scores), key=lambda x: -x[1])
    sorted_fed_datasets = [client for client, _ in sorted_clients]
    
    # 转换为JAX数组格式
    fed_datasets = []
    for data, labels in sorted_fed_datasets:
        fed_datasets.append( (jax.array(data), jax.array(labels)) )
    
    # 初始化TaskManager
    manager = TaskManager(jax.random.PRNGKey(args.seed), args)
    manager.train_ds = fed_datasets
    
    # 执行训练
    for epoch in range(args.n_rounds):
        manager.run_epoch(epoch, jax.random.PRNGKey(args.seed + epoch))

if __name__ == '__main__':
    main()
