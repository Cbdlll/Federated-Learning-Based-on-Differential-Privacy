# 联邦学习框架 (Federated Learning Framework)

这个项目实现了一个完整的联邦学习框架，集成了LeNet模型和差分隐私技术，支持10个客户端进行分布式训练。

## 项目特点

- 基于PyTorch实现的LeNet模型
- 差分隐私(DP-SGD)保护客户端数据隐私
- 支持IID和非IID数据分布
- 数据完全解耦，每个客户端仅访问自己的本地数据
- 模块化设计，易于扩展

## 项目结构

```
.
├── main.py             # 主入口点
├── server.py           # 联邦服务器实现
├── client.py           # 客户端实现
├── data_loader.py      # 数据加载和分布
├── privacy.py          # 差分隐私实现
├── utils.py            # 工具函数
├── models/             # 模型定义
│   └── lenet.py        # LeNet模型实现
├── requirements.txt    # 依赖包列表
└── README.md           # 项目说明
```

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd federated-learning-dp
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

运行联邦学习实验：

```bash
python main.py
```

### 可选参数

- `--num_rounds`: 通信轮数 (默认: 100)
- `--num_clients`: 客户端数量 (默认: 10)
- `--batch_size`: 本地批次大小 (默认: 64)
- `--local_epochs`: 本地训练轮数 (默认: 5)
- `--lr`: 学习率 (默认: 0.01)
- `--momentum`: SGD动量 (默认: 0.9)
- `--dp_epsilon`: 差分隐私ε参数 (默认: 1.0)
- `--dp_delta`: 差分隐私δ参数 (默认: 1e-5)
- `--dp_max_grad_norm`: 梯度裁剪范数 (默认: 1.0)
- `--seed`: 随机种子 (默认: 42)
- `--device`: 运行设备 (默认: CUDA若可用，否则CPU)

例如：

```bash
python main.py --num_rounds 50 --local_epochs 2 --dp_epsilon 0.5
```

## 实现细节

### 联邦学习流程

1. 服务器初始化全局模型
2. 服务器将全局模型分发给每个客户端
3. 每个客户端使用本地数据训练模型（集成差分隐私）
4. 客户端将更新后的模型参数发送回服务器
5. 服务器聚合参数（通过FedAvg算法）
6. 重复步骤2-5，直到达到预设轮数

### 差分隐私实现

通过以下步骤保护客户端隐私：

1. 梯度裁剪：限制每个样本的梯度影响，防止单样本过大影响
2. 噪声添加：通过添加高斯噪声保护隐私
3. 隐私预算跟踪：监控累计隐私损失

## 扩展方向

- 支持更多模型架构
- 实现更多联邦聚合算法（如FedProx、FedNova）
- 添加模型压缩技术减少通信开销
- 集成更完备的隐私预算追踪器

## 参考文献

1. Mcmahan H B , et al. "Communication-efficient learning of deep networks from decentralized data."
2. Martí, Abadi N , Chu A ,et al. "Deep Learning with Differential Privacy."