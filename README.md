# DQN CartPole 实验

使用 DQN 求解 Gymnasium `CartPole-v1`，包含经验回放、目标网络、epsilon-greedy 探索、训练日志和曲线可视化。

## 项目结构

```text
DQN/
├── src/                  # DQN 实现
├── docs/                 # 实验指南与模板
├── reports/              # 实验报告
├── outputs/              # 训练产物
│   ├── figures/          # 奖励、成功率、loss 曲线
│   ├── logs/             # CSV 训练日志
│   └── models/           # 策略网络权重
├── requirements.txt
└── README.md
```

## 运行

```bash
conda activate rl_course
pip install -r requirements.txt
python dqn_cartpole.py
```

完整实验说明见 `reports/REPORT.md`。
