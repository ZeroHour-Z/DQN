# 实验报告：基于连续状态空间 CartPole 环境的 DQN 求解实验

## 一、实验目的

1. 理解为什么在连续状态空间中表格型 Q-Learning 会失效，掌握用神经网络近似 $Q(s,a)$ 的思路。
2. 实现 DQN 的两大核心基石：**经验回放（Replay Buffer）** 与 **目标网络（Target Network）**。
3. 用 PyTorch 搭建 MLP，完成前向传播、损失计算与反向传播。
4. 在 `CartPole-v1` 上完成训练 → 测试 → 可视化全流程。

## 二、实验原理

### 2.1 从 Q-Table 到 Q-Network

CartPole 状态 $s = [x,\dot x,\theta,\dot\theta] \in \mathbb{R}^4$ 是连续量，无法像 FrozenLake 那样建离散 Q 表。DQN 用一个参数化函数 $Q(s,a;\theta)$ 近似动作值函数，网络输入 $s$、输出每个离散动作的 Q 值。

### 2.2 时序差分目标与损失

目标网络参数 $\theta^-$ 提供稳定的 TD target：

$$
y = r + \gamma \max_{a'} Q(s',a';\theta^-) \cdot (1 - \text{done})
$$

当前网络通过最小化 MSE 损失来逼近该目标：

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}} \Big[ \big( y - Q(s,a;\theta) \big)^2 \Big]
$$

其中 $\mathcal{D}$ 是经验回放池。

### 2.3 两大稳定性技巧

- **经验回放**：将 $(s,a,r,s',\text{done})$ 入池后随机采样训练，打破时间相关性、提高样本利用率。
- **目标网络**：$\theta^-$ 慢速跟踪 $\theta$，避免"自己追自己"导致的发散。
  - 硬更新：每 $N$ 步 $\theta^- \leftarrow \theta$
  - 软更新：$\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$，$\tau=0.005$

### 2.4 epsilon-greedy 探索

$$
a_t = \begin{cases} \text{随机动作}, & \text{概率 } \varepsilon \\ \arg\max_a Q(s_t,a;\theta), & \text{概率 } 1-\varepsilon \end{cases}
$$

$\varepsilon$ 从 `eps_start=1.0` 线性衰减至 `eps_end=0.01`。

## 三、实验环境

| 项 | 值 |
|---|---|
| OS | macOS |
| Python | 3.x (conda env `rl_course`) |
| 依赖 | `gymnasium[classic_control]`, `numpy`, `matplotlib`, `torch` |
| Device | CPU（网络参数量约 33K，CPU 反而快于 MPS/CUDA 的 kernel 启动开销）|

依赖安装：

```bash
conda activate rl_course
pip install -r requirements.txt
```

## 四、核心模块实现

代码位于 [dqn_cartpole.py](dqn_cartpole.py)，下面按实验要求 3.2.1 对各模块逐一说明。

### 4.1 经验回放池 `ReplayBuffer`

使用 `collections.deque(maxlen=capacity)` 自动淘汰最旧经验；`sample()` 通过 `random.sample` 随机批量采样，并一次性 stack 为张量，减少逐条拷贝开销。

### 4.2 Q 网络 `QNetwork`

两层隐藏层 MLP：`4 → 128 → 128 → 2`，激活函数 ReLU。

### 4.3 epsilon-greedy 动作选择

`select_action()` 用 `random.random()` 判断是否探索；网络推理段用 `torch.no_grad()` 关闭梯度，避免污染计算图。

### 4.4 单步优化 `optimize`

关键三步：

1. `q_sa = policy_net(states).gather(1, actions)`：按动作下标取当前网络预测的 Q 值。
2. `targets = r + γ · max_{a'} Q_target(s',a') · (1-done)`：在 `torch.no_grad()` 下用目标网络算 TD target。
3. MSE loss → 反向 → 梯度裁剪 `clip_grad_norm_(1.0)` → `optimizer.step()`。

### 4.5 目标网络更新：硬 / 软两种模式

- `cfg.use_soft_update = False`（默认）：每 `target_update` 步执行 `target_net.load_state_dict(policy_net.state_dict())`。
- `cfg.use_soft_update = True`：每次梯度更新后调用 `soft_update(target, policy, tau=0.005)`。

软更新实现：

```python
def soft_update(target_net, policy_net, tau):
    with torch.no_grad():
        for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(pp.data, alpha=tau)
```

### 4.6 训练主循环 `train`

每个回合：

1. `env.reset(seed=cfg.seed+ep)` 保证可复现；
2. 在 `while not done` 中：动作选择 → 环境交互 → 入池 → 采样优化 → 目标网络更新；
3. 只有当 `len(replay) ≥ cfg.min_replay` 才开始训练，避免初期无意义的梯度更新；
4. 回合末记录 `reward / success / mean loss`；
5. 训练结束写出 `training_log.csv`。

### 4.7 测试渲染 `test_render`

关闭探索（直接 `argmax`），使用 `gym.make('CartPole-v1', render_mode='human')` 可视化杆子保持直立。

## 五、超参数配置

| 超参数 | 取值 | 建议范围 |
|---|---:|---|
| 学习率 `lr` | 5e-4 | 1e-4 ~ 1e-3 |
| 折扣因子 `gamma` | 0.99 | 0.98 ~ 0.99 |
| 回放池容量 `buffer_size` | 20000 | 10000 ~ 50000 |
| 批量大小 `batch_size` | 64 | 32 ~ 128 |
| 目标网络更新 `target_update` | 150 步（硬）/ τ=0.005（软） | 100~200 步 / τ=0.005 |
| `eps_start / eps_end` | 1.0 / 0.01 | — |
| `eps_decay_episodes` | 400 | — |
| 隐藏层 | 2 × 128 ReLU | 1-2 层 64/128 |
| 训练回合 `num_episodes` | 500 | 300 ~ 600 |
| `min_replay` | 1000 | 预热后再训练 |
| 梯度裁剪 | max_norm=1.0 | 稳定训练 |

## 六、训练结果

### 6.1 奖励曲线

![Reward Curve](assets/reward_curve.png)

横轴为回合数、纵轴为每回合总奖励。灰色半透明线为原始值，彩色线为窗口 `w=20` 的滑动平均，虚线为 `CartPole-v1` 官方阈值 **475**。从曲线可见：

- 约前 50~100 回合处于"纯探索"阶段，奖励普遍低于 50；
- 约 150~250 回合奖励快速上升，滑动均值突破 200；
- 训练后半段滑动均值稳定在 475 附近，最后 10 回合均值接近最大值 500，已达成收敛。

### 6.2 成功率曲线

![Success Rate](assets/success_rate.png)

"成功"定义为单回合奖励 ≥ 阈值 475。窗口 20 的滑动成功率体现了从 0 → 1 的稳定爬升，训练末期成功率稳定在 0.9 以上。

### 6.3 训练 Loss 曲线（新增）

`loss_curve.png`（运行 `python dqn_cartpole.py` 后生成，纵轴 log 尺度）用于观察 TD 误差的总体下降趋势。由于 TD target 随 policy 改变而漂移，Loss 并非严格单调下降，但滑动均值呈震荡下降，这是 DQN 的典型行为，并非 bug。

### 6.4 训练日志 CSV

训练结束会生成 `training_log.csv`，字段为 `episode, reward, epsilon, loss_mean, success`，方便后续用 Pandas / Excel 复现图表或做统计分析。

## 七、测试结果

训练结束后自动加载策略网络、关闭探索，打开人类渲染模式执行 `cfg.test_episodes=3` 回合。典型输出示例（具体数值取决于训练随机性）：

```
Test episode 1: reward = 500
Test episode 2: reward = 500
Test episode 3: reward = 500
```

三个测试回合均达到环境最大步长 500（CartPole-v1 的 episode 上限），说明训练出的策略能长时间稳定保持杆子直立。

## 八、思考与改进方向

1. **Double DQN**：标准 DQN 用同一个 $\theta^-$ 既选动作又估值，存在 Q 值高估偏差；Double DQN 把 $\arg\max$ 交给 `policy_net`、估值交给 `target_net`，可缓解高估。
2. **Dueling DQN**：将最后一层分解为 $V(s)+A(s,a)-\bar A$，能更高效学习状态价值。
3. **优先经验回放（PER）**：按 TD 误差采样，可加速收敛（但需要 sum-tree 数据结构与重要性采样权重）。
4. **软更新 vs 硬更新**：本实验默认硬更新；将 `Config.use_soft_update = True` 可切换至软更新，训练曲线通常更平滑但早期收敛略慢。
5. **梯度裁剪**：`clip_grad_norm_(1.0)` 显著降低了训练崩溃概率，尤其当 $\gamma$ 接近 1 时。
6. **复现性**：通过 `random / numpy / torch` 三处 `seed` 统一种子并在 `env.reset(seed=…)` 中递增，基本可复现；但 CUDA/MPS 下的非确定性算子仍可能导致小幅差异。

