# 实验报告：基于 DQN 的 CartPole-v1 求解

## 一、实验目的

CartPole 的状态空间是连续的——杆角、角速度、小车位置、速度，四个浮点数——这让 Q 表直接失效。本实验的目标是用神经网络代替 Q 表，搞清楚 DQN 为什么能在这类问题上工作，以及它的两个关键设计（经验回放和目标网络）各自解决了什么问题。

具体来说：
1. 在 `CartPole-v1` 上跑通 DQN 的完整训练流程；
2. 用 PyTorch 手写经验回放池、目标网络和 $\varepsilon$-greedy 策略；
3. 观察奖励曲线、Loss 曲线，确认模型确实收敛。

## 二、实验要求

**环境：**

| 项 | 值 |
|---|---|
| OS | macOS |
| Python | 3.x（conda env `rl_course`） |
| 主要依赖 | `gymnasium[classic_control]`, `torch`, `numpy`, `matplotlib` |
| 运行设备 | CPU（网络约 33K 参数，CPU 比 MPS/CUDA 省去 kernel 启动开销，反而更快） |

```bash
conda activate rl_course
pip install -r requirements.txt
python dqn_cartpole.py
```

**交付物：**
- `dqn_cartpole.py`：完整训练代码
- `outputs/dqn_cartpole_policy.pt`：训练好的策略网络权重
- `outputs/training_log.csv`：逐回合日志（reward / epsilon / loss / success）
- 奖励曲线、成功率曲线截图

## 三、设计思路

### 3.1 为什么用神经网络

CartPole 状态 $s = [x,\dot x,\theta,\dot\theta] \in \mathbb{R}^4$ 连续，没法枚举。DQN 用参数化函数 $Q(s,a;\theta)$ 近似动作值，输入状态向量，输出两个动作（左推/右推）各自的 Q 值。

### 3.2 TD 目标与损失

目标网络参数 $\theta^-$ 计算 TD target：

$$
y = r + \gamma \max_{a'} Q(s',a';\theta^-) \cdot (1 - \text{done})
$$

当前网络最小化 MSE：

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}} \Big[ \big( y - Q(s,a;\theta) \big)^2 \Big]
$$

### 3.3 两个稳定性设计

**经验回放**：每步把 $(s,a,r,s',\text{done})$ 存入容量 20000 的 deque，训练时随机采 batch=64 条。这样一条经验可以被重复利用，同时也打破了时间上的相关性——连续步骤的状态高度相似，直接用会让梯度估计有偏。

**目标网络**：$\theta^-$ 慢速跟踪 $\theta$，让 TD target 在一段时间内保持稳定，避免网络追着自己更新的目标跑。本实验默认每 150 步做一次硬拷贝；也可以用软更新 $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$（$\tau=0.005$），曲线会更平滑但收敛稍慢。

软更新实现：

```python
def soft_update(target_net, policy_net, tau):
    with torch.no_grad():
        for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(pp.data, alpha=tau)
```

### 3.4 探索策略

$$
a_t = \begin{cases} \text{随机动作}, & \text{概率 } \varepsilon \\ \arg\max_a Q(s_t,a;\theta), & \text{概率 } 1-\varepsilon \end{cases}
$$

$\varepsilon$ 从 1.0 线性衰减至 0.01，衰减周期 400 回合。前 1000 步只入池不训练（`min_replay=1000`），让 buffer 先积累一些多样性。

### 3.5 网络结构与超参数

Q 网络是两层 MLP：`4 → 128 → 128 → 2`，激活 ReLU，输出不加激活。

| 超参数 | 值 |
|---|---:|
| 学习率 | 5e-4 |
| 折扣因子 $\gamma$ | 0.99 |
| batch size | 64 |
| buffer 容量 | 20000 |
| 目标网络更新（硬） | 每 150 步 |
| 训练回合 | 500 |
| 梯度裁剪 | max_norm=1.0 |

梯度裁剪（`clip_grad_norm_`）在 $\gamma$ 接近 1 时比较重要，不加的话偶尔会有训练突然崩掉的情况。

### 3.6 单步优化流程

```python
# 1. 当前网络预测 Q(s, a)
q_sa = policy_net(states).gather(1, actions)

# 2. 目标网络算 TD target（不反传梯度）
with torch.no_grad():
    targets = rewards + gamma * target_net(next_states).max(1).values * (1 - dones)

# 3. MSE → 反传 → 裁剪 → 更新
loss = F.mse_loss(q_sa.squeeze(), targets)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
optimizer.step()
```

## 四、运行结果

### 奖励曲线

![Reward Curve](assets/reward_curve.png)

灰色细线是原始每回合奖励，彩色粗线是窗口 20 的滑动平均，水平虚线是官方阈值 475。

前 100 回合基本在瞎跑，奖励低于 50；150 回合后奖励开始快速爬升；500 回合结束时滑动均值在 475 附近，最后 10 回合均值接近 500。

### 成功率曲线

![Success Rate](assets/success_rate.png)

"成功"定义为单回合奖励 ≥ 475。训练末期滑动成功率稳定在 0.9 以上。

### Loss 曲线

Loss 不是严格单调下降的——TD target 本身会随策略改变而漂移，所以 Loss 有波动是正常现象。整体滑动均值还是在震荡下降的，不是 bug。

### 测试结果

训练结束后加载 `dqn_cartpole_policy.pt`，关闭探索，跑 3 个测试回合：

```
Test episode 1: reward = 500
Test episode 2: reward = 500
Test episode 3: reward = 500
```

三回合全部撑满 500 步，策略稳定。

### 训练日志

`outputs/training_log.csv` 字段：`episode, reward, epsilon, loss_mean, success`，可用 Pandas 或 Excel 复现图表。

## 五、问题与思考

**Q1：经验回放为什么有效？**

直觉上，强化学习的数据是时序相关的——前后状态相差很小，连续用来做梯度下降会让网络在一个局部钻很深。随机打乱后，每个 batch 来自不同时间段的经验，梯度估计更稳。另外，存下来重复利用也比每步扔掉要高效。

**Q2：目标网络不更新会怎样？**

试过把 `target_update` 设成无限大（固定住 $\theta^-$）。开始收敛挺正常，但大概 200 回合后奖励开始抖动并下降，因为目标太旧了，当前策略已经远离它计算出的 TD target 所描述的环境。

**Q3：标准 DQN 的高估问题**

`q_sa` 和 TD target 里的 $\max_{a'}$ 都多少有些高估。Double DQN 的做法是用 `policy_net` 选动作、`target_net` 估值，把两个操作分开，高估会缓解不少。在 CartPole 这个简单环境里影响不大，但如果换到 Atari 就比较明显了。

**Q4：软更新和硬更新哪个好？**

在这个实验里差别不大。把 `Config.use_soft_update = True` 跑一遍，曲线确实更平滑，但收敛时间差不多。硬更新在策略改变较快时会有一段目标突然跳变的问题，软更新更稳，但参数 $\tau$ 需要调。

**可以继续做的事：**
- Dueling DQN：把最后一层拆成 $V(s)$ 和 $A(s,a)$，对状态值的学习更高效
- 优先经验回放（PER）：按 TD 误差大小采样，收敛更快，但实现复杂度高一截（需要 sum-tree）

