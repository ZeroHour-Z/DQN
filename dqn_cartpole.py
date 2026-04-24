"""
CartPole-v1 DQN 实验
经验回放 | 双网络（Policy / Target）| epsilon-greedy | 训练 + 可视化 + 测试渲染
"""
from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ────────────────────────────── 超参数 ──────────────────────────────

@dataclass
class Config:
    lr: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 20000
    batch_size: int = 64
    target_update: int = 150          # 每 N 步硬同步 target 网络（硬更新模式下使用）
    use_soft_update: bool = False     # True: 软更新 θ_target ← τθ + (1-τ)θ_target；False: 硬更新
    tau: float = 0.005                # 软更新系数
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_episodes: int = 400     # 线性衰减至 eps_end 所需回合
    hidden_dim: int = 128
    num_episodes: int = 500
    min_replay: int = 1000            # 回放池最少经验数才开始训练
    plot_window: int = 20             # 滑动平均窗口
    test_episodes: int = 3
    seed: int = 42
    log_csv: str = "outputs/training_log.csv"

# ────────────────────────────── 经验回放池 ──────────────────────────

class ReplayBuffer:
    """存储 (s, a, r, s', done)，支持随机批量采样。"""

    def __init__(self, capacity: int) -> None:
        self._buf: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self._buf.append((state.copy(), action, reward, next_state.copy(), done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self._buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self._buf)

# ────────────────────────────── Q 网络 ──────────────────────────────

class QNetwork(nn.Module):
    """两层隐藏层 MLP：state_dim -> hidden -> hidden -> action_dim"""

    def __init__(self, n_obs: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ────────────────────────────── 动作选择 ────────────────────────────

def select_action(state: np.ndarray, policy_net: QNetwork,
                  epsilon: float, n_actions: int,
                  device: torch.device) -> int:
    """epsilon-greedy：以 epsilon 概率随机，否则选 Q 值最大的动作。"""
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        x = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        return int(policy_net(x).argmax(dim=1).item())

# ────────────────────────────── 单步优化 ────────────────────────────

def optimize(policy_net: QNetwork, target_net: QNetwork,
             replay: ReplayBuffer, optimizer: optim.Optimizer,
             gamma: float, batch_size: int, device: torch.device) -> float:
    states, actions, rewards, next_states, dones = replay.sample(batch_size)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # 当前网络 Q(s, a)
    q_sa = policy_net(states).gather(1, actions)

    # 目标网络计算 target = r + gamma * max_a' Q_target(s', a') * (1 - done)
    with torch.no_grad():
        next_q_max = target_net(next_states).max(dim=1, keepdim=True).values
        targets = rewards + gamma * next_q_max * (1.0 - dones)

    loss = nn.functional.mse_loss(q_sa, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


def soft_update(target_net: QNetwork, policy_net: QNetwork, tau: float) -> None:
    """θ_target ← τ·θ + (1-τ)·θ_target"""
    with torch.no_grad():
        for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(pp.data, alpha=tau)

# ────────────────────────────── epsilon 衰减 ───────────────────────

def linear_epsilon(episode: int, cfg: Config) -> float:
    if episode >= cfg.eps_decay_episodes:
        return cfg.eps_end
    frac = episode / max(cfg.eps_decay_episodes, 1)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

# ────────────────────────────── 滑动平均 ───────────────────────────

def moving_average(data: list[float], window: int) -> np.ndarray:
    if window <= 1 or len(data) < window:
        return np.array([])
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")

# ────────────────────────────── 训练主循环 ──────────────────────────

def train(cfg: Config, device: torch.device):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = gym.make("CartPole-v1")
    threshold = float(env.spec.reward_threshold) if env.spec.reward_threshold else 475.0

    n_obs = int(env.observation_space.shape[0])   # 4
    n_actions = int(env.action_space.n)            # 2

    policy_net = QNetwork(n_obs, n_actions, cfg.hidden_dim).to(device)
    target_net = QNetwork(n_obs, n_actions, cfg.hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.buffer_size)

    episode_rewards: list[float] = []
    episode_success: list[float] = []
    episode_losses: list[float] = []
    global_step = 0
    best_mean50 = -float("inf")
    best_ckpt = "outputs/dqn_cartpole_policy.pt"

    for ep in range(cfg.num_episodes):
        state, _ = env.reset(seed=cfg.seed + ep)
        eps = linear_epsilon(ep, cfg)
        ep_reward = 0.0
        ep_loss_sum = 0.0
        ep_loss_cnt = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = select_action(state, policy_net, eps, n_actions, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.push(state, action, float(reward), next_state, done)
            state = next_state
            ep_reward += reward
            global_step += 1

            if len(replay) >= cfg.min_replay:
                loss_val = optimize(policy_net, target_net, replay, optimizer,
                                    cfg.gamma, cfg.batch_size, device)
                ep_loss_sum += loss_val
                ep_loss_cnt += 1

                if cfg.use_soft_update:
                    soft_update(target_net, policy_net, cfg.tau)

            if (not cfg.use_soft_update) and global_step % cfg.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(ep_reward)
        episode_success.append(1.0 if ep_reward >= threshold else 0.0)
        episode_losses.append(ep_loss_sum / ep_loss_cnt if ep_loss_cnt else float("nan"))

        # 实时保存最优检查点
        if len(episode_rewards) >= 50:
            mean50 = float(np.mean(episode_rewards[-50:]))
            if mean50 > best_mean50:
                best_mean50 = mean50
                torch.save(policy_net.state_dict(), best_ckpt)

        if (ep + 1) % 50 == 0:
            recent = np.mean(episode_rewards[-50:])
            print(f"Episode {ep+1:4d}  eps={eps:.3f}  "
                  f"last-50 mean={recent:.1f}  step={global_step}")

    env.close()

    # 加载最优检查点（而非末尾权重）
    policy_net.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"Best last-50 mean during training: {best_mean50:.1f}")

    # 写入 CSV 日志
    import csv
    with open(cfg.log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "epsilon", "loss_mean", "success"])
        for i, (r, s, lv) in enumerate(zip(episode_rewards, episode_success, episode_losses), 1):
            writer.writerow([i, r, linear_epsilon(i - 1, cfg), lv, int(s)])

    return episode_rewards, episode_success, episode_losses, threshold, policy_net

# ────────────────────────────── 可视化 ─────────────────────────────

def plot_curves(rewards: list[float], success: list[float],
                losses: list[float], threshold: float, window: int) -> None:
    episodes = np.arange(1, len(rewards) + 1)
    ma_r = moving_average(rewards, window)
    ma_s = moving_average(success, window)

    # ---- 奖励曲线 ----
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(episodes, rewards, alpha=0.35, label="raw")
    if len(ma_r):
        ax1.plot(np.arange(window, len(rewards) + 1), ma_r,
                 label=f"moving avg (w={window})")
    ax1.axhline(threshold, color="gray", ls="--", lw=1, label="threshold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("CartPole-v1 DQN — Episode Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig("assets/reward_curve.png", dpi=150)

    # ---- 成功率曲线 ----
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(episodes, success, alpha=0.35, drawstyle="steps-post", label="raw (0/1)")
    if len(ma_s):
        ax2.plot(np.arange(window, len(success) + 1), ma_s,
                 label=f"success rate (w={window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success (reward >= threshold)")
    ax2.set_title("CartPole-v1 DQN — Success Rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("assets/success_rate.png", dpi=150)

    # ---- Loss 曲线 ----
    losses_arr = np.array(losses, dtype=float)
    if np.any(~np.isnan(losses_arr)):
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.plot(episodes, losses_arr, alpha=0.4, label="per-episode mean loss")
        ma_l = moving_average([0.0 if np.isnan(x) else x for x in losses], window)
        if len(ma_l):
            ax3.plot(np.arange(window, len(losses) + 1), ma_l,
                     label=f"moving avg (w={window})")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("TD Loss (MSE)")
        ax3.set_title("CartPole-v1 DQN — Training Loss")
        ax3.set_yscale("log")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig("assets/loss_curve.png", dpi=150)

    plt.show()

# ────────────────────────────── 测试渲染 ───────────────────────────

def _run_episode(policy_net: QNetwork, env: gym.Env,
                 device: torch.device, seed: int) -> float:
    """不渲染地跑一个回合，返回总奖励。"""
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            x = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            action = int(policy_net(x).argmax(dim=1).item())
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward


def test_render(policy_net: QNetwork, device: torch.device,
                episodes: int, seed: int) -> None:
    """静默跑 episodes 局，找出奖励最高的一局，再渲染展示。"""
    policy_net.eval()

    # ── 1. 静默评估，找最优 seed ──
    silent_env = gym.make("CartPole-v1")
    best_seed, best_reward = seed + 10_000, -1.0
    for ep in range(episodes):
        ep_seed = seed + 10_000 + ep
        r = _run_episode(policy_net, silent_env, device, ep_seed)
        print(f"Test episode {ep+1}: reward = {r:.0f}")
        if r > best_reward:
            best_reward, best_seed = r, ep_seed
    silent_env.close()

    # ── 2. 渲染奖励最高的一局 ──
    print(f"\nRendering best episode (reward={best_reward:.0f}) ...")
    render_env = gym.make("CartPole-v1", render_mode="human")
    state, _ = render_env.reset(seed=best_seed)
    terminated = truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            x = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            action = int(policy_net(x).argmax(dim=1).item())
        state, _, terminated, truncated, _ = render_env.step(action)
        time.sleep(0.02)
    render_env.close()

# ────────────────────────────── main ───────────────────────────────

def main() -> None:
    # CartPole 网络极小（~33K参数）+ batch_size=64，CPU 反而比 MPS/CUDA 快。
    # 如需测试大网络（Atari 等）可改为 mps 或 cuda。
    device = torch.device("cpu")
    cfg = Config()
    print(f"device: {device}")

    rewards, success, losses, threshold, policy_net = train(cfg, device)
    print(f"\nTraining done. Last 10 mean reward: "
          f"{np.mean(rewards[-10:]):.1f}  (threshold={threshold})")

    plot_curves(rewards, success, losses, threshold, cfg.plot_window)
    print("Saved assets/reward_curve.png, assets/success_rate.png, assets/loss_curve.png")
    print("Saved outputs/training_log.csv")

    print("Best checkpoint already saved as outputs/dqn_cartpole_policy.pt")

    test_render(policy_net, device, cfg.test_episodes, cfg.seed)


if __name__ == "__main__":
    main()
