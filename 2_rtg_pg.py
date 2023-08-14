import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i == len(sizes)-2:
            layers.append(output_activation())
        else:
            layers.append(activation())
    return nn.Sequential(*layers)

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, episodes_to_render=0):
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box) and len(env.observation_space.shape) == 1
    assert isinstance(env.action_space, Discrete)

    observation, info = env.reset()
    obs_n = observation.shape[0]
    act_n = env.action_space.n
    logits_net = mlp([obs_n] + hidden_sizes + [act_n])

    env.close()

    optim = torch.optim.Adam(logits_net.parameters(), lr=lr)

    def train_epoch():
        rewards = []
        logprobs = []
        episode_idx = 0

        while len(rewards) < batch_size:
            should_render = episode_idx < episodes_to_render
            episode_idx += 1
            env = gym.make(env_name, render_mode="human" if should_render else None)
            observation, info = env.reset()
            episode_rewards = []

            while True:
                observation = torch.as_tensor(observation, dtype=torch.float32)
                policy = Categorical(logits=logits_net(observation))
                action = policy.sample()
                observation, reward, terminated, truncated, info = env.step(action.item())

                episode_rewards.append(reward)
                logprobs.append(policy.log_prob(action))

                if terminated or truncated:
                    rtg = []
                    cur = 0
                    for x in reversed(episode_rewards):
                        cur += x
                        rtg.append(cur)
                    rtg = reversed(rtg)
                    rewards += rtg
                    env.close()
                    break
        
        optim.zero_grad()
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        logprobs = torch.stack(logprobs)
        loss = -torch.mean(rewards * logprobs)
        loss.backward()
        optim.step()

        return torch.mean(rewards)

    for epoch in range(epochs):
        mean_reward = train_epoch()
        print(f"Epoch {epoch} mean reward: {mean_reward}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--episodes_to_render', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing reward-to-go policy gradient.\n')
    train(env_name=args.env_name, episodes_to_render=args.episodes_to_render, lr=args.lr)
