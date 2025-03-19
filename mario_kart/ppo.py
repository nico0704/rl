import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.logprobs, self.dones, self.values = [], [], []

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.logprobs, self.dones, self.values = [], [], []


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # actor (Policy)
        self.actor = nn.Linear(64, action_dim)
        # critic (Value Function)
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        shared_features = self.shared(state)
        action_mean = torch.tanh(self.actor(shared_features))  # output in range [-1, 1]
        state_value = self.critic(shared_features)
        return action_mean, state_value

    def act(self, state):
        action_mean, state_value = self.forward(state)
        dist = MultivariateNormal(action_mean, torch.diag(torch.ones_like(action_mean) * 0.1))
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach(), state_value.detach()


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, epochs=10):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs

        # initialize networks
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.loss_fn = nn.MSELoss()
        self.buffer = RolloutBuffer()

    def select_action(self, state):
        # choose action based on current policy
        state = torch.FloatTensor(state).to(device)
        action, log_prob, value = self.policy.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(log_prob)
        self.buffer.values.append(value)
        return action.cpu().numpy()

    def compute_advantages(self, rewards, values, dones):
        # compute GAE (Generalized Advantage Estimation)
        advantages, returns = [], []
        adv = 0
        last_value = values[-1]

        for t in reversed(range(len(rewards))):
            if dones[t]:
                last_value = 0
            td_error = rewards[t] + self.gamma * last_value - values[t]
            adv = td_error + self.gamma * 0.95 * adv
            advantages.insert(0, adv)
            returns.insert(0, adv + values[t])
            last_value = values[t]

        return torch.tensor(advantages, dtype=torch.float32).to(device), torch.tensor(returns, dtype=torch.float32).to(device)

    def update(self):
        states = torch.stack(self.buffer.states).to(device)
        actions = torch.stack(self.buffer.actions).to(device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(device)
        values = torch.stack(self.buffer.values).squeeze().to(device)

        rewards = self.buffer.rewards
        dones = self.buffer.dones

        advantages, returns = self.compute_advantages(rewards, values, dones)
        for _ in range(self.epochs):
            new_action_means, value_preds = self.policy(states)
            dist = MultivariateNormal(new_action_means, torch.diag_embed(torch.ones_like(new_action_means) * 0.1))
            new_logprobs = dist.log_prob(actions)
            ratios = torch.exp(new_logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.loss_fn(value_preds.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.buffer.clear()

    def train(self, env, num_episodes=1000):
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = env.step(action)
                self.buffer.rewards.append(reward)
                self.buffer.dones.append(done)
                state = next_state
                episode_reward += reward
                if done:
                    print(f"Episode {episode}, Reward: {episode_reward}")
                    self.update()
            env.render()
