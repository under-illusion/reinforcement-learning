import torch.nn as nn
import random
import numpy as np
import torch
import PIL.Image as Image
import torch.nn.functional as F

# dqn组件
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.index = -1

    def add(self, state, action, reward, next_state, done):
        if(self.size() <= self.capacity):
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.index = (self.index + 1) % self.capacity
            self.buffer[self.index] = [state, action, reward, next_state, done]

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class simplize_agent:
    def __init__(self, net):
        self.net = net
        self.epsilon = 1
        self.image_shape = [84, 84]
        self.image_stack = 4
        self.action_dim = 6
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def state_transform(self, pre_state = None, state = None):
        img = Image.fromarray(state, "RGB")
        img = img.resize(self.image_shape).convert('L')
        img = np.asarray(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0])
        if pre_state is None:
            next_state = np.array([img] * self.image_stack)
        else:
            next_state = np.append(pre_state[1:], [img], axis=0)
        return next_state

    def state_process(self, state):
        state =  torch.Tensor(state).to(self.device)
        state = state.unsqueeze(0)
        return state

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = self.state_process(state)
            action = self.net(state).argmax().item()
        self.epsilon -= 0.00005
        self.epsilon = max(self.epsilon, 0.01)
        return action

class VAnet(nn.Module):
    def __init__(self, action_dim):
        super(VAnet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_A = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        self.fc_V = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        A = self.fc_A(F.relu(self.conv(x)).view(x.size(0), -1))
        V = self.fc_V(F.relu(self.conv(x)).view(x.size(0), -1))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q

class DQN_net(nn.Module):
    def __init__(self, action_dim):
        super(DQN_net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output

class DQN(nn.Module):
    def __init__(self, input_shape, action_dim, learning_rate, gamma, epsilon, target_update, buffer_size, device, dqn_type = "DQN"):
        super(DQN, self).__init__()
        self.dqn_type = dqn_type
        self.ReplayBuffer = ReplayBuffer(buffer_size)

        if self.dqn_type == "DuelingDQN":
            self.q_net = VAnet(action_dim).to(device)
            self.target_q_net = VAnet(action_dim).to(device)
        else:
            self.q_net = DQN_net(action_dim).to(device)
            self.target_q_net = DQN_net(action_dim).to(device)


        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr= learning_rate)
        self.gamma = gamma
        self.count = 0
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.action_dim = action_dim
        self.image_shape = (input_shape[0], input_shape[1])
        self.image_stack = input_shape[2]


    def state_transform(self, pre_state = None, state = None):
        img = Image.fromarray(state, "RGB")
        img = img.resize(self.image_shape).convert('L')
        img = np.asarray(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0])
        if pre_state is None:
            next_state = np.array([img] * self.image_stack)
        else:
            next_state = np.append(pre_state[1:], [img], axis=0)
        return next_state

    def state_process(self, state):
        state =  torch.Tensor(state).to(self.device)
        state = state.unsqueeze(0)
        return state

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = self.state_process(state)
            action = self.q_net(state).argmax().item()
        self.epsilon -= 0.00005
        self.epsilon = max(self.epsilon, 0.01)
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        if self.dqn_type == "DoubleDQN":
            max_action = self.q_net(next_states).max(1)[1].view(-1,1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1



# ppo组件
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class  PolicyNet(torch.nn.Module):
    def __init__(self, action_dim):
        super(PolicyNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_A = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        A = self.fc_A(F.relu(self.conv(x)).view(x.size(0), -1))
        return F.softmax(A, dim = 1)

class ValueNet(torch.nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_V = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv(x)).view(x.size(0), -1)

        V = self.fc_V(x)
        return V

class PPO:
    def __init__(self,  input_shape, action_dim, actor_lr, critic_lr, lmbda, epochs, gamma, eps, device):
        super(PPO, self).__init__()
        self.actor = PolicyNet(action_dim).to(device)
        self.critic = ValueNet().to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.action_dim = action_dim
        self.image_shape = (input_shape[0], input_shape[1])
        self.image_stack = input_shape[2]

    def state_transform(self, pre_state = None, state = None):
        img = Image.fromarray(state, "RGB")
        img = img.resize(self.image_shape).convert('L')
        img = np.asarray(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0])
        if pre_state is None:
            next_state = np.array([img] * self.image_stack)
        else:
            next_state = np.append(pre_state[1:], [img], axis=0)
        return next_state

    def state_process(self, state):
        state =  torch.Tensor(state).to(self.device)
        state = state.unsqueeze(0)
        return state

    def take_action(self, state):
        state = self.state_process(state)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        print(next_states.shape)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

class PPO_simplize_agent:
    def __init__(self, net):
        self.net = net
        self.epsilon = 1
        self.image_shape = [84, 84]
        self.image_stack = 4
        self.action_dim = 6
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def state_transform(self, pre_state = None, state = None):
        img = Image.fromarray(state, "RGB")
        img = img.resize(self.image_shape).convert('L')
        img = np.asarray(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0])
        if pre_state is None:
            next_state = np.array([img] * self.image_stack)
        else:
            next_state = np.append(pre_state[1:], [img], axis=0)
        return next_state

    def state_process(self, state):
        state =  torch.Tensor(state).to(self.device)
        state = state.unsqueeze(0)
        return state

    def take_action(self, state):
        state = self.state_process(state)
        print(state.shape)
        probs = self.net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action







