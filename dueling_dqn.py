import random
import time
import gym
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from core import DQN, simplize_agent
from multiprocessing import Process, Pipe

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

def process_env(env_name, pipe):
    print("子进程 : %d " %os.getpid(), end='\n')
    env = gym.make(env_name).unwrapped
    state = env.reset()
    lives = 3
    episode_return = 0
    while 1:
        net = pipe.recv()
        agent = simplize_agent(net.cuda())
        action = agent.take_action(state)
        state = agent.state_transform(None, state)
        next_state, reward, done, info = env.step(action)
        env.render()
        next_state = agent.state_transform(state, next_state)
        dead = info['ale.lives'] < lives
        lives = info['ale.lives']
        episode_return += reward
        if dead:
            reward = -60
        data = [state, action, reward, next_state, done, episode_return]
        pipe.send(data)
        if done:
            state = env.reset()
            lives = 3
            episode_return = 0

if __name__ == '__main__':
    env_name = "SpaceInvaders-v0"
    num_process = 4
    env =gym.make(env_name).unwrapped
    lr = 2e-3
    num_episodes = 1e4
    gamma = 0.98
    epsilon = 1
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    action_dim = env.action_space.n
    input_size = [84, 84, 4]
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    agent = DQN(input_size, action_dim, lr, gamma, epsilon, target_update, buffer_size, device, "DuelingDQN")

    #load_net = int(input("load or not(if this is your first time to run, input 0, else 1):"))
    #if load_net == 1:
    agent.q_net = torch.load(r'E:\上课\RL\atari\dueling_dqn_model.pkl')
    agent.target_q_net = torch.load(r'E:\上课\RL\atari\dueling_dqn_model.pkl')
    pipe_dict = dict((i, (child_pipe, main_pipe)) for i in range(num_process) for child_pipe, main_pipe in (Pipe(), ))

    s = time.time()
    [pipe_dict[j][1].send(agent.q_net) for j in range(num_process)]
    child_process_list = []
    for i in range(num_process):
        p = Process(target=process_env, args=("SpaceInvaders-v0", pipe_dict[i][0], ))
        child_process_list.append(p)
        p.daemon = True
        p.start()
    try:
        with tqdm(total=num_episodes) as pbar:
            flag = False
            for episode in range(1, int(num_episodes)):
                for i in range(num_process):
                    [state, action, reward, next_state, done, episode_return] = pipe_dict[i][1].recv()
                    flag = episode_return > 400
                    pbar.set_postfix({'return': '%.3f' %episode_return})
                    agent.ReplayBuffer.add(state, action, reward, next_state, done)
                if flag:
                    break
                if episode % minimal_size == 0:
                    b_s, b_a, b_r, b_ns, b_d = agent.ReplayBuffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                       'dones': b_d}
                    agent.update(transition_dict)

                if episode % 100 == 0:
                    torch.save(agent.q_net, "dueling_dqn_model.pkl")
                pbar.update(1)
                [pipe_dict[i][1].send(agent.q_net) for i in range(num_process)]
            env.close()
    except KeyboardInterrupt:
        print("over")
        env.close()
        #torch.save(agent.q_net, "model.pkl")
    [p.terminate() for p in child_process_list]
    e = time.time()
    print(e - s)