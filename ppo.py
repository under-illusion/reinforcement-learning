import random
import time
import gym
import os
import numpy as np
from tqdm import tqdm
import torch
from core import PPO, PPO_simplize_agent
from multiprocessing import Process, Pipe

def process_env(env_name, pipe):
    print("子进程 : %d " %os.getpid(), end='\n')
    env = gym.make(env_name).unwrapped
    state = env.reset()
    lives = 3
    episode_return = 0
    while 1:
        actor_net = pipe.recv()
        agent = PPO_simplize_agent(actor_net.cuda())

        state = agent.state_transform(None, state)
        print(state.shape)
        action = agent.take_action(state)

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
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1e4
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2

    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    action_dim = env.action_space.n
    input_size = [84, 84, 4]
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    agent = PPO(input_size, action_dim, actor_lr, critic_lr, lmbda, epochs, gamma, eps, device)

    #load_net = int(input("load or not(if this is your first time to run, input 0, else 1):"))
    #if load_net == 1:
    pipe_dict = dict((i, (child_pipe, main_pipe)) for i in range(num_process) for child_pipe, main_pipe in (Pipe(), ))

    s = time.time()
    [pipe_dict[j][1].send(agent.actor) for j in range(num_process)]
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
                    pbar.set_postfix({'return': '%.3f' % episode_return})
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                if flag:
                    break
                agent.update(transition_dict)
                if episode % 100 == 0:
                    torch.save(agent.actor, "actor_model.pkl")
                    torch.save(agent.critic, "critic_model.pkl")
                pbar.update(1)
                [pipe_dict[i][1].send(agent.actor) for i in range(num_process)]
            env.close()
    except KeyboardInterrupt:
        print("over")
        env.close()
        #torch.save(agent.q_net, "model.pkl")
    [p.terminate() for p in child_process_list]
    e = time.time()
    print(e - s)