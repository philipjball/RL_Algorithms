import os
import random
import datetime
from collections import namedtuple, deque
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tensorboardX import SummaryWriter

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(list(self.memory), batch_size) # https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3
        return Transition(*(zip(*batch))) # https://stackoverflow.com/questions/7558908/unpacking-a-list-tuple-of-pairs-into-two-lists-tuples

    def __len__(self):
        return len(self.memory)


class DQNDense(nn.Module):

    def __init__(self, action_set, frame_stack=1, input_dim=4):
        super(DQNDense, self).__init__()
        assert frame_stack == 1, "there can only be one frame in dense network experiments (i.e., non-image-based)"
        num_actions = action_set.n
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)


class DQNConv(nn.Module):

    def __init__(self, action_set, frame_stack=4, input_dim=84):
        super(DQNConv, self).__init__()
        num_actions = action_set.n
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(self.calculate_final_size(input_dim, input_dim), 512)
        self.linear2 = nn.Linear(512, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x / 255.                  # normalise the input to [0,1]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        return self.linear2(x)

    @staticmethod
    def calculate_conv_out(input_height, input_width, kernel_size, stride):
        def convcalc(x, ks, s): return (x - ks) / s + 1
        h_out = convcalc(input_height, kernel_size, stride)
        w_out = convcalc(input_width, kernel_size, stride)
        assert h_out.is_integer(), "h_out is not an integer, is in fact %r" % h_out
        assert w_out.is_integer(), "w_out is not an integer, is in fact %r" % w_out
        return int(h_out), int(w_out)

    def calculate_final_size(self, input_height, input_width):
        ho1, wo1 = self.calculate_conv_out(input_height, input_width, 8, 4)
        ho2, wo2 = self.calculate_conv_out(ho1, wo1, 4, 2)
        ho3, wo3 = self.calculate_conv_out(ho2, wo2, 3, 1)
        return int(ho3 * wo3 * 64)


class DQNAgent(object):

    def __init__(self, action_set, frame_stack=4, input_dim=84, conv=True):
        self.frame_stack = frame_stack
        self.conv = conv
        if self.conv:
            dqn = DQNConv
        else:
            dqn = DQNDense
        self.q_network = dqn(action_set=action_set, frame_stack=frame_stack, input_dim=input_dim).to(device)
        self.q_target = dqn(action_set=action_set, frame_stack=frame_stack, input_dim=input_dim).to(device)
        self.eps = 1.0
        self.action_set = action_set

    def update_target(self):
        self.q_target.load_state_dict(self.q_network.state_dict())

    def random_action(self):
        return self.action_set.sample()

    def get_action(self, in_frame):
        # eps-greedy exploration
        if np.random.rand() < self.eps:     # if rand number is greater than eps, then explore
            return self.random_action()
        else:
            return int(torch.argmax(self.q_network(in_frame)))


class DQNLoss(nn.Module):

    def __init__(self, q_network, q_target, action_set, gamma=0.9):
        super(DQNLoss, self).__init__()
        self.q_network = q_network
        self.q_target = q_target
        self.action_set = action_set
        self.gamma = gamma
        self.loss = nn.SmoothL1Loss()

    def forward(self, transition_in):
        states = torch.tensor(np.stack(transition_in.state), dtype=torch.float, device=device).squeeze()
        actions_index = torch.tensor(transition_in.action, dtype=torch.long, device=device)
        next_states = torch.tensor(np.stack(transition_in.next_state), dtype=torch.float, device=device).squeeze()
        rewards = torch.tensor(transition_in.reward, dtype=torch.float, device=device)
        done = torch.tensor(transition_in.done, dtype=torch.float, device=device)
        pred_return_all = self.q_network(states)
        pred_return = pred_return_all.gather(1, actions_index.unsqueeze(1)).squeeze()  # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
        one_step_return = rewards + self.gamma * self.q_target(next_states).detach().max(1)[0] * (1 - done)
        return self.loss(pred_return, one_step_return)


class Runner(object):
    """Runs the experiments (training and testing inherit from this)"""
    def __init__(self, env, agent, downscale=84, max_ep_steps=1000000, frame_skip=3):
        self.env = env
        self.agent = agent
        self.transformer = transforms.Compose([transforms.Grayscale(), transforms.Resize([downscale,downscale])])
        self.total_steps = 0
        self.max_ep_steps = max_ep_steps
        frame_stack = self.agent.frame_stack
        self.frame_stacker = deque(maxlen=frame_stack)
        self.frame_skip = frame_skip

    def preprocess_image(self, input_image):
        """Does a few things:
        0. Checks if it is an image
        1. Convert to GrayScale
        2. Reshape image to 84x84
        3. Convert to a uint8 numpy array
        """
        if self.agent.conv:
            input_image = Image.fromarray(input_image)
            input_image = self.transformer(input_image)
            input_image = np.array(input_image, dtype=np.uint8)
        return input_image

    def episode(self):
        """Runs an episode"""
        raise NotImplementedError

    def run_experiment(self):
        """Runs a number of epsiodes"""
        raise NotImplementedError

    def get_recent_states(self):
        assert len(self.frame_stacker) == self.agent.frame_stack, "Not filled enough frames for stacking!"
        return np.array(self.frame_stacker)


class Trainer(Runner):
    def __init__(self, env, agent, memory_func, batch_size=32, downscale=84, frame_skip=3, num_samples_pre=30000,
                 memory_size=50000, max_ep_steps=1000000, reset_target=10000, final_exp_frame=100000, gamma=0.9,
                 optimizer=optim.Adam, save_freq=100000):
        super(Trainer, self).__init__(env, agent, downscale, max_ep_steps, frame_skip)
        self.memory = memory_func(memory_size)
        self.optimizer = optimizer(self.agent.q_network.parameters(), lr=1e-4)
        self.batch_size = batch_size
        self.reset_target = reset_target
        self.final_exp_frame = final_exp_frame  # Final frame for exploration (whereby we go to eps = 0.01 henceforth)
        self.reward_per_ep = []
        self.tb_writer = SummaryWriter()
        self.loss = DQNLoss(self.agent.q_network, self.agent.q_target, self.agent.action_set, gamma=gamma)
        self.save_freq = save_freq
        self.num_samples_pre=num_samples_pre
        self.episode_cnt = 0

    def episode(self):
        steps = 0
        done = False
        # Do resets
        state = self.preprocess_image(self.env.reset())
        self.frame_stacker.clear()
        rewards = []
        self.frame_stacker.append(state)                        # Append the first state into the stacker
        # Start training episode
        while done is False and steps < self.max_ep_steps:
            if steps < self.agent.frame_stack:                  # Need to fill frame stacker
                action = 0
            else:
                psi_state = self.get_recent_states()
                psi_state_tensor = torch.tensor(psi_state, dtype=torch.float, device=device).unsqueeze(0)
                action = self.agent.get_action(psi_state_tensor)
            reward = 0  # zero the reward for frame skipping
            for i in range(self.frame_skip):
                obs, r, done, _ = self.env.step(action)
                reward += r
                if done:
                    break
            rewards.append(reward)
            state = self.preprocess_image(obs)
            self.frame_stacker.append(state)
            if steps > self.agent.frame_stack:
                psi_next_state = self.get_recent_states()
                self.memory.push(psi_state, action, psi_next_state, reward, done)
            self.total_steps += 1
            steps += 1
            if len(self.memory) <= self.num_samples_pre:    # if there's not enough samples in the memory, don't backprop
                continue
            self.set_eps()                                  # decrease exploration
            trans_batch = self.memory.sample(self.batch_size)
            loss = self.loss(trans_batch)
            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.total_steps % self.reset_target == 0:   # sync up the target to our current q network
                # print("Updating Target")
                self.agent.update_target()
            # prints and logs
            self.tb_writer.add_scalar('DQN_' + self.env.spec.id + '/loss', float(loss), self.total_steps)
            self.tb_writer.add_scalar('DQN_' + self.env.spec.id + '/reward_per_ep', np.mean(self.reward_per_ep), self.total_steps)
            self.tb_writer.add_scalar('DQN_' + self.env.spec.id + '/epsilon', self.agent.eps, self.total_steps)
            self.tb_writer.add_scalar('DQN_' + self.env.spec.id + '/cum_episodes', self.episode_cnt, self.total_steps)
            if self.total_steps % 1000 == 0:
                print('Loss at %d steps is %.5f' % (self.total_steps, float(loss)))
                print('Mean reward per episode is:', np.mean(self.reward_per_ep))
                print('epsilon is', self.agent.eps)
                self.reward_per_ep = []
            if self.total_steps % self.save_freq == 0:
                self.save_model()

        self.reward_per_ep.append(np.sum(rewards))
        if len(self.memory) > self.num_samples_pre:
            self.episode_cnt += 1

    def set_eps(self):
        """Function to decrease exploration linearly as we increase steps"""
        self.agent.eps = np.max([0.01, (0.01 + 0.99 * (1 - (self.total_steps - self.num_samples_pre)
                                                       / self.final_exp_frame))])

    def run_experiment(self, num_episodes=1000):
        print('Beginning Training...')
        for i in range(num_episodes):
            self.episode()
        self.env.close()

    def save_model(self):
        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d-%H-%M")
        steps_str = '_%dsteps' % self.total_steps
        print('Saving Model at %d steps...' % self.total_steps)
        if not os.path.exists('./params'):
            os.makedirs('./params')
        torch.save(self.agent.q_network.state_dict(), './params/params_dqn_' + self.env.spec.id + '_' + now_str +
                   steps_str + '.pth')


class Tester(Runner):

    def __init__(self, env, agent, downscale, max_ep_steps, frame_skip, visualise):
        super(Tester, self).__init__(env, agent, downscale, max_ep_steps, frame_skip)
        self.visualise = visualise

    def load_model(self, path):
        params = torch.load(path, map_location={'cuda:0': 'cpu'})
        self.agent.q_network.load_state_dict(params)

    def episode(self):
        steps = 0
        done = False
        # Do resets
        state = self.preprocess_image(self.env.reset())
        self.frame_stacker.clear()
        rewards = []
        self.frame_stacker.append(state)
        while done is False and steps < self.max_ep_steps:
            # Need to fill frame stacker
            if self.visualise:
                self.env.render()
            if steps < self.agent.frame_stack:
                action = 0
            else:
                psi_state = self.get_recent_states()
                psi_state_tensor = torch.tensor(psi_state, dtype=torch.float, device=device).unsqueeze(0)
                action = self.agent.get_action(psi_state_tensor)
            reward = 0  # zero the reward for frame skipping
            for i in range(self.frame_skip):
                obs, r, done, _ = self.env.step(action)     # NB: Even if the environment is 'done', it will still accept actions and continue returning 'done'
                reward += r
            rewards.append(reward)
            state = self.preprocess_image(obs)
            self.frame_stacker.append(state)
            self.total_steps += 1
            steps += 1
        total_reward = np.sum(rewards)
        print('This episode had %.2f reward' % total_reward)
        return total_reward

    def run_experiment(self, num_episodes=1000):
        reward_list = []
        print('Beginning Testing...')
        for i in range(num_episodes):
            reward_list.append(self.episode())
        self.env.close()
        print('Average reward over %d episodes is: %.1f' % (num_episodes, np.mean(reward_list)))
