# Reinforcement Learning Algorithms in PyTorch

Inspired by [Spinning Up](https://spinningup.openai.com/en/latest/), we will implement various salient reinforcement learning (RL) 
 algorithms in PyTorch. This is a "fork" of my original [Deep FlappyBird repo](https://github.com/fiorenza2/dqnflappy).

| ![example](./docs/final_model.gif) |
| :---: |
| *DQN Playing Flappy Bird* |

In its current version, I get the following performance:

| Algorithm | Game |Performance | Episodes  |
| :----:       | :---: |:----:         |  :---: |
| DQN Vanilla  | FlapPy Bird | 66.5   |   20 |
| DQN Vanilla   | CartPole-v0 | 200.0 | 100 |

## Algorithms Implmented
- [x] [Vanilla DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [ ] [Mini-Rainbow DQN](https://arxiv.org/pdf/1507.06527)
  - [ ] [Dueling DQN](https://arxiv.org/pdf/1511.06581.pdf)
  - [ ] [Prioritized Replay Experience](https://arxiv.org/pdf/1511.05952.pdf)
  - [ ] [Double Q-Learning](https://arxiv.org/pdf/1509.06461.pdf)
- [ ] [DRQN](https://arxiv.org/pdf/1507.06527)
- [ ] [Vanilla Policy Gradient](http://rll.berkeley.edu/deeprlcoursesp17/docs/lec2.pdf)
- [ ] [A2C](https://arxiv.org/pdf/1602.01783.pdf)
- [ ] [PPO](https://arxiv.org/pdf/1707.06347.pdf)
- [ ] [DDPG](https://arxiv.org/pdf/1509.02971.pdf)

## Requirements

* Python 3.6
* NumPy 1.15
* PyTorch 0.4.1
* OpenCV-Python 3.4.3
* TensorboardX 1.4
* PyGame 1.9.4
* [PyGame Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment) 0.0.1
* OpenAI Gym 0.10.5

## Usage

To visualise the agent on FlappyBird with pre-trained weights, simply type:
```bash
python runFlappy.py
```
The various options are as follows:
```bash
python runExp.py

## High Level Settings
--exp FlappyBird-v0                         # one of 'CartPole-v0' or 'FlappyBird-v0'
--mode 'test'                               # one of 'train' or 'test'
--testfile './params/trained_params_gym_fb.pth'    # location of pretrained model (if 'test' is selected)
--visualise False                           # visualise the trained agent

## Training Settings
--frame_skip 3                              # how many frames will be skipped and the same action will be applied
--reward_shaping True                       # include a living reward
--frame_stack 4                             # how many frames to stack
--gamma 0.9                                 # future return discounting
--batch_size 32                             # size of batch from replay memory buffer
--memory_size 100000                        # size of the entire replay memory buffer
--max_ep_steps 1000000                      # how many steps we can spend in a single episode
--reset_target 10000                        # how many steps before syncing the target q-network
--final_exp_frame 500000                    # how many steps before we settle on the final exploration value
--save_freq 10000                           # how many steps between saving model parameters
--num_episodes 100000000                    # how many episodes to run in total (basically infinite)
--num_samples_pre 3000                      # how many samples under a random policy to initially load into the replay memory
--weight_decay 0                            # weight decay (L2 regularisation) amount
--lr_scheduler None                         # takes a list if provided; divides learning rate by 10 for every entry in the list
```

## TODO:
* Document the helper functions:
    * viewAgent.py
    * trainReplicate.py
* Add more
    * Games (Pong, OpenAI Gym)
    * [Algorithms](https://spinningup.openai.com/en/latest/spinningup/spinningup.html#learn-by-doing)
    

