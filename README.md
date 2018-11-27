# Reinforcement Learning Algorithms in PyTorch

Inspired by [Spinning Up](https://spinningup.openai.com/en/latest/), we will implement various salient reinforcement learning (RL) 
 algorithms in PyTorch.
 
| ![example](./docs/final_model.gif) |
| :---: |
| *DQN Playing Flappy Bird* |

In its current version, I get the following performance averaged over 20 episodes:

| Algorithm | Game |Performance |
| :----:       | :---: |:----:         |
| DQN Vanilla  | FlapPy Bird| 119.1   |

## Algorithms Implmented
- [x] [Vanilla DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [ ] [DRQN](https://arxiv.org/pdf/1507.06527)
- [ ] [Mini-Rainbow DQN](https://arxiv.org/pdf/1507.06527)
  -[ ] [Dueling DQN]()
  -[ ] [Prioritized Replay Experience]()
  -[ ] [Double Q-Learning]()
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

## Usage

To visualise the agent with pre-trained weights, simply type:
```bash
python runFlappy.py
```
The various options are as follows:
```bash
python runFlappy.py

## High Level Settings
--mode 'test'                               # one of 'train' or 'test'
--testfile './models/trained_params.pth'    # location of pretrained model (if 'test' is selected)
--slow False                                # run at native 30 FPS (seems less stable)

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
```

## TODO:
* Add more
    * Games (Pong, OpenAI Gym)
        * Make the model [OpenAI Gym friendly](https://github.com/lusob/gym-ple)?
    * [Algorithms](https://spinningup.openai.com/en/latest/spinningup/spinningup.html#learn-by-doing)
    

