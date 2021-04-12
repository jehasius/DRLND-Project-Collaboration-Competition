# Collaboration and Competition - Udacity Deep Reinforcement Learning Expert Nanodegree
#### J. Hampe 04/2021.
---

[1. Introduction](#intro)  
[2. Getting Started](#start)  
[3. Learning Algorithm](#algo)  
[4. Plot of Rewards](#plot)  
[5. Simulation](#sim)  
[6. Ideas for future work](#future)  

[//]: # (Image References)
[image1]: ./media/score_episode.png "Score over Episode"
[image2]: ./media/game_set_match.mp4 "Trained Agent"

<a name="intro"></a>
## 1. Introduction 
This project was a part of the Udacity Deep Reinforcement Learning Expert Nanodegree course. The main task of this project, called Collaboration and Competition, was to solve a multi agent problem. The algorithm should be able to train two artificial agents to play tennis in an Unity game engine enironment. To find out more about the used Unity environment please look at the [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) explainations. The used MADDPG - Multi Agent Deep Deterministic Policy Gradient appraoch is a multi-agent policy gradient algorithm where agents learn a centralized critic based on the observations and actions of all agents, for detailed information see paper [Multi-Agent Actor-Critic for MixedCooperative-Competitive Environments](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)

### 1.1 The Environment 
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

### 1.2 The Observation Space 
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

Specifically, After each episode, rewards that each agent received were add up (without discounting), to get a score for each agent. 
This yields 2 (potentially different) scores. The maximum of these 2 scores was taken. This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.
Here are some information about the environment:
```python
Number of agents: 2
Size of each action: 2
There are 2 agents. Each observes a state with length: 24
The state for the first agent looks like: 
[ 0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.         -6.65278625 -1.5
 -0.          0.          6.83172083  6.         -0.          0.        ]
```
### 1.3 AI - Machine Learning - Reinforment Learning
Machine learning, is a branch of artificial intelligence, focuses on learning patterns from data. The three main classes of machine learning algorithms include: unsupervised learning, supervised learning and reinforcement learning. Each class of algorithm learns from a different type of data. You can find a good overview over the main concepts of machine learning at [Unity Background-Machine-Learning](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Background-Machine-Learning.md)

<a name="start"></a>
## 2. Getting Started 
At frist set up your python environment by following the [Deep Reinforcement Learning Nanodegree - dependencies](https://github.com/udacity/deep-reinforcement-learning#dependencies)!
To set up and run the environment please follow the generell instructions in the [README.md](./README.md) and jupyter notebook [Continuous_Control.ipynb](./Collaboration-and-Competition.ipynb). The Jupyter notebook also contains the whole project algorithms.

<a name="algo"></a>
## 3. Learning Algorithm 
The used MADDPG - Multi Agent Deep Deterministic Policy Gradient appraoch is a multi-agent policy gradient algorithm where agents learn a centralized critic based on the observations and actions of all agents, for detailed information see paper [Multi-Agent Actor-Critic for MixedCooperative-Competitive Environments](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)

### 3.1. Network Architecture 
The two networks, one for the policy to map states to actions and one for a critic to map state and action pairs as Q-values can be found in the [model.py](./model.py) file.

Build an actor (policy) network that maps states -> actions
```python
Actor(
  (fc1): Linear(in_features=33, out_features=128, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=4, bias=True)
)
```
Build a critic (value) network that maps (state, action) pairs -> Q-values
```python
Critic(
  (fcs1): Linear(in_features=33, out_features=128, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=132, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)
```

### 3.2 DRL Hyperparameter 
To tune the a Deep Reinforcement Learning system you are always have a bunch of hyperprameter in your algoritm. With this hyperparameters you can influence and optimize the hole learnig prozess. This is a very challenging project and it took me a long time to find the right hyperparameter combination. Thanks to the knowlage base of Udacity I found useful hints and commends to tune the hyperparameters and also the network feature sizes.

```python
config.update_every = 1           # update rate
config.batch_size = 512           # minibatch size
config.buffer_size = int(1e6)     # replay buffer size
config.discount = 0.99            # discount factor
config.tau = 0.2                  # for soft update of target parameters
config.seed = 2                   # random seed
config.lr_actor = 1e-4            # learning rate of the actor
config.lr_critic = 1e-4           # learning rate of the critic
config.action_size = action_size  # Two continuous actions are available.
config.num_agents = num_agents    # There are 2 agents,
config.state_size = state_size    # each observes a state with length: 24
```
### 3.3 Build and train the MADDPG algorithm 
The MADDPG algorithm and trainig process was realized with an deep learning agent [ddpg_agent.py](./ddpg_agent.py), the model [model.py](./model.py) file and the [Continuous_Control.ipynb](./Collaboration-and-Competition.ipynb). The model parameter were saved and can be accessed for further tests an explorations.


<a name="plot"></a>
## 4. Plot of Rewards 
The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).
The Environment was solved in Environment solved in 45 episodes with an average score of 30.43.
```python
Episode: 0   Average Score: 0.00	Critic Loss: 0.000000	Actor Loss: 0.000000	t_step:       15
Episode: 50  Average Score: 0.00	Critic Loss: 0.000015	Actor Loss: -0.005425	t_step:      797
Episode: 100 Average Score: 0.00	Critic Loss: 0.000034	Actor Loss: -0.000644	 t_step:     1558
Episode: 150 Average Score: 0.01	Critic Loss: 0.000033	Actor Loss: -0.007449	 t_step:    50461
...
Episode: 1350 Average Score: 0.20	Critic Loss: 0.000352	Actor Loss: -0.143723	 t_step:    55027
Episode: 1400 Average Score: 0.31	Critic Loss: 0.000421	Actor Loss: -0.193846	 t_step:    63224
Episode: 1418 Average Score: 0.51  Score: 2.60  Critic Loss: 0.000948  Actor Loss: -0.288310  
t_step:    72316
Environment solved in 1418 episodes!	Average Score: 0.51
```

In the picture below you can see the approproate plot of the rewards during the training process as score over episodes.

![alt text][image1] 

## 5. Simulation<a name="sim"></a>
The model was successfully trained and the agents were able to play tennis against each other.
A video of the successful tennis playing agents was recorded and con be seen here:
<!-- a![Video of the successful tennis playing agents][image2]-->
   
<video width="640" height="480" controls>
  <source src="./media/game_set_match.mp4" type="video/mp4">
</video>


```python
Score (max over agents) from episode 1: 2.600000038743019
Score (max over agents) from episode 2: 0.0
Score (max over agents) from episode 3: 2.600000038743019
Score (max over agents) from episode 4: 2.7000000402331352
Score (max over agents) from episode 5: 2.600000038743019
Score (max over agents) from episode 6: 2.600000038743019
Score (max over agents) from episode 7: 2.600000038743019
Score (max over agents) from episode 8: 2.7000000402331352
Score (max over agents) from episode 9: 2.600000038743019
```

<a name="future"></a>
## 6. Ideas for future work
The alorithm is running quit well but we can try to tune hyper-parameters.
Testing different network architecture that uses other network architectures.
Solving the second optional task to train agents to play soccer.


