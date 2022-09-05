from turtle import forward
import gym
import torch.nn as nn
import torch as tc
from torch.nn import functional as F
import numpy as np
import math

# Environment Initialization
env = gym.make("CartPole-v1")
init_observations = env.reset()

class PG_Appx_NN(nn.Module):

    def __init__(self) -> None:

        super(PG_Appx_NN,self).__init__()

        self.fc1 = nn.Linear(4,10)
        nn.init.normal_(self.fc1.weight,0,0.3)
        nn.init.constant_(self.fc1.bias,0.1)

        self.fc2 = nn.Linear(10,2)
        nn.init.normal_(self.fc2.weight,0,0.3)
        nn.init.constant_(self.fc2.bias,0.1)

    def forward(self,x):

        # convert numpy array to pytorch tensor
        out = tc.from_numpy(x).float()

        out = self.fc1(out)
        out = F.sigmoid(out)

        out = self.fc2(out)
        feature_value = out
        out = F.softmax(out,dim=0)

        return out,feature_value

def simple_policy(ob):

    env.reset()

    is_done = False
    observations = []
    rewards = []
    action_took = []

    while(is_done != True):
        

        # using current policy network to compute action
        action_probs,feature_vals = network.forward(ob)
        cur_action = simple_action(action_probs)
        # take the action & record 
        observation, reward, done, info = env.step(int(cur_action))
        # add an additional dim for later concatenate
        observation_store = np.expand_dims(observation,axis=0)
        env.render() 
        action_took.append(cur_action)
        rewards.append(reward)
        observations.append(observation_store)

        # updates for next iteration
        ob = observation
        is_done = done
        


    
    return np.concatenate(observations,axis=0),action_took,rewards


def expected_feature(feature_values):

    # computing expected feature value
    total_obs = 0
    accum = 0
    for ob in range(len(feature_values)):
        total_obs += 1
        accum += feature_values[ob].sum()

    exp_fearture =  accum / total_obs
    return exp_fearture

def simple_action(probs):
    action = np.random.choice(a = 2, p = probs.detach().numpy())
    return action

def learn(observations,action_took,rewards):

    
    # get feature value tensor
    probs,feature_vals = network(observations)
    log_prob = criterion(feature_vals,tc.tensor(action_took))

    # accumulating reward
    acc_reward = []
    for i in range(len(rewards)):
        acc_r = 0
        for j in range(i, len(rewards)):
            acc_r += r_decay ** (j-i) * rewards[j]
        acc_reward.append(acc_r)
    acc_reward = tc.tensor(acc_reward)
    acc_reward -= acc_reward.mean()
    acc_reward /= acc_reward.std()

    log_reward = log_prob * acc_reward
    loss = log_reward.mean()

    optim.zero_grad()
    loss.backward()
    optim.step()


def train(episodes):

    for i_train in range(episodes):

        observations,action_took,rewards =  simple_policy(init_observations)
        learn(observations,action_took,rewards)
        print("Episode {} is finished, with total rewards {} ...".format(i_train,tc.tensor(rewards).sum()))

    


# network initialization & hyper-parameters
network = PG_Appx_NN()
r_decay = 0.99
lr = 0.02
optim = tc.optim.Adam(network.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss(reduction = 'none')


train(1000)

#env.close()


'''
for _ in range(10000):
    observation, reward, done, info = env.step(1)
    print(reward)
    env.render()

x = env.action_spaces 
'''

