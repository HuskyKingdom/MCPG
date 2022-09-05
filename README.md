

<br />
<div align="center" id="readme-top">
  
 <br />
<img src="https://www.gymlibrary.dev/_images/cart_pole.gif" width="700" height="300"></br>
</br>
  <h1 align="center">Monte-Carlo Policy Gradient</h1>

  <p align="center" >
  Distributed under the MIT License.

This project is an implementation of Monte-Carlo Policy Gradient algorithm to address CartPole-v0 problem under Open-AI GYM environment.
</br>
<br />
<a href="https://yuhang.topsoftint.com">Contact me at: <strong>yuhang@topsoftint.com</strong></a>

<a href="https://yuhang.topsoftint.com"><strong>View my full bio.</strong></a>
    <br />
    <br />
  </p>
</div>






<!-- ABOUT THE PROJECT -->
## Environment
<p id="1"></p>

The environment simulation is provided by OpenAI, the action space contains 2 discrete actions, and the observation space consists of 4 continous floting features. Find more about the environment <a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">here</a>. 

## The Algorithm

### Theory 

The goal is to maximize the target function:

$$
J(\Theta) = E_{\pi_\Theta}[R]
$$

Where in Monte-Carlo evaluation method, R here stands for the expected reward of states in trajectories produced by following policy $\pi_\Theta$. Expand it further:

$$
J(\Theta) = \sum_{s \in S}{d(s)} \sum_{a \in A}{\pi_{\Theta}(s,a)R_{s,a}}
$$

We want to maximize this objective function, as the following(Using MaxLikehood trick):

$$
\nabla_{\Theta} J(\Theta) = \sum_{s \in S}{d(s)} \sum_{a \in A}{\pi_{\Theta} (s,a) * \nabla_{\Theta}log\pi_{\Theta}(s,a) * R_{s,a}} \\ 
= E_{\pi_\Theta}[\nabla_{\Theta}log\pi_{\Theta}(s,a) * r]
$$

Update the weights according to above function:

$$
\Delta \Theta = \alpha \nabla log\pi_{\Theta}(s,a)v_t
$$


Since we need to maximize $J(\Theta)$, the Adam optimazor of pytorch updates the weights by negative direction by defult, we need to add an negative operation in front of the eqution, the final weights updatng can be written as:

$$
\Delta \Theta = - \alpha \nabla log\pi_{\Theta}(s,a)v_t
$$

The policy $\pi$ in this case is parameterized by $\Theta$, where in my implementation, I have used a single Neural Network to represent it. The network got the following architecture:
$$
4_{input-states} \to^{linear} 10_{hidden-nodes} \to^{Sigmoid} 2_{action-scores} \to^{Softmax} 2_{action-prob.}
$$

Please note that, by experiments, since the derivative of **Sigmoid** function has only the range of $[0,0.25]$, using sigmoid function in this case might suffer the issue of Vanishing Gradients, since the average losses are observed to be very small. Whereas the derivative of **Tanh** function has got range of $[0,1]$, it is more recommand to use tanh activation instead of sigmoid.

### Algorithm
A environment trajectory is sampled every episode $t$ times based on the current policy network, from the initial state up to the episode ends, in the form of the following:

$$
\tau_{\pi_\Theta}=[S_0,A_0,R_0,...,S_{t-1},A_{t-1},R_{t-1}]
$$

We then need to calculate a $E_{\pi_\Theta}[-\nabla_{\Theta}log\pi_{\Theta}(s,a) * r]$ to backward propagatee through the whole network to update weights. In my implementation, $-log\pi_{\Theta}(s,a)$ is calculated by _torch.nn.CrossEntropyLoss(y_input,y_target)_, which returns the following:

$$
-\sum_{y \in outputs}{y_{target}*log(Softmax(y_{input}))}
$$

Where the $y_{ture}$ is equal to 1 in true action, and equal to 0 in all other actions, plus, $Softmax(y_{input}) = \pi_{\Theta}(s,a)$, that makes the equation is equavalent to:

$$
-1*log(\pi_{\Theta}(s,a))
$$

Which is exactly what we want, passing all the $S_t$ in the sampled trajectory into the network to obtain collections of actions, rewards and the network outputs for each actions, in each step, we can calculate $- log\pi_{\Theta}(s,a)v_t$ for each time step, then compute the average of these values to get $E_{\pi_\Theta}[-log\pi_{\Theta}(s,a) * r]$, finally perform backpropagation by the average $-log\pi_{\Theta}(s,a)v_t$, let the torch.tensor.autogrand to handle the calculations of deravatives.

Run the algorithm by sample $N$ episodes, update the weights in each episode, the optimal model is expected to be obtained.

## Evaluations

The algorithm converges in ~500 episodes after the Sigmoid activation is used, and converges in ~300 episodes while using Tanh function.

## References / Related Works
<p id="6"></p>

[0] https://blog.csdn.net/qq_41626059/article/details/115196106

[1] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learningwith double q-learning." Proceedings of the AAAI conference on artificialintelligence. Vol. 30. No. 1. 2016.

[2] https://gym.openai.com/


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License.

<p align="right">(<a href="#readme-top">back to top</a>)</p>





