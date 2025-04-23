import time
import numpy as np
from maze.Maze import Maze

class Q_Learning():
    def __init__(self,env,seed = 42,eps=(0.3,0.4),gamma= 0.95,stepsize = 0.05):
        self.env = env
        self.eps = eps[0]
        self.eps_decay = eps[1]
        self.gamma = gamma
        self.stepsize = stepsize
        self.Q = np.zeros((env.size*env.size,env.action_space.n))
        np.random.seed(seed)
    
    def greedy(self,s):
        '''
        Greedy policy in regards to Q

        Parameters:
        s: state index

        Returns:
        The index of the action that with the maximum state action value given the current state
        '''
        return np.argmax(self.Q[s])

    def eps_greedy(self,s):
        '''
        Epsilon greedy policy in regards to Q

        Parameters:
        s: state index

        Returns:
        The index of the action chosen
        '''
        if np.random.random() < self.eps:
            return np.random.randint(self.Q.shape[1])
        else:
            return self.greedy(s)
    
    def episode(self):
        '''
        Implementation of Q-Learning algorithm based on pseudo code from Chapter 6 of "Reinforcement Learning: An Introduction",
        with an addition of using a decaying epsilon
        '''
        time_steps = 0 
        s,_ = self.env.reset()
        terminated = False

        action = self.eps_greedy(s)
        while not terminated:
            s_prime,reward,terminated,_,_ = self.env.step(action)
            action_prime = self.eps_greedy(s_prime)
            self.Q[s,action] = self.Q[s,action] + self.stepsize * (reward + (self.gamma * self.Q[s_prime,np.argmax(self.Q[s_prime])]) - self.Q[s,action])
            s = s_prime
            action = action_prime
            time_steps += 1

        self.eps = self.eps * self.eps_decay 
        return

    def train(self,num_episodes:int):
        '''
        Function to run through 'num_episodes' episodes and watch for how optimized the agent's policy becomes over time

        Parameters:
        num_episodes: The number of episodes to train over
        '''
        for ep in range(num_episodes):
            self.episode()

        return 
    
    def watch(self):
        '''
        Function for the user to watch the agent go through the environment with the trained policy
        '''
        watch_env = Maze(render_mode="human")
        
        s,_ = watch_env.reset()
        terminated = False
        while not terminated:
            action = self.eps_greedy(s)
            s,reward,terminated,_,_ = watch_env.step(action)
            time.sleep(0.3)
        
        watch_env.close()
        return

if __name__ == "__main__":
    env = Maze()
    q_learning = Q_Learning(env)
    q_learning.train(500)
    env.close()
    q_learning.watch()