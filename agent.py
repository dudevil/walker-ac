#!/usr/bin/python3
import numpy as np
import gym
import random
import theano 
import theano.tensor as T
import lasagne
from collections import deque


class ReplayMemmory:

    def __init__(self, size=100000):
        self.size = size
        self.queue = deque(maxlen=size)
        random.seed(42)

    @property
    def full(self):
        return len(self.queue) == self.size

    def add(self, state, action, reward, next_state, done):
        self.queue.append((state, action, reward, next_state, done))

    def sample(self, size=512):
        batch = random.sample(self.queue, size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for s, a, r, ns, d in batch:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(float(d))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)          
    

class Actor:

    def __init__(self, env, learning_rate=0.0001):
        input_state = T.dmatrix('input_state')
        grad_from_critic = T.dmatrix('c_grads')

        
        state_in = lasagne.layers.InputLayer((None, 24), input_state)

        l_hid1 = lasagne.layers.DenseLayer(
            state_in, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
	
        actions = lasagne.layers.DenseLayer(
            l_hid2, num_units=4,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform(),
            b=None
        )

        params = lasagne.layers.get_all_params(actions, trainable=True)
        prediction = lasagne.layers.get_output(actions, deterministic=True)

        self.predict_fn = theano.function(
            [input_state],
            prediction
        )

        grads = theano.gradient.grad(None, params, known_grads={prediction: -1 * grad_from_critic})

        updates = lasagne.updates.adam(grads, params, learning_rate=learning_rate)
        
        self.train_fn = theano.function(
            [input_state, grad_from_critic],
            prediction,
            updates=updates
        )

    def get_actions(self, states):
        return self.predict_fn(states)

    def get_action(self, state):
        return self.predict_fn(state[np.newaxis, ...]).squeeze()
    
    def train(self, states, gradients):
        return self.train_fn(states, gradients)
       

class Critic:
    
    def __init__(self, env, learning_rate=0.01):
        input_reward = T.dvector('input_reward')
        input_state = T.dmatrix('input_state')
        input_action = T.dmatrix('input_action')

        state_in = lasagne.layers.InputLayer((None, 24), input_state)
        action_in = lasagne.layers.InputLayer((None, 4), input_action)

        l_hid1 = lasagne.layers.DenseLayer(
            state_in, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        l_hid2 = lasagne.layers.DenseLayer(
            action_in, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
	
        l_sum = lasagne.layers.ElemwiseSumLayer([l_hid1, l_hid2])
	
        l_value = lasagne.layers.DenseLayer(
            l_sum, num_units=1,
            nonlinearity=None,
            W=lasagne.init.GlorotUniform())

        params = lasagne.layers.get_all_params(l_value, trainable=True)
        prediction = lasagne.layers.get_output(l_value, deterministic=True)

        loss = T.sqr(prediction - input_reward).mean()
        updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

        self.train_fn = theano.function(
	    [state_in.input_var, action_in.input_var, input_reward],
	    loss,	
	    updates=updates)

        self.prediction = theano.function(
	    [state_in.input_var, action_in.input_var],
	    prediction
	)

        self.actor_grad = theano.function(
            [input_state, input_action],
            theano.gradient.grad(prediction.sum(), input_action)
        )
	

    def train(self, states, actions, td_target):
        return self.train_fn(states, actions, td_target)

    def predict(self, state, action):
        return self.prediction(state, action).squeeze()

    def actor_gradient(self, states, actions):
        return self.actor_grad(states, actions)


if __name__ == "__main__":
    np.random.seed(42)
    NUM_EPISODES = 1000
    R = 0. # total reward
    gamma = 0.99

    env = gym.make('BipedalWalker-v2')
    # policy = EGreedyPolicy(env)
    critic = Critic(env)
    actor = Actor(env)
    replay_mem = ReplayMemmory(size=1000)
    
    try:
        for ep in range(NUM_EPISODES):
            ep_R = 0.
	    
            done = False
            curr_s = env.reset()
            losses = []
            while not done:
                action = actor.get_action(curr_s)
                s, r, done, _ = env.step(action)
                done = done or env.hull.position.y < 5.0
                print(action)
                replay_mem.add(curr_s, action, r, s, done) 
                curr_s = s
               
                if replay_mem.full:
                    states, actions, rewards, next_states, dones = replay_mem.sample()

                    next_q = critic.predict(next_states, actor.get_actions(next_states))
                    td_targets = rewards + gamma * next_q * dones
                    c_loss = critic.train(states, actions, td_targets)
                    losses.append(c_loss)

                    na = actor.get_actions(states)
                    actor_grads = critic.actor_gradient(states, na)
                    actor.train(states, actor_grads)

            ep_R += r
			        
            print("[Episode {}] Got reward of {} critic loss was {}".format(ep, ep_R, np.mean(losses)))
            R += ep_R

    except KeyboardInterrupt:
        print("Got total reward of {} in {} episodes".format(R, ep))
