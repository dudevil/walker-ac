#!/usr/bin/python3
import numpy as np
import gym
import random
import theano 
import theano.tensor as T
import lasagne
from lasagne.utils import floatX
import pickle
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


def copy_params(src_net, dest_net, tau=0.001):
    src_params = lasagne.layers.get_all_param_values(src_net.net)
    dest_params = lasagne.layers.get_all_params(dest_net.net)
    mtau = 1 - tau
    
    assert len(src_params) == len(dest_params)
    for src_p, dest_p in zip(src_params, dest_params):
        
        dest_p.set_value(tau * src_p + mtau * dest_p.get_value())


class TargetActor:

    def __init__(self, env):
        self.input_state = T.fmatrix('input_state')
        self.state_in = lasagne.layers.InputLayer((None, 24), self.input_state)

        self.l_hid1 = lasagne.layers.DenseLayer(
            self.state_in, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        self.l_hid2 = lasagne.layers.DenseLayer(
            self.l_hid1, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
	
        self.net = lasagne.layers.DenseLayer(
            self.l_hid2, num_units=4,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform(),
            b=None
        )

        self.output = lasagne.layers.get_output(self.net, deterministic=True)

        self.predict_fn = theano.function(
            [self.input_state],
            self.output,
            allow_input_downcast=True
        )

    def get_actions(self, states):
        return self.predict_fn(states)
        
    def get_action(self, state):
        return self.get_actions(state[np.newaxis, ...]).squeeze()

    def save_params(self):
        pass

        
class LearningActor(TargetActor):

    def __init__(self, env, learning_rate=0.0001):
        super(LearningActor, self).__init__(env)
        self.grad_from_critic = T.fmatrix('c_grads')

        params = lasagne.layers.get_all_params(self.net, trainable=True)
        grads = theano.gradient.grad(None, params, known_grads={self.output: -1 * self.grad_from_critic})
        updates = lasagne.updates.adam(grads, params, learning_rate=learning_rate)
        
        self.train = theano.function(
            [self.input_state, self.grad_from_critic],
            self.output,
            updates=updates,
            allow_input_downcast=True

        )
        self.params = params
    


class TargetCritic:
    
    def __init__(self, env, learning_rate=0.01):
        self.input_state = T.fmatrix('input_state')
        self.input_action = T.fmatrix('input_action')

        self.state_in = lasagne.layers.InputLayer((None, 24), self.input_state)
        self.action_in = lasagne.layers.InputLayer((None, 4), self.input_action)

        self.l_hid1 = lasagne.layers.DenseLayer(
            self.state_in, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        self.l_hid2 = lasagne.layers.DenseLayer(
            self.action_in, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
	
        self.l_sum = lasagne.layers.ElemwiseSumLayer([self.l_hid1, self.l_hid2])
	
        self.net = lasagne.layers.DenseLayer(
            self.l_sum, num_units=1,
            nonlinearity=None,
            W=lasagne.init.GlorotUniform())

        self.output = lasagne.layers.get_output(self.net, deterministic=True)

        self.prediction = theano.function(
	    [self.input_state, self.input_action],
	    self.output,
            allow_input_downcast=True

	)

    def predict(self, state, action):
        return self.prediction(state, action).squeeze()


class LearningCritic(TargetCritic):

    def __init__(self, env, learning_rate=0.001):
        super(LearningCritic, self).__init__(env) 

        self.input_target = T.fvector('input_target')

        params = lasagne.layers.get_all_params(self.net, trainable=True)
        loss = T.sqr(self.output - self.input_target).mean()
        updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

        self.train = theano.function(
	    [self.input_state, self.input_action, self.input_target],
	    loss,	
	    updates=updates,
            allow_input_downcast=True
        )

        self.actor_gradient = theano.function(
            [self.input_state, self.input_action],
            theano.gradient.grad(self.output.sum(), self.input_action),
            allow_input_downcast=True
        )
        self.params = params
  

if __name__ == "__main__":
    np.random.seed(42)
    NUM_EPISODES = 1000
    R = 0. # total reward
    gamma = 0.99

    env = gym.make('BipedalWalker-v2')

    l_critic = LearningCritic(env)
    l_actor = LearningActor(env)
    target_critic = TargetCritic(env)
    target_actor = TargetActor(env)

    replay_mem = ReplayMemmory(size=10000)
    
    try:
        for ep in range(NUM_EPISODES):
            ep_R = 0.
	    
            done = False
            curr_s = floatX(env.reset())
            losses = []
            while not done:
                action = target_actor.get_action(curr_s)
                s, r, done, _ = env.step(action)
                s, r  = floatX(s), floatX(r)
                done = done or env.hull.position.y < 5.0
                # print(action)   
                replay_mem.add(curr_s, action, r, s, done) 
                curr_s = s
               
                if replay_mem.full:
                    states, actions, rewards, next_states, dones = replay_mem.sample()

                    next_q = target_critic.predict(next_states, target_actor.get_actions(next_states))
                    td_targets = rewards + gamma * next_q * dones
                    c_loss = l_critic.train(states, actions, td_targets)
                    losses.append(c_loss)

                    na = l_actor.get_actions(states)
                    actor_grads = l_critic.actor_gradient(states, na)
                    l_actor.train(states, actor_grads)

                    copy_params(l_actor, target_actor)
                    copy_params(l_critic, target_critic)
                    
                    
            ep_R += r
			        
            print("[Episode {}] Got reward of {} critic loss was {}".format(ep, ep_R, np.mean(losses)))
            R += ep_R

    except KeyboardInterrupt:
        print("Got total reward of {} in {} episodes".format(R, ep))
