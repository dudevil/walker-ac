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

    def __init__(self, size=100000, batch_size=64):
        self.size = size
        self.queue = deque(maxlen=size)
        self.batch_size = batch_size
        random.seed(42)

    @property
    def full(self):
        return len(self.queue) == self.size

    def add(self, state, action, reward, next_state, done):
        self.queue.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.queue, self.batch_size)
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


def ornstein_uhlenbeck_noise(x, t, theta=.2, sigma=.2):
    W = np.random.normal(scale=sigma/(t+1), size=x.shape)
    mu = np.zeros(x.shape)
    return  sigma * W # theta * (mu - x) / t


class TargetActor:

    def __init__(self, env, snapshot=None):
        self.input_state = T.fmatrix('input_state')
        self.state_in = lasagne.layers.InputLayer((None, 24), self.input_state)

        self.l_hid1 = lasagne.layers.DenseLayer(
            self.state_in, num_units=400,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        self.l_hid2 = lasagne.layers.DenseLayer(
            self.l_hid1, num_units=300,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
	
        self.net = lasagne.layers.DenseLayer(
            self.l_hid2, num_units=4,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=(np.random.sample(size=(300, 4)) * 1e-4 - 5e-5) ,
            b=None
        )

        if snapshot:
            with open(snapshot, 'rb') as f:
                params = pickle.load(f)
            lasagne.layers.set_all_param_values(self.net, params)

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

    def save_params(self, filename="snapshots/actor.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(lasagne.layers.get_all_param_values(self.net), f,  pickle.HIGHEST_PROTOCOL)

class LearningActor(TargetActor):

    def __init__(self, env, learning_rate=0.0001):
        super(LearningActor, self).__init__(env)
        self.grad_from_critic = T.fmatrix('c_grads')

        params = lasagne.layers.get_all_params(self.net, trainable=True)
        grads = theano.gradient.grad(None, params, known_grads={self.output: - self.grad_from_critic})
        updates = lasagne.updates.adam(grads, params, learning_rate=learning_rate)
        
        self.train = theano.function(
            [self.input_state, self.grad_from_critic],
            self.output,
            updates=updates,
            allow_input_downcast=True

        )
        self.params = params
    


class TargetCritic:
    
    def __init__(self, env, learning_rate=0.001, snapshot=None):
        self.input_state = T.fmatrix('input_state')
        self.input_action = T.fmatrix('input_action')

        self.state_in = lasagne.layers.InputLayer((None, 24), self.input_state)
        self.action_in = lasagne.layers.InputLayer((None, 4), self.input_action)

        self.l_hid1 = lasagne.layers.DenseLayer(
            self.state_in, num_units=400,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
            b=None
        )

        self.l_hid2 = lasagne.layers.DenseLayer(
            self.l_hid1, num_units=300,
            nonlinearity=None,
            W=lasagne.init.GlorotUniform())


        self.l_acts = lasagne.layers.DenseLayer(
            self.action_in, num_units=300,
            nonlinearity=None,
            W=lasagne.init.GlorotUniform())
	
        self.l_sum = lasagne.layers.NonlinearityLayer(
            lasagne.layers.ElemwiseSumLayer([self.l_hid2, self.l_acts]))

        self.net = lasagne.layers.DenseLayer(
            self.l_sum, num_units=1,
            nonlinearity=None,
            W=(np.random.sample(size=(300, 1)) * 1e-4 - 5e-5))

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
        l2_penalty = lasagne.regularization.regularize_network_params(self.net, lasagne.regularization.l2) * 1e-2
        loss = T.sqr(self.output - self.input_target).mean() + l2_penalty
        updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

        self.train = theano.function(
	    [self.input_state, self.input_action, self.input_target],
	    loss,	
	    updates=updates,
            allow_input_downcast=True
        )

        self.actor_gradient = theano.function(
            [self.input_state, self.input_action],
            theano.gradient.grad(self.output.mean(), self.input_action),
            allow_input_downcast=True
        )
        self.params = params
  

if __name__ == "__main__":
    np.random.seed(42)
    NUM_EPISODES = 1000000
    R = 0. # total reward
    max_epr = -np.inf
    gamma = 0.99

    env = gym.make('BipedalWalker-v2')

    l_critic = LearningCritic(env)
    l_actor = LearningActor(env)
    target_critic = TargetCritic(env)
    target_actor = TargetActor(env)
    copy_params(l_critic, target_critic, tau=1.)
    copy_params(l_actor, target_actor, tau=1.)

    replay_mem = ReplayMemmory(size=100000, batch_size=64)
    
    try:
        for ep in range(NUM_EPISODES):
            ep_R = 0.
            step = 0
            done = False
            curr_s = floatX(env.reset())
            losses = []
            while not done:
                mu_action = target_actor.get_action(curr_s)
                action = mu_action # + ornstein_uhlenbeck_noise(mu_action, step)
                s, r, done, _ = env.step(action)
                if env.hull.position.y < 5.0 or step > 2500:
                    done = True
                    r = -100.
                s, r  = floatX(s), floatX(r)
                replay_mem.add(curr_s, action, r, s, done) 
                curr_s = s 
                ep_R += r
                step += 1
                if replay_mem.full:
                    states, actions, rewards, next_states, dones = replay_mem.sample()

                    next_q = target_critic.predict(next_states, target_actor.get_actions(next_states))
                    td_targets = rewards + gamma * next_q * dones
                    c_loss = l_critic.train(states, actions, td_targets)
                    losses.append(c_loss)

                    actor_grads = l_critic.actor_gradient(states, l_actor.get_actions(states))
                    l_actor.train(states, actor_grads)

                    copy_params(l_actor, target_actor)
                    copy_params(l_critic, target_critic)

            if ep_R > max_epr or (ep % 100) == 0:
                target_actor.save_params("snapshots/{0}_{1:.3f}.pkl".format(ep, ep_R))
                max_epr = max(ep_R, max_epr)

            if replay_mem.full:
                print("[Episode {}] Reward: {} Critic loss: {} steps: {}".format(ep, ep_R, np.mean(losses), step))
                R += ep_R

    except KeyboardInterrupt:
        print("Got Avg reward of {} in {} episodes".format(R / ep, ep))
