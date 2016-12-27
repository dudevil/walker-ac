#!/usr/bin/python3
import numpy as np
import gym
import theano 
import theano.tensor as T
import lasagne
from collections import deque


class Actor:

    def __init__(self, env):
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

        updates = lasagne.updates.adam(grads, params)
        self.train_fn = theano.function(
            [input_state, grad_from_critic],
            prediction,
            updates=updates
        )

    def update(self, states, actions, rewards):
        return self.train_fn(
	    states[np.newaxis, ...],
	    actions[np.newaxis, ...],
	    np.array([rewards]))

    def get_action(self, state):
        return self.predict_fn(state[np.newaxis, ...])

    def train(self, state, gradient):
        return self.train_fn(state, gradient)
       

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
            theano.gradient.jacobian(prediction.flatten(), input_action)
        )
	

    def update(self, states, actions, td_target):
        return self.train_fn(
	    states[np.newaxis, ...],
	    actions[np.newaxis, ...],
	    td_target[:, 0])

    def predict(self, state, action):
        return self.prediction(
	    state[np.newaxis, ...],
	    action)

    def actor_gradient(self, state, action):
        return self.actor_grad(state[np.newaxis, ...], action)


if __name__ == "__main__":
    np.random.seed(42)
    REPLAY_MEM_SIZE = 1000
    REPLAY_MEM = []
    NUM_EPISODES = 1000
    R = 0. # total reward
    gamma = 0.99

    env = gym.make('BipedalWalker-v2')
    # policy = EGreedyPolicy(env)
    critic = Critic(env)
    actor = Actor(env)
    try:
        for ep in range(NUM_EPISODES):
            ep_R = 0.
	    
            done = False
            curr_s = env.reset()
            losses = []
            while not done:
                next_action = actor.get_action(curr_s)
                actor_grads = critic.actor_gradient(curr_s, next_action)
                actor.train(curr_s[np.newaxis, ...], actor_grads[0])

                s, r, done, _ = env.step(next_action[0]) # take a random action

                next_q = critic.predict(s, next_action)

                if env.hull.position.y < 5.0:
                    done = True

                td_target = r + gamma * next_q if not done else np.array([r])[np.newaxis, ...]
                losses.append(critic.update(s, next_action[0], td_target))

                curr_s = s
                ep_R += r
			        
            print("[Episode {}] Got reward of {} critic loss was {}".format(ep, ep_R, np.mean(losses)))
            R += ep_R

    except KeyboardInterrupt:
        print("Got total reward of {} in {} episodes".format(R, ep))
