#!/usr/bin/python3
import numpy as np
import gym
import theano 
import theano.tensor as T
import lasagne


class Actor:

def __init__(self, env):
		input_state = T.dmatrix('input_state')

		state_in = lasagne.layers.InputLayer((None, 24), input_state)

		l_hid1 = lasagne.layers.DenseLayer(
            state_in, num_units=256,
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
		updates = lasagne.updates.adam(loss, params)

		self.train_fn = theano.function(
			[state_in.input_var, action_in.input_var, input_reward],
			loss,	
			updates=updates)

	def update(self, states, actions, rewards):
		return self.train_fn(
			states[np.newaxis, ...],
		 	actions[np.newaxis, ...],
		 	np.array([rewards]))

	def get_action(self, state):
		pass


class Critic:

	def __init__(self, env):
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
		updates = lasagne.updates.adam(loss, params)

		self.train_fn = theano.function(
			[state_in.input_var, action_in.input_var, input_reward],
			loss,	
			updates=updates)

		self.prediction = theano.function(
			[state_in.input_var, action_in.input_var],
			prediction
			)
			

	def update(self, states, actions, rewards):
		return self.train_fn(
			states[np.newaxis, ...],
		 	actions[np.newaxis, ...],
		 	np.array([rewards]))

	def predict(self, state, action):
		return self.prediction(
			states[np.newaxis, ...],
		 	actions[np.newaxis, ...])



# class EGreedyPolicy:

# 	def __init__(self, env):
# 		self.env = env
		
# 		self.action_fn = theano.function([input_var], prediction)
# 		#self.W = np.random.normal(0., 1e-3, size=(24, 4, 4))
# 		#self.b = np.zeros(4, 4)
# 		self.e = 0.25

# 	def action(self, state):
# 		if np.random.sample() < self.e:
# 			return env.action_space.sample()
# 		else:
# 			self.action_fn(state)


# 	def update(self, prev_action):
# 		pass


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
			while not done:
				# q[prev_s][action] += alpha * (reward + gamma * q[s].max() - q[prev_s][action])
				action = env.action_space.sample()
				s, r, done, _ = env.step(action) # take a random action
				next_action = actor.get_action(s)
				next_q = critic.prediction(s, next_action)

				target = r + gamma * next_q
				loss = critic.update(s, action, target)
				print(loss)
				curr_s = s
				ep_R += r
			
			print("[Episode {}] Got reward of {}".format(ep, ep_R))
			R += ep_R

	except KeyboardInterrupt:
		print("Got total reward of {} in {} episodes".format(R, ep))