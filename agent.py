#!/usr/bin/python3
import numpy as np
import gym


class EGreedyPolicy:

	def __init__(self, env):
		self.env = env
		self.W = np.random.normal(0., 1e-3, size=(24, 4))
		self.b = np.zeros(4)
		self.e = 0.25

	def action(self, state):
		if np.random.sample() < self.e:
			return env.action_space.sample()
		else:
			return np.dot(state, self.W) + self.b

	#def update(self, )


if __name__ == "__main__":
	np.random.seed(42)
	REPLAY_MEM_SIZE = 1000
	REPLAY_MEM = []
	NUM_EPISODES = 1000
	R = 0. # total reward
	

	env = gym.make('BipedalWalker-v2')
	policy = EGreedyPolicy(env)
	try:
	
		for ep in range(NUM_EPISODES):
			ep_R = 0.
			
			done = False
			curr_s = env.reset()
			while not done:
				env.render()
				# q[prev_s][action] += alpha * (reward + gamma * q[s].max() - q[prev_s][action])
				s, r, done, _ = env.step(policy.action(curr_s)) # take a random action
				curr_s = s
				ep_R += r
			
			print("[Episode {}] Got reward of {}".format(ep, ep_R))
			R += ep_R

	except KeyboardInterrupt:
		print("Got total reward of {} in {} episodes".format(R, ep))