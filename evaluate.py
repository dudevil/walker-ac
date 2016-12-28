import gym
import argparse
from lasagne.utils import floatX

from agent import TargetActor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Walker evaluation tool.')
    parser.add_argument('--snapshot', '-s', type=str, help='Which snapshot to load')
    parser.add_argument('--num_episodes', '-n', metavar='N', type=int, default=100, nargs='?',
                        help='Number of episodes')
    args = parser.parse_args()

    env = gym.make('BipedalWalker-v2')
    actor = TargetActor(env, snapshot=args.snapshot)
    R = 0.
    try:
        for ep in range(args.num_episodes):
            ep_R = 0.
	    
            done = False
            s = floatX(env.reset())
            while not done:
                env.render()
                action = actor.get_action(s)
                s, r, done, _ = env.step(action)
                s, r  = floatX(s), floatX(r)
                #print(env.hull.position.y < 5.0) 
                # print(action)   
                ep_R += r
			        
            print("[Episode {}] Got reward of {}".format(ep, ep_R))
            R += ep_R

    except KeyboardInterrupt:
        print("Got avg reward of {} in {} episodes".format(R / args.num_episodes, ep))
