import gym

from cartpole_wrapper import CartPoleWrapper
from models import DQN
from nec_agent import NECAgent

def main():
  env = CartPoleWrapper(gym.make('CartPole-v1'))
  embedding_model = DQN(5)
  agent = NECAgent(env, embedding_model, test_period=25)
  agent.train()

if __name__ == "__main__":
  main()        
