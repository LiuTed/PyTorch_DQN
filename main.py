import DQN
import Param
import gym
from gym import logger
import torch

if __name__ == '__main__':
    env = Param.get_env()
    logger.set_level(logger.INFO)
    env = gym.wrappers.Monitor(env, directory='/tmp/DQN', force=True)
    dqn = DQN.DQN(Param.MEMORY_SIZE, env)
    avgloss = 0.
    cnt = 0
    for i in range(Param.NUM_EPISODE):
        env.reset()
        done = False
        state = Param.get_screen(env)
        while not done:
            action = dqn.get_action(state)
            _, reward, done, _ = env.step(action.item())

            if done:
                next_state = None
            else:
                next_state = Param.get_screen(env)
            
            dqn.push([state, action, next_state, torch.tensor([reward]).to(Param.device)])
            state = next_state

            loss = dqn.learn()
            if loss is not None:
                avgloss += loss.item()
                cnt += 1
            if dqn.step % 100 == 0 and cnt != 0:
                print(dqn.step, avgloss / cnt)
        
    env.close()



