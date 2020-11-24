import DQN
import Param
import gym
from gym import logger, wrappers
import torch

if __name__ == '__main__':
    env = Param.get_env()
    logger.set_level(logger.INFO)
    env = wrappers.Monitor(env, directory='/tmp/DQN', force=True)
    dqn = DQN.DQN(Param.MEMORY_SIZE, env)
    avgloss = 0.
    cnt = 0
    for i in range(Param.NUM_EPISODE):
        print('Start Episode', i)
        state = env.reset()
        # state = torch.tensor([state], dtype=torch.float32, device=Param.device)
        done = False
        state = Param.get_screen(env)
        step = 0
        while not done:
            action = dqn.get_action(state)
            _, reward, done, _ = env.step(action.item())
            # next_state, reward, done, _ = env.step(action.item())
            step += 1

            if done:
                next_state = None
                reward = -1
            else:
                next_state = Param.get_screen(env)
                # next_state = torch.tensor([next_state], dtype=torch.float32, device=Param.device)
            
            dqn.push([
                state.clone().detach(),
                action.clone().detach(),
                next_state if next_state is None else next_state.clone().detach(),
                torch.tensor([reward], dtype=torch.float32, device=Param.device)
            ])
            state = next_state

            loss = dqn.learn()
            if loss is not None:
                avgloss += loss.item()
                cnt += 1
            if dqn.step % 50 == 0 and cnt != 0:
                print(dqn.step, avgloss / cnt)
                cnt = 0
                avgloss = 0.
        print('Episode finished after %d steps' % step)
        
    env.close()



