import Net
import Memory
import Param
import random
import torch

class DQN(object):
    def __init__(self, capacity, env):
        self.memory = Memory.Memory(capacity)

        screen = Param.get_screen(env)
        _, _, h, w = screen.shape
        self.num_act = env.action_space.n
        self.policy = Net.ResNet(h, w, self.num_act).to(Param.device)
        self.target = Net.ResNet(h, w, self.num_act).to(Param.device)
        # self.policy = Net.FullyConnected(env.observation_space.shape[0], self.num_act).to(Param.device)
        # self.target = Net.FullyConnected(env.observation_space.shape[0], self.num_act).to(Param.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.train(False)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=Param.LEARNING_RATE, weight_decay=0.01)
        self.step = 0
        self.eps = Param.EPS

    def get_action(self, state):
        r = random.random()
        if(r < self.eps):
            return torch.tensor([random.randrange(self.num_act)], device=Param.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy(state).max(1)[1]
    
    def push(self, val): # val should be [state, action, next_state, reward]
        self.memory.push(val)
    
    def learn(self):
        if(len(self.memory) < Param.BATCH_SIZE):
            return
        batch = self.memory.sample(Param.BATCH_SIZE)
        states = torch.cat([v[0] for v in batch])
        actions = [[v[1]] for v in batch]
        with torch.no_grad():
            y = [[v[3] if v[2] is None else (v[3]+Param.GAMMA * self.target(v[2]).max(1)[0])] for v in batch]

        self.optimizer.zero_grad()
        Q_sa = self.policy(states).gather(1, torch.tensor(actions, device=Param.device))
        loss = torch.nn.functional.l1_loss(Q_sa, torch.tensor(y, device=Param.device))
        loss.backward()
        self.optimizer.step()

        self.step += 1
        if(self.step % Param.UPDATE == 0):
            print('Update Network')
            self.update()
        self.eps *= Param.EPS_DECAY
        if(self.eps < 0.05):
            self.eps = 0.05
        return loss

    def update(self):
        self.target.load_state_dict(self.policy.state_dict())
