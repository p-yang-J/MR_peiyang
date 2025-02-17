import numpy as np

class ArmEnv(object):
    dt = 0.02    # refresh rate
    #action_bound = [-1, 1]
    #goal = {'x': 100., 'y': 100., 'l': 40}
    state_dim = 1
    action_dim = 1

    def __init__(self):
        self.scale_info = 19998

    def step(self, action):
        done = False
        #action = np.clip(action, *self.action_bound)
        self.scale_info += action*self.dt
        s = self.scale_info
        # if test_main_ddpg.finsh_task == 1:
        #     #每小步reward（参与者的评价）
        #     r = 。。
        #if test_main_ddpg.S_values
        #if test_main_ddpg.G_values
        #每小步的reward
        #r= 。。。
        print(s)
        #让实验参与者完成一次后说一句finish one step，识别后一步结束
        # if test_main_ddpg.cluster == 1:
        #     done = True
        #     #每大步reward
        #     r = 1
        # return s,r,done


    def reset(self):
        self.scale_info = 19998 + np.random.uniform(-1000,1000)
        return self.scale_info

    def render(self):
        pass

    def sample_action(self):
        return np.random.uniform(-200,200)   # two radians




if __name__ == '__main__':
    env = ArmEnv()
    while True:
        #env.render()
        env.step(env.sample_action())
