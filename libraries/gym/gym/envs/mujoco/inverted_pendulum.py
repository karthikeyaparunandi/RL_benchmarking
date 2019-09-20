import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, '../../../../../../models/pendulum.xml', 1)

    def step(self, a):
        
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() #and (np.abs(ob[1]) <= .2)
        done = not notdone
        reward = (-(2*(abs(ob[0])-np.pi)**2 + ob[1]**2) - 0.1*a[0]**2)/10.0

        return ob, reward, done, {}

    def reset_model(self):

        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.angle_normalize(self.sim.data.qpos), self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def angle_normalize(self, x):
        return -((-x+np.pi) % (2*np.pi)) + np.pi