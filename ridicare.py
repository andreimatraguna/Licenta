from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np
import time

class HumanoidStandupEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "humanoidstandup.xml", 5)
        utils.EzPickle.__init__(self)
    def get_env(self):
        return self
    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        data = self.sim.data
        uph_cost = (data.qpos[3] - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1 + data.qpos[3]/10 - 1
        done = bool(False)
        if 0.75<=data.qpos[3]/10 and (time.perf_counter_ns()-self.startsims)>1000000000:
            done = bool(True)
        if (time.perf_counter_ns()-self.startsims)>10000000000:
            self.reset()
        elif (time.perf_counter_ns()-self.startsims)>3000000000 and data.qpos[3]<5:
            self.reset()
        self.last=data.qpos[3]/10
        return (
            self._get_obs(),
            reward,
            done,
            dict(
                reward_linup=uph_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_impact=-quad_impact_cost,
                lungime=data.qpos[3]
            ),
        )
    def reset_model(self):
        c = 0.01
        self.startsims=time.perf_counter_ns()
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 0.8925
        self.viewer.cam.elevation = -20
'''import gym
import numpy as np
import time

class HumanoidStandupEnv():

    def __init__(self):
        enviroment_name='HumanoidStandup-v2'
        self.env=gym.make(enviroment_name)
        self.sim=self.env
        self.startsims=time.perf_counter_ns()
        self.drept=None
    def reset(self):
        self.env.reset()
    def render(self):
        self.env.render()
    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        data = self.sim.data
        uph_cost = (data.qpos[3] - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1 + data.qpos[3]/10 - 1
        done = bool(False)
        if 0.9<=data.qpos[3]/10 :
            if self.drept==None:
                self.drept=time.perf_counter_ns()
            else:
                if (time.perf_counter_ns()-self.drept)>300000000:
                    done = bool(True)
        else:
            self.drept=None
        if (time.perf_counter_ns()-self.startsims)>10000000000:
            self.reset()
        elif (time.perf_counter_ns()-self.startsims)>3000000000 and data.qpos[3]<5:
            self.reset()
        self.last=data.qpos[3]/10
        return (
            self._get_obs(),
            reward,
            done,
            dict(
                reward_linup=uph_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_impact=-quad_impact_cost,
                sol_distance=data.qpos[3]
            ),
        )
    def reset_model(self):
        c = 0.01
        self.startsims=time.perf_counter_ns()
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 0.8925
        self.viewer.cam.elevation = -20
'''