from typing import Dict

import gym

from racecar_gym.envs.scenarios import MultiAgentScenario, SingleAgentScenario


class SingleAgentRaceCarEnv(gym.Env):

    def __init__(self, scenario: SingleAgentScenario):
        self._scenario = scenario
        self._initialized = False
        self._time = 0.0
        self.observation_space = scenario.agent.observation_space
        self.action_space = scenario.agent.action_space

    def step(self, action: Dict):
        assert self._initialized, 'Reset before calling step'
        state = self._scenario.world.state()
        observation, info = self._scenario.agent.step(action=action)
        done = self._scenario.agent.done(state)
        reward = self._scenario.agent.reward(state, action)
        self._time = self._scenario.world.update()
        return observation, reward, done, state

    def reset(self):
        if not self._initialized:
            self._scenario.world.init()
            self._initialized = True
        else:
            self._scenario.world.reset()
        obs = self._scenario.agent.reset(self._scenario.world.initial_pose(0))
        obs['time'] = 0
        return obs
