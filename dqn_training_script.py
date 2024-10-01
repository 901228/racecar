from time import sleep
import sys

import gymnasium
import numpy as np
from matplotlib import pyplot as plt

import racecar_gym.envs.gym_api  # Necessary!!!Cannot be deleted!!!
from agent.Agent import get_training_agent

if 'racecar_gym.envs.gym_api' not in sys.modules:
    raise RuntimeError('Please run: pip install -e . and import racecar_gym.envs.gym_api first!')


def main():
    # ======================================================================
    # Create the environment
    # ======================================================================
    render_mode = 'human'  # 'human', 'rgb_array_birds_eye' and 'rgb_array_follow'
    env = gymnasium.make(
        'SingleAgentAustria-v0',
        render_mode=render_mode,
        # scenario='scenarios/circle_cw.yml',  # change the scenario here (change map)
        scenario='scenarios/austria.yml',
        # scenario='scenarios/validation.yml',  # change the scenario here (change map), ONLY USE THIS FOR VALIDATION
        # scenario='scenarios/validation2.yml',   # Use this during the midterm competition, ONLY USE THIS FOR VALIDATION
    )

    EPOCHS = 1000
    best_reward = -np.inf
    best_progress = 0
    agent = get_training_agent(agent_name='DQN')

    # ======================================================================
    # Run the environment
    # ======================================================================
    for e in range(EPOCHS):
        obs, info = env.reset(options=dict(mode='grid'))
        t = 0
        total_reward = 0
        old_progress = 0
        done = False

        while not done:
            # ==================================
            # Execute RL model to obtain action
            # ==================================
            action = agent.get_action(obs)

            next_obs, _, done, truncated, states = env.step(action)

            # Calculate reward
            reward = 0
            reward += np.linalg.norm(states['velocity'][:3])  # speed
            reward += states['progress'] - old_progress
            old_progress = states['progress']

            reward += states['checkpoint'] * 2

            # print(f"{next_obs['lidar']}")
            reward += next_obs['lidar'][len(next_obs['lidar']) // 2] / 10

            if states['wrong_way'] or states['wall_collision']:
                reward = -10
                done = True

            total_reward += reward

            if states['time'] >= 120:
                done = True

            agent.store_transition(obs, action, reward, next_obs, done)
            agent.learn()

            if t % 1 == 0 and "rgb" in render_mode:
                # ==================================
                # Render the environment
                # ==================================

                image = env.render()
                plt.clf()
                plt.title("Pose")
                plt.imshow(image)
                plt.pause(0.01)
                plt.ioff()

            t += 1
            obs = next_obs
            if done or truncated:
                break

        env.close()

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_model()

        if old_progress > best_progress:
            best_progress = old_progress
            agent.save_model()

        print(
            f"Epoch: {e}, Total reward: {total_reward:.3f}, Final Progress: {old_progress * 100:.2f}%, Best reward: {best_reward:.3f}, Best Progress: {best_progress * 100:.2f}%"
        )
        # print(f"Epoch: {e}, Best Progress: {best_progress * 100:.2f}%")


if __name__ == '__main__':
    main()
