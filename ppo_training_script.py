import os
from time import sleep
import sys

import gymnasium
import numpy as np
import torch
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
        # scenario='scenarios/columbia.yml',
        # scenario='scenarios/barcelona.yml',
        # scenario='scenarios/circle_ccw.yml',
        # scenario='scenarios/berlin.yml',
        # scenario='scenarios/torino.yml',
        # scenario='scenarios/validation.yml',  # change the scenario here (change map), ONLY USE THIS FOR VALIDATION
        # scenario='scenarios/validation2.yml',   # Use this during the midterm competition, ONLY USE THIS FOR VALIDATION
    )

    EPOCHS = 500000
    MAX_STEP = 20000
    best_reward = -np.inf
    best_progress = 0
    agent = get_training_agent(agent_name='PPO')
    # f = open('./log.csv', mode='w+')
    # print('speed, rotation, progress', file=f)

    # ======================================================================
    # Run the environment
    # ======================================================================
    for e in range(EPOCHS):
        obs, info = env.reset(options=dict(mode='grid'))
        t = 0
        total_reward = 0
        old_action = [0, 0]
        old_progress = 0
        done = False
        wall_counter = 0
        back_counter = 0

        rotate_counter = 0
        rotate_progress = 0

        while not done and t < MAX_STEP - 1:
            # ==================================
            # Execute RL model to obtain action
            # ==================================
            action, a_logp, value = agent.get_action(obs)

            next_obs, _, done, truncated, states = env.step({'motor': np.clip(action[0], -1, 1), 'steering': np.clip(action[1], -1, 1)})
            if states['opponent_collisions']:
                print(states['opponent_collisions'])
                exit(0)

            # Calculate reward
            reward = 0
            # reward += np.linalg.norm(states['velocity'][:3])  # speed

            # steering > 0: turn right
            # steering < 0: turn left
            # steering = action[1]
            # rotation = np.linalg.norm(states['velocity'][3:])
            # if rotation > 2.5:
            #     if rotate_counter == 0:
            #         rotate_progress = old_progress
            #     rotate_counter += 1

            #     r_max, l_max = find_rl_max(next_obs['lidar'])
            #     if states['progress'] >= rotate_progress and ((r_max > l_max and steering > 0) or (l_max > r_max and steering < 0)):
            #         print('rotate reward')
            #         reward += 10

            #     # # 急轉
            #     # if rotate_counter > 5:
            #     #     print(f"rotate? {states['progress'] > rotate_progress}")
            #     #     if states['progress'] >= rotate_progress:
            #     #         reward += 10
            #     #     # rotate_counter = 0
            # else:
            #     rotate_counter = 0

            reward += (states['progress'] - old_progress) * 10

            # front_distance = get_lidar_by_angle(next_obs['lidar'], 0)
            # if front_distance > 1 and (states['progress'] - old_progress) > 0:
            #     reward += front_distance / 10

            # lidar_diff = get_lidar_by_angle(next_obs['lidar'], 0) - get_lidar_by_angle(obs['lidar'], 0)
            # if lidar_diff > 0:
            #     reward += lidar_diff * 2

            reward += middle_reward(next_obs['lidar'])

            if states['wall_collision']:
                reward -= 0.01
                wall_counter += 1

                if wall_counter >= 100:
                    # reward = -10
                    # print('wall_collision')
                    wall_counter = 0
                    # done = True
            else:
                if wall_counter > 0:
                    reward += 0.05
                wall_counter = 0

            speed = np.linalg.norm(states['velocity'][:3])
            if reward <= 0:
                if speed < 1:
                    back_counter += 1
                    reward = -0.02

                    if back_counter >= 400:
                        reward = -10
                        print('car stop')
                        back_counter = 0
                        done = True
                else:
                    back_counter = 0

            reward += speed  # speed

            if states['wrong_way'] or abs(states['progress'] - old_progress) > 10:
                if states['wrong_way']:
                    print('wrong_way')
                elif abs(states['progress'] - old_progress) > 10:
                    print('progress jump')
                reward = -10
                done = True

            total_reward += reward

            # if total_reward < -100 or states['time'] >= 120:
            if total_reward < -100:
                if total_reward < -100:
                    print('reward too low')
                # elif states['time'] >= 120:
                #     print('timeout')
                done = True

            agent.store_trajectory(obs, action, value, a_logp, reward)

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
            info = states

            if done:
                agent.store_trajectory(obs, action, value, a_logp, reward)
                break
            old_action = action

            old_progress = states['progress']

        if not done:
            print('exceed STEP')

        env.close()
        agent.learn()

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_model()

        if old_progress > best_progress:
            best_progress = old_progress
            agent.save_model()

        print(
            f"Epoch: {e}, Total reward: {total_reward:.3f}, Final Progress: {old_progress * 100:.2f}%, Best reward: {best_reward:.3f}, Best Progress: {best_progress * 100:.2f}%"
        )
        # print(f"Epoch: {e}, Total reward: {total_reward:.3f}, Best reward: {best_reward:.3f}")


def get_lidar_by_angle(lidar, angle: float):
    """
    angle = 0 -> front \n
    angle < 0 -> left \n
    angle > 0 -> right
    """
    start_angle = -135
    end_angle = 135
    scan = len(lidar)

    angle -= start_angle
    idx = int((angle * scan) / (end_angle - start_angle))
    if idx < 0:
        idx = 0
    elif idx >= scan:
        idx = scan - 1

    return lidar[idx]


def middle_reward(lidar):
    idx = 0
    min_l = lidar[0]
    for i, (l) in enumerate(lidar):
        if l < min_l:
            idx = i
            min_l = l

    start_angle = -135
    end_angle = 135
    scan = len(lidar)
    angle = ((idx * (end_angle - start_angle)) / scan) + start_angle

    # print(f'min_angle: {angle}')
    # if (start_angle <= angle and angle <= end_angle - 180) or (start_angle + 180 <= angle and angle <= end_angle):
    #     print(abs(min_l - get_lidar_by_angle(lidar, angle + 180 if angle < 0 else 180 - angle)))
    if min_l < 0.2 or (angle > end_angle - 180 and start_angle + 180 > angle):
        return 0
    elif abs(min_l - get_lidar_by_angle(lidar, angle + 180 if angle < 0 else 180 - angle)) <= 0.3:
        return 0.1
    else:
        return 0


def find_rl_max(lidar):
    start_angle = -135
    end_angle = 135

    r_max = get_lidar_by_angle(lidar, 1)
    l_max = get_lidar_by_angle(lidar, -1)

    # right
    for i in range(1, end_angle):
        lidar_data = get_lidar_by_angle(lidar, i)
        if lidar_data > r_max:
            r_max = lidar_data

    # left
    for i in range(-1, start_angle, -1):
        lidar_data = get_lidar_by_angle(lidar, i)
        if lidar_data > l_max:
            l_max = lidar_data

    return (r_max, l_max)


if __name__ == '__main__':
    main()
