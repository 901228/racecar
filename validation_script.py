from time import sleep
import sys
import os

import cv2
import gymnasium
import numpy as np
from matplotlib import pyplot as plt

import racecar_gym.envs.gym_api  # Necessary!!!Cannot be deleted!!!
from agent.Agent import get_valid_agent


if 'racecar_gym.envs.gym_api' not in sys.modules:
    raise RuntimeError('Please run: pip install -e . and import racecar_gym.envs.gym_api first!')
def main():
    # ======================================================================
    # Create the environment
    # ======================================================================
    render_mode = 'rgb_array_follow'  # 'human', 'rgb_array_birds_eye' and 'rgb_array_follow'
    env = gymnasium.make(
        'SingleAgentAustria-v0',
        render_mode=render_mode,
        # scenario='scenarios/circle_cw.yml',  # change the scenario here (change map)
        # scenario='scenarios/austria.yml',
        scenario='scenarios/validation.yml',  # change the scenario here (change map), ONLY USE THIS FOR VALIDATION
        # scenario='scenarios/validation2.yml',   # Use this during the midterm competition, ONLY USE THIS FOR VALIDATION
    )
    done = False
    agent = get_valid_agent("PPO")
    # video_path = fr"Z:\Student_Work\_Share\RL競賽影片\{_map}\{student_id}.mp4"
    video_path = fr"{save_root}/{_map}/{student_id}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    output_video_writer = cv2.VideoWriter(filename=video_path,
                                          fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                          fps=20,
                                          frameSize=(640, 480))
    print("save at", video_path)
    # ======================================================================
    # Run the environment
    # ======================================================================
    obs, info = env.reset(options=dict(mode='grid'))
    t = 0
    while not done:
        # ==================================
        # Execute RL model to obtain action
        # ==================================
        action = agent.get_action(obs)
        obs, rewards, done, truncated, states = env.step(action)

        if t % 5 == 0 and "rgb" in render_mode:
            # ==================================
            # Render the environment
            # ==================================

            image = env.render().astype(np.uint8)

            text = f"Student ID: {student_id}"
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

            # 將影像寫入視頻
            output_video_writer.write(image)

            # 顯示影像
            # cv2.imshow(f"{student_id}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(1)
        # if t > 200:
        #     break

        t += 1
        if done or truncated:
            break

        print(f"\r{t} / 12000", end="")

    env.close()
    output_video_writer.release()
    print("finish", video_path)


if __name__ == '__main__':
    save_root = f"videos"
    # _map = "validation1"
    _map = "validation2"
    student_id = "M11252014"
    main()
