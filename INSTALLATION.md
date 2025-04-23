** Step 1 **
1. Clone this repository
2. conda create --name safety python=3.6
3. pip install stable-baselines3[extra]
4. pip install absl-py

** Step 2 ** 
1. Install Mujoco and mujoco-py. Use instructions here: https://github.com/openai/mujoco-py
2. Clone safety-gym: git clone https://github.com/openai/safety-gym.git
3. cd safety-gym
4. pip install -e .