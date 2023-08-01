import gym

env = gym.make("CartPole-v0")
observation = env.reset()

for _ in range(100000):
    env.render()
    position, velocity, angle, angle_velocity = observation

    if angle > 0:
        action = 1
    else:
        action = 0

    observation, reward , done, info = env.step(action)
