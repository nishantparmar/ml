import gym
import numpy as np
import tensorflow as tf
from CartPoleNN import CartPoleNN

# Create the CartPole-v0 environment
env = gym.make('CartPole-v0')

# Sets the seed for this env's random number generator(s).
seed = 42
env.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Create the neural network model
num_actions = env.action_space.n
model = CartPoleNN(num_actions)

# Set up the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Training parameters
num_episodes = 100
gamma = 0.99

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    env.render()
    total_reward = 0
    while True:
        # Choose action using the current model
        state_tensor = tf.convert_to_tensor(state.reshape((1, -1)), dtype=tf.float32)
        action_probs = model(state_tensor)
        action = tf.random.categorical(action_probs, 1)[0, 0]

        # Take the action and observe the next state and reward
        next_state, reward, done, _ = env.step(action.numpy())

        # Update total reward
        total_reward += reward

        # Calculate the target Q-value using the Bellman equation
        next_state_tensor = tf.convert_to_tensor(next_state.reshape((1, -1)), dtype=tf.float32)
        next_action_probs = model(next_state_tensor)
        target = reward + gamma * tf.reduce_max(next_action_probs)

        # with tf.GradientTape() as tape:
        #     # Compute the Q-value of the chosen action for the current state
        #     action_one_hot = tf.one_hot(action, num_actions)
        #     predicted_q = tf.reduce_sum(model(state_tensor) * action_one_hot, axis=1)
        #
        #     # Calculate the loss and update the model
        #     loss = loss_fn(target, predicted_q)
        #
        # gradients = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        state = next_state

        if done:
            break

    # Print the total reward for each episode
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Close the environment
env.close()
