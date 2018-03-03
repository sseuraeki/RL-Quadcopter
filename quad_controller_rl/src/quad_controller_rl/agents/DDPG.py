import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.ReplayBuffer import ReplayBuffer
from quad_controller_rl.agents.Actor import Actor
from quad_controller_rl.agents.Critic import Critic
from quad_controller_rl.agents.OUNoise import OUNoise
import os
import pandas as pd
from quad_controller_rl import util

class DDPG(BaseAgent):
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):

        self.task = task

        # Load/save parameters
        self.load_weights = True  # try to load weights from previously saved models
        self.save_weights_every = 100  # save weights every n episodes, None to disable
        self.model_dir = util.get_param('out')
        self.model_name = "{}-model".format(task.taskname)
        self.model_ext = ".h5"
        if self.load_weights or self.save_weights_every:
            self.actor_filename = os.path.join(self.model_dir,
                "{}_actor{}".format(self.model_name, self.model_ext))
            self.critic_filename = os.path.join(self.model_dir,
                "{}_critic{}".format(self.model_name, self.model_ext))
            print("Actor filename :", self.actor_filename)  # [debug]
            print("Critic filename:", self.critic_filename)  # [debug]

        if self.task.taskname == 'combined':
            self.actor_filename2 = 'combined-model_actor2.h5'
            self.actor_filename3 = 'combined-model_actor3.h5'
            self.critic_filename2 = 'combined-model_critic2.h5'
            self.critic_filename3 = 'combined-model_critic3.h5'

        # Task (environment) information
        self.state_size = 2 # it seems z position and z velocity are all that needed
        self.action_size = 1 # it seems only z linear force is needed for all the tasks

        # Actor (Policy) Model - 3 separate models to handle combined tasks
        self.action_low = -25. # min value for linear z force
        self.action_high = 25. # max value for linear z force
        self.actor_local1 = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target1 = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_local2 = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target2 = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_local3 = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target3 = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model - 3 separate models to handle combined tasks
        self.critic_local1 = Critic(self.state_size, self.action_size)
        self.critic_target1 = Critic(self.state_size, self.action_size)
        self.critic_local2 = Critic(self.state_size, self.action_size)
        self.critic_target2 = Critic(self.state_size, self.action_size)
        self.critic_local3 = Critic(self.state_size, self.action_size)
        self.critic_target3 = Critic(self.state_size, self.action_size)

        # Load pre-trained model weights, if available
        if self.load_weights and os.path.isfile(self.actor_filename):
            if self.task.taskname == 'combined':
                try:
                    self.actor_local1.model.load_weights(self.actor_filename)
                    self.critic_local1.model.load_weights(self.critic_filename)
                    self.actor_local2.model.load_weights(self.actor_filename)
                    self.critic_local2.model.load_weights(self.critic_filename)
                    self.actor_local3.model.load_weights(self.actor_filename)
                    self.critic_local3.model.load_weights(self.critic_filename)
                    print("Model weights loaded from file!")  # [debug]
                except Exception as e:
                    print("Unable to load model weights from file!")
                    print("{}: {}".format(e.__class__.__name__, str(e)))
            else:
                try:
                    self.actor_local1.model.load_weights(self.actor_filename)
                    self.critic_local1.model.load_weights(self.critic_filename)
                    print("Model weights loaded from file!")  # [debug]
                except Exception as e:
                    print("Unable to load model weights from file!")
                    print("{}: {}".format(e.__class__.__name__, str(e)))

        if self.save_weights_every:
            print("Saving model weights", "every {} episodes".format(
                self.save_weights_every) if self.save_weights_every else "disabled")  # [debug]

        # Initialize target model parameters with local model parameters
        self.critic_target1.model.set_weights(self.critic_local1.model.get_weights())
        self.actor_target1.model.set_weights(self.actor_local1.model.get_weights())
        self.critic_target2.model.set_weights(self.critic_local2.model.get_weights())
        self.actor_target2.model.set_weights(self.actor_local2.model.get_weights())
        self.critic_target3.model.set_weights(self.critic_local3.model.get_weights())
        self.actor_target3.model.set_weights(self.actor_local3.model.get_weights())

        # Noise process
        self.noise = OUNoise(self.action_size)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

        # Episode variables
        self.episode = 0
        self.reset_episode_vars()

        # Save episode stats
        self.stats_filename = os.path.join(
            util.get_param('out'),
            "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))  # [debug]

    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))  # write header first time only

    def reset_episode_vars(self):
        self.episode += 1
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0

    def step(self, state, reward, done, mode=1):

        state = state.reshape(1, -1)  # convert to row vector

        # Choose an action
        action = self.act(state, mode)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            self.total_reward += reward

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences, mode)

        self.last_state = state
        self.last_action = action

        if done:
            # Write episode stats
            self.write_stats([self.episode + 1, self.total_reward])
            print('Total reward: {}'.format(self.total_reward))

            # Save model weights at regular intervals
            if self.save_weights_every and self.episode % self.save_weights_every == 0:
                self.actor_local1.model.save_weights(self.actor_filename)
                self.critic_local1.model.save_weights(self.critic_filename)
                print("Model weights saved at episode", self.episode)  # [debug]
                if self.task.taskname == 'combined':
                    self.actor_local2.model.save_weights(self.actor_filename2)
                    self.critic_local2.model.save_weights(self.critic_filename2)
                    self.actor_local3.model.save_weights(self.actor_filename3)
                    self.critic_local3.model.save_weights(self.critic_filename3)
            self.reset_episode_vars()

        return action

    def act(self, states, mode=1):
        """Returns actions for given state(s) as per current policy."""
        states = np.reshape(states, [-1, self.state_size])

        if mode == 1:
            predict_model = self.actor_local1.model
        elif mode == 2:
            predict_model = self.actor_local2.model
        elif mode == 3:
            predict_model = self.actor_local3.model

        actions = predict_model.predict(states)
        return actions + self.noise.sample()  # add some noise for exploration

    def learn(self, experiences, mode=1):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        if mode == 1:
            actor_target = self.actor_target1
            critic_target = self.critic_target1
            critic_local = self.critic_target1
            actor_local = self.actor_local1
        elif mode == 2:
            actor_target = self.actor_target2
            critic_target = self.critic_target2
            critic_local = self.critic_target2
            actor_local = self.actor_local2
        elif mode == 3:
            actor_target = self.actor_target3
            critic_target = self.critic_target3
            critic_local = self.critic_target3
            actor_local = self.actor_local3

        actions_next = actor_target.model.predict_on_batch(next_states)
        Q_targets_next = critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(critic_local.model, critic_target.model)
        self.soft_update(actor_local.model, actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)