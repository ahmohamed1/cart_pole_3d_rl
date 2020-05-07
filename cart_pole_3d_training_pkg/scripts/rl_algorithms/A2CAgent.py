import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

class A2CAgent:
  def __init__(self, state_size, action_size, actor_lr=1e-3, critic_lr=5e-3, discount_factor=0.99):
    self.state_size = state_size
    self.action_size = action_size
    self.value_size = 1

    self.discount_factor = discount_factor
    self.actor_lr = actor_lr
    self.critic_lr = critic_lr

    self.actor_model = self._build_actor()
    self.critic_model = self._build_critic()


  def _build_actor(self):
    actor = Sequential()
    actor.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
    actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))

    actor.compile(loss="categorical_crossentropy", optimizer=Adam(self.actor_lr))
    return actor

  def _build_critic(self):
    critic = Sequential()
    critic.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
    critic.add(Dense(self.value_size, activation='linear', kernel_initializer='he_uniform'))

    critic.compile(loss='mse', optimizer=Adam(self.critic_lr))
    return critic

  def act(self, state):
    policy = self.actor_model.predict(state, batch_size=1).flatten()
    return np.random.choice(self.action_size, 1, p=policy)[0]

  def train_model(self, state, action, reward, next_state, done):
      target = np.zeros((1, self.value_size))
      advantages = np.zeros((1, self.action_size))

      value = self.critic_model.predict(state)[0]
      next_value = self.critic_model.predict(next_state)[0]

      if done:
          advantages[0][action] = reward - value
          target[0][0] = reward
      else:
          advantages[0][action] = reward + self.discount_factor * (next_value) - value
          target[0][0] = reward + self.discount_factor * next_value

      self.actor_model.fit(state, advantages, epochs=1, verbose=0)
      self.critic_model.fit(state, target, epochs=1, verbose=0)

  def save_model(self, path):
      self.actor_model.save(path + ' actor')
      self.critic_model.save(path + 'critic')

  def load_model(self, path):
      self.actor_model.load_model(path + ' actor')
      self.critic_model.load_model(path + 'critic')
