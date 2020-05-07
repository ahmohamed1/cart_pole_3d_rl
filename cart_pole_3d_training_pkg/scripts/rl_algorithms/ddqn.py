from keras.layers import Dense
from keras import Sequential, models
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque
import copy


class DQNAgent():
    def __init__(self, state_size, action_size, learning_rate=1e-3, gamma=0.95,
           epsilon=0.3,memory_length=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory=deque(maxlen=memory_length)
        self.model = self._build_model()
        # create a target model by copying the actual Q network
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def update_target_model(self):
        # copy weight from model to the target model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action =  random.randrange(self.action_size)
        else:
            action = np.argmax(self.model.predict(state))
        return action

    def replay(self, batch_size):
        # select a list of random memorize equal to the batch_size
        minibatch = random.sample(self.memory, batch_size)
        # seperate each variablies to futher process
        for state, action , reward, next_state, done in minibatch:
            updated_q_value = self.model.predict(state)
            if done:
                updated_q_value[0][action]= reward
            if not done:
                #Compute the Q value using the target network
                target_q_value = self.target_model.predict(next_state)[0]
                # applied the Q function
                updated_q_value[0][action] = reward + self.gamma * np.amax(target_q_value)
            # train the model
            self.model.fit(state, updated_q_value, epochs=1, verbose=0)
        # update epsilon
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

    def replay_improved(self, batch_size):
        # select a list of random memorize equal to the batch_size
        minibatch = random.sample(self.memory, batch_size)
        states , targets_q = [], []
        # seperate each variablies to futher process
        for state, action , reward, next_state, done in minibatch:
            updated_q_value = self.model.predict(state)
            if not done:
                #Compute the Q value using the target network
                target_q_value = self.target_model.predict(next_state)[0]
                # applied the Q function
                updated_q_value[0][action] = reward + self.gamma * np.amax(target_q_value)
                states.append(state[0])
                targets_q.append(updated_q_value[0])
        # train the model
        H = self.model.fit(np.array(states), np.array(targets_q), epochs=1, verbose=0)
        # Keeping track of loss
        loss = H.history['loss'][0]
        # update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load_weights(path)
