import time
import numpy as np
from keras import backend as K
from keras import layers
from keras import optimizers
from keras.models import Model
import scipy

from agent_dir.agent import Agent


class Agent_PG(Agent):
    def __init__(self, env, args):
        super(Agent_PG, self).__init__(env)

        self.name = 'pg_basic'
        self.input_dim = (80, 80)
        self.output_dim = 2
        self.discount_rate = 0.99
        self.env = env
        self.highest_reward = -5
        self.last_obs = None
        self.state = None
        self.load_saved_model = False

        self.__build_network()
        self.__build_train_fn()

        if args.test_pg or self.load_saved_model:
            self.model.load_weights('pg_model')
            print('loading trained model')

    def __build_network(self):
        kernel_initializer = 'lecun_normal'
        K.set_learning_phase(1)
        _input = layers.Input(shape=self.input_dim)
        net = layers.Reshape([*self.input_dim, 1])(_input)
        net = layers.Conv2D(16, (8, 8), strides=(4, 4), activation='relu', kernel_initializer=kernel_initializer)(net)
        net = layers.Conv2D(32, (4, 4), strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer)(net)
        net = layers.Flatten()(net)
        net = layers.Dense(64, activation='relu', kernel_initializer=kernel_initializer)(net)
        net = layers.Dense(self.output_dim, activation='softmax')(net)

        self.model = Model(inputs=_input, outputs=net)
        self.model.summary()

    def __build_train_fn(self):
        action_prob_placeholder = self.model.output
        action_int_placeholder = K.placeholder(shape=(None,), dtype='uint8', name="action_int")
        discount_reward_placeholder = K.placeholder(shape=(None,), name="discount_reward")

        action_onehot = K.one_hot(action_int_placeholder, num_classes=self.output_dim)
        action_prob = K.sum(action_prob_placeholder * action_onehot, axis=-1)
        log_action_prob = K.log(action_prob)

        loss = K.sum(- log_action_prob * discount_reward_placeholder)

        adam = optimizers.Adam(lr=1e-4)
        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_int_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[loss],
                                   updates=updates)

    def init_game_setting(self):
        pass

    def train(self):
        seed = 11037
        total_episodes = 1000000
        self.env.seed(seed)
        for episode in range(total_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            episode_observations = []
            episode_states = []
            episode_rewards = []
            episode_actions = []

            while not done:
                action = self.make_action(state, test=False)

                episode_observations.append(self.last_obs)
                episode_states.append(self.state)
                episode_actions.append(action - 2)

                state, reward, done, info = self.env.step(action)

                episode_rewards.append(reward)

            observations = np.concatenate(episode_observations, axis=0)
            states = np.concatenate(episode_states, axis=0)
            actions_int = np.array(episode_actions, dtype=np.uint8)
            discount_reward = self.__discount_rewards(episode_rewards)
            total_reward = sum(episode_rewards)
            zero_ratio = 100 * (1. - np.mean(actions_int, dtype=np.float))

            if total_reward >= self.highest_reward:
                log_str = '%s S: %d R: %d Z: %d E: %d' % (time.strftime("%Y%m%d-%H%M%S"),
                                                          len(episode_rewards),
                                                          total_reward, zero_ratio, episode)
                self.model.save_weights('model_%s' % log_str)
                self.highest_reward = total_reward
                np.save(log_str + '.npy', observations.astype(np.uint8))

            loss = self.train_fn([states, actions_int, discount_reward])
            print(loss)

            self.model.save_weights('model')

            log_str = '%s S: %d R: %d Z: %d E: %d L: %.4f' % (time.strftime("%Y%m%d-%H%M%S"),
                                                              len(episode_rewards),
                                                              total_reward, zero_ratio, episode, loss[0])

            with open('log', 'a') as f:
                f.write(log_str + '\n')

            print(log_str)
            print(actions_int)

    def make_action(self, observation, test=True):
        self.__preprocess_and_update_state(observation)
        predict = self.model.predict(self.state)
        action_prob = np.squeeze(predict)
        try:
            return np.random.choice([2, 3], p=action_prob)
        except RuntimeWarning:
            print('model had corrupted!', action_prob)
            exit(-1)

    def __preprocess_and_update_state(self, observation):
        # observation = observation[35:195:2, ::2, 0]
        # observation[observation == 144] = 0
        # observation[observation == 109] = 0
        # observation[observation != 0] = 1
        # observation = observation.reshape([1, *self.input_dim]).astype(np.float)

        image_size = [80, 80]
        y = 0.2126 * observation[:, :, 0] + 0.7152 * observation[:, :, 1] + 0.0722 * observation[:, :, 2]
        y = y.astype(np.uint8)
        resized = scipy.misc.imresize(y, image_size)
        resized = resized.astype(np.float32).reshape([1, *self.input_dim])

        if self.last_obs is None:
            self.last_obs = np.zeros_like(resized)

        self.state = resized - self.last_obs
        self.last_obs = resized

    def __discount_rewards(self, rewards, norm=True):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.discount_rate + rewards[t]
            discounted_r[t] = running_add

        if norm:
            discounted_r -= np.mean(discounted_r)
            discounted_r /= np.std(discounted_r)

        return discounted_r

if __name__ == '__main__':
    import os
    os.system('python ../main.py --train_pg')
