# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import time
import numpy as np
from keras import backend as K
from keras import layers
from keras import optimizers
from keras.models import Model
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU

from agent_dir.agent import Agent


class Agent_DQN(Agent):
    def __init__(self, env, args):

        super(Agent_DQN, self).__init__(env)

        self.name = 'dqn_dueling_512_max_2'
        self.env = env
        self.highest_reward = 5
        self.load_saved_model = False
        self.input_dim = (84, 84, 4)
        self.output_dim = 4

        self.lr = 1e-4
        self.gamma = 0.99

        self.env_step_max = 10000000
        self.update_online_freq = 4
        self.update_target_freq = 1200
        self.save_network_freq = 80
        self.batch_size = 32
        self.highest_episode_reward = 5

        self.epsilon = 0.
        self.epsilon_inc = 0.000002
        self.epsilon_max = 0.95

        self.memory_counter = 0
        self.memory_size = 10000
        self.memory_s = np.zeros((self.memory_size, *self.input_dim))
        self.memory_s_ = np.zeros((self.memory_size, *self.input_dim))
        self.memory_a = np.zeros((self.memory_size,), dtype=np.uint8)
        self.memory_d = np.zeros((self.memory_size,), dtype=np.uint8)
        self.memory_r = np.zeros((self.memory_size,))
        self.memory_l = np.zeros((self.memory_size,))

        self.online_network = None
        self.target_network = None
        self.train_fn = None

        self._build_network()
        self._build_train_fn()

        if args.test_dqn or self.load_saved_model:
            self.online_network.load_weights(self.name + '/model')
            self.target_network.load_weights(self.name + '/model')
            print('loading trained model')

    def init_game_setting(self):
        pass

    def _build_network(self):
        K.set_learning_phase(1)

        _input = layers.Input(shape=self.input_dim)
        net = layers.Conv2D(32, 8, strides=4, padding='valid', activation='relu')(_input)
        net = layers.Conv2D(64, 4, strides=2, padding='valid', activation='relu')(net)
        net = layers.Conv2D(64, 3, strides=1, padding='valid', activation='relu')(net)
        net_a = layers.Lambda(lambda x: x[:, :, :, :32])(net)
        net_v = layers.Lambda(lambda x: x[:, :, :, 32:])(net)
        net_a = layers.Flatten()(net_a)
        net_v = layers.Flatten()(net_v)
        net_a = layers.Dense(256, activation='linear')(net_a)
        net_v = layers.Dense(256, activation='linear')(net_v)
        net_a = LeakyReLU(0.1)(net_a)
        net_v = LeakyReLU(0.1)(net_v)
        net_a = layers.Dense(self.output_dim, activation='linear')(net_a)
        net_v = layers.Dense(1, activation='linear')(net_v)
        net_q = layers.Lambda(lambda x: x[1] + (x[0] - K.mean(x[0], axis=-1, keepdims=True)))([net_a, net_v])

        self.online_network = Model(inputs=_input, outputs=net_q)
        self.online_network.summary()

        _input = layers.Input(shape=self.input_dim)
        net = layers.Conv2D(32, 8, strides=4, padding='valid', activation='relu')(_input)
        net = layers.Conv2D(64, 4, strides=2, padding='valid', activation='relu')(net)
        net = layers.Conv2D(64, 3, strides=1, padding='valid', activation='relu')(net)
        net_a = layers.Lambda(lambda x: x[:, :, :, :32])(net)
        net_v = layers.Lambda(lambda x: x[:, :, :, 32:])(net)
        net_a = layers.Flatten()(net_a)
        net_v = layers.Flatten()(net_v)
        net_a = layers.Dense(256, activation='linear')(net_a)
        net_v = layers.Dense(256, activation='linear')(net_v)
        net_a = LeakyReLU(0.1)(net_a)
        net_v = LeakyReLU(0.1)(net_v)
        net_a = layers.Dense(self.output_dim, activation='linear')(net_a)
        net_v = layers.Dense(1, activation='linear')(net_v)
        net_q = layers.Lambda(lambda x: x[1] + (x[0] - K.mean(x[0], axis=-1, keepdims=True)))([net_a, net_v])

        self.target_network = Model(inputs=_input, outputs=net_q)
        self.target_network.summary()

    def _build_train_fn(self):
        # self.train_fn([s, a, r, d, s_, a_])
        action_q = self.online_network.output
        action_true = K.placeholder(shape=(None,), dtype='uint8', name="action_true")
        reward = K.placeholder(shape=(None,), name="reward")
        done = K.placeholder(shape=(None,), name="done")
        next_action_q = self.target_network.output
        next_action = K.placeholder(shape=(None,), dtype='uint8', name="next_action")

        current_q = K.sum(action_q * K.one_hot(action_true, num_classes=self.output_dim), axis=-1)
        next_q = K.sum(next_action_q * K.one_hot(next_action, num_classes=self.output_dim), axis=-1)
        # next_q = K.max(next_action_q, axis=-1)
        # target_q = K.print_tensor(reward + self.gamma * next_q * K.print_tensor((1 - done), message='inv_done'))
        target_q = reward + self.gamma * next_q * (1 - done)

        losses = K.square(target_q - current_q)
        loss = K.max(losses, axis=-1)

        rmsprop = optimizers.RMSprop(lr=self.lr, rho=0.99)
        updates = rmsprop.get_updates(params=self.online_network.trainable_weights, loss=loss)

        self.train_fn = K.function(inputs=[self.online_network.input,
                                           action_true,
                                           reward,
                                           done,
                                           self.target_network.input,
                                           next_action
                                           ],
                                   outputs=[loss, losses],
                                   updates=updates)

    def make_action(self, observation, test=True):
        observation = observation.reshape([1, *self.input_dim])

        if test:
            actions_q = self.online_network.predict(observation)
            # print(actions_q)
            action = np.argmax(np.squeeze(actions_q))
        else:
            if np.random.uniform() < self.epsilon:
                actions_q = self.online_network.predict(observation)
                action = np.argmax(np.squeeze(actions_q))
            else:
                action = np.random.randint(0, self.output_dim)

            self.epsilon = (self.epsilon + self.epsilon_inc) if self.epsilon < self.epsilon_max else self.epsilon_max

        return action

    def memorize(self, s, a, r, d, s_):
        index = self.memory_counter % self.memory_size
        self.memory_s[index] = s
        self.memory_a[index] = a
        self.memory_r[index] = r
        self.memory_d[index] = d
        self.memory_s_[index] = s_
        self.memory_counter += 1

    def recall(self, index):
        return self.memory_s[index], self.memory_a[index], self.memory_r[index], self.memory_d[index], self.memory_s_[index]

    def train(self):
        # seed = 11037
        done_count = 0
        episode_states = []
        episode_rewards = []
        episode_actions = []
        episode_losses = []

        # self.env.seed(seed)
        state = self.env.reset()
        self.init_game_setting()

        for env_step in range(self.env_step_max):
            # self.env.env.render()

            action = self.make_action(state, test=False)
            next_state, reward, done, info = self.env.step(action)

            self.memorize(state, action, reward, done, next_state)

            if done:
                state = self.env.reset()
                self.init_game_setting()
            else:
                state = next_state

            if done:
                episode_len = len(episode_rewards)
                episode_total_reward = sum(episode_rewards)
                episode_loss = float(np.mean(episode_losses))

                log_str = '%s S: %d R: %d Z: %d E: %d L: %.4f' % (time.strftime("%Y%m%d-%H%M%S"),
                                                                  episode_len,
                                                                  episode_total_reward,
                                                                  100 * sum(np.equal(episode_actions, 0)) / episode_len,
                                                                  done_count,
                                                                  episode_loss)
                done_count += 1

                print(log_str)
                with open(self.name + '/log', 'a') as f:
                    print(log_str, file=f)

                if episode_total_reward >= self.highest_episode_reward:
                    self.online_network.save_weights(self.name + '/model_%s' % log_str)
                    self.highest_episode_reward = episode_total_reward
                    np.save(self.name + '/state_%s.npy' % log_str, episode_states)

                episode_states = []
                episode_rewards = []
                episode_actions = []
                episode_losses = []
            else:
                episode_states.append(state[:, :, -1])
                episode_rewards.append(reward)
                episode_actions.append(action)

            if env_step % self.save_network_freq == 0:
                self.online_network.save_weights(self.name + '/model')

            if env_step % self.update_target_freq == 0:
                self.target_network.load_weights(self.name + '/model')
                print('Update target_network.')
                print('epsilon: %f' % self.epsilon)

            if env_step % self.update_online_freq == 0:
                choice_max = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter
                sample_idx = np.random.choice(choice_max, size=self.batch_size)

                (s, a, r, d, s_) = self.recall(sample_idx)
                a_ = np.argmax(self.online_network.predict(s_), axis=-1)

                [loss, losses] = self.train_fn([s, a, r, d, s_, a_])

                self.memory_l[sample_idx] = np.sum(losses, axis=-1)
                episode_losses.append(loss)

if __name__ == '__main__':
    import os
    os.system('python ../main.py --train_dqn')
