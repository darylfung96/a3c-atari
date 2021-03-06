import numpy as np
import tensorflow as tf
from env import Env
import time

from model import A3CLSTM


LOCAL_MAX_STEP = 20
gamma = 0.99
tau = 1.


class SingleThread:
    def __init__(self, sess, thread_index, global_network, initial_learning_rate,
                 grad_applier, max_global_time_step, action_size, env_name, device='/CPU:0'):

        self.thread_index = thread_index
        self.global_network = global_network
        self.initial_learning_rate = initial_learning_rate
        self.grad_applier = grad_applier
        self.max_global_time_step = max_global_time_step
        self.device = device
        self.action_size = action_size
        self.env = Env(env_name)

        # prepare model
        self.local_network = A3CLSTM(action_size, self.thread_index, self.device)
        self.local_network.loss_calculate_scaffold()

        # get gradients for local network
        v_ref = [v for v in self.local_network.get_vars()]
        self.gradients = tf.gradients(self.local_network.total_loss, v_ref,
                                      colocate_gradients_with_ops=False, gate_gradients=False,
                                      aggregation_method=None)
       # self.apply_gradients = grad_applier.apply_gradient(self.global_network.get_vars(),
       #                                                     self.gradients)

        self.apply_gradients = tf.train.RMSPropOptimizer(initial_learning_rate).apply_gradients(zip(self.gradients, self.global_network.get_vars()))

        self.sync = self.local_network.sync_from(self.global_network)

        # intiialize states
        self.episode_reward = 0
        self.done = False
        self.state = self.env.reset()




    def choose_action(self, policy):
        return np.random.choice(range(len(policy)), p=policy)

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate


    def write_summary(self, summary, train_writer, global_step):
        if self.thread_index == 0 and global_step % 10 == 0:
            train_writer.add_summary(summary, global_step)

    def process(self, sess, summary_op, train_writer, score, global_step):
        states = []
        values = []
        rewards = []
        discounted_rewards = []
        actions = []

        deltas = []
        gaes = []


        # first we sync local network with global network
        sess.run(self.sync)

        initial_lstm_state = self.local_network.lstm_state_output

        if self.done:
            self.state = self.env.reset()
            self.done = False

        # now our local network is the same as global network
        for i in range(0, LOCAL_MAX_STEP):
            #self.env.render()
            policy, value = self.local_network.get_policy_value(sess, self.state)
            action = self.choose_action(policy)

            states.append(self.state)
            actions.append(action)

            self.state, reward, self.done = self.env.step(action)
            rewards.append(reward)

            values.append(value[0])

            self.episode_reward += reward


            if self.done:
                print('Episode reward: {}'.format(self.episode_reward))

                self.episode_reward = 0
                self.state = self.env.reset()
                self.local_network.reset_lstm_state()

                break

        R = 0.0
        gae = 0.0

        if self.done is False:
            _, R = self.local_network.get_policy_value(sess, self.state) # run and get the last value
            R = R[0]
            #states.append(self.state)


        a = []
        action_batch = []
        for i in reversed(range(len(rewards))):
            R = R * gamma + rewards[i]
            #R = R - values[i] # this is temporal difference
            discounted_rewards.append(R)
            a = np.zeros(self.action_size)
            a[actions[i]] = 1

            action_batch.append(a)

            #delta = rewards[i] + gamma * values[i+1] - values[i]
            #deltas.append(delta)

            #gae = gamma * tau * gae + delta
            #gaes.append(gae)
        #gaes = np.expand_dims(gaes, 1)

        states.reverse()
        states = np.array(states).reshape(-1, 47, 47, 1)
        discounted_rewards = np.array(discounted_rewards).reshape(-1, 1)
        #rewards.reverse()


        _, summary = sess.run([self.apply_gradients, summary_op],
                     feed_dict={
                         self.local_network.s: states,
                         #self.local_network.rewards: rewards,
                         #self.local_network.values: values,
                         self.local_network.step_size: [len(states)],
                         #self.local_network.deltas: deltas,
                        # self.local_network.gaes: gaes,
                         #self.local_network.td: td,
                         self.local_network.a: action_batch,
                         self.local_network.discounted_rewards: discounted_rewards,
                         self.local_network.LSTMState: initial_lstm_state,
                         score: self.episode_reward
        })

        self.write_summary(summary, train_writer, global_step)


        time.sleep(2)
