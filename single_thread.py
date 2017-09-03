import numpy as np
import tensorflow as tf

from model import A3CLSTM


LOCAL_MAX_STEP = 20
gamma = 0.99
tau = 1.


class SingleThread:
    def __init__(self, thread_index, global_network, initial_learning_rate,
                 grad_applier, max_global_time_step, action_size, env, device='/CPU:0'):

        self.thread_index = thread_index
        self.global_network = global_network
        self.initial_learning_rate = initial_learning_rate
        self.grad_applier = grad_applier
        self.max_global_time_step = max_global_time_step
        self.device = device
        self.action_size = action_size
        self.env = env

        # prepare model
        self.local_network = A3CLSTM(action_size, self.thread_index, self.device)
        self.local_network.loss_calculate_scaffold()

        # get gradients for local network
        v_ref = [v for v in self.local_network.get_vars()]
        self.gradients = tf.gradients(self.local_network.total_loss, v_ref,
                                      colocate_gradients_with_ops=False, gate_gradients=False,
                                      aggregation_method=None)
        self.apply_gradients = grad_applier.apply_gradient(self.global_network.get_vars(),
                                                            self.gradients)

        self.sync = self.local_network.sync_from(self.global_network)

        self.episode_reward = 0


    def choose_action(self, policy):
        return np.random.choice(range(len(policy)), p=policy)

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate



    def process(self, sess):
        states = []
        values = []
        rewards = []
        actions = []
        td = []

        deltas = []
        gaes = []

        done = False

        # first we sync local network with global network
        sess.run(self.sync)

        initial_lstm_state = self.local_network.lstm_state_output

        state = self.env.reset()
        # now our local network is the same as global network
        for i in range(0, LOCAL_MAX_STEP):
            self.env.render()
            policy, value = self.local_network.get_policy_value(sess, state)
            action = self.choose_action(policy)

            states.append(state)

            state, reward, done = self.env.step(action)
            rewards.append(reward)
            actions.append(action)
            values.append(value[0])
            actions.append(action)

            self.episode_reward += reward


            if done:
                print('Episode reward: {}'.format(self.episode_reward))

                self.episode_reward = 0
                self.env.reset()
                self.local_network.reset_lstm_state()
                break

        R = 0.0
        gae = 0.0

        if done is False:
            _, value = self.local_network.get_policy_value(sess, state) # run and get the last value
            R = value[0]

        values.append(R)

        for i in reversed(range(len(rewards))):
            R = R * gamma + rewards[i]
            R = R - values[i]
            td.append(R)

            delta = rewards[i] + gamma * values[i+1] - values[i]
            deltas.append(delta)

            gae = gamma * tau * gae + delta
            gaes.append(gae)
        gaes = np.expand_dims(gaes, 1)

        states.reverse()
        states = np.array(states).reshape(-1, 47, 47, 1)
        rewards.reverse()
        values.reverse()


        sess.run([self.apply_gradients],
                     feed_dict={
                         self.local_network.s: states,
                         self.local_network.rewards: rewards,
                         self.local_network.values: values,
                         self.local_network.step_size: [len(actions)],
                         self.local_network.deltas: deltas,
                         self.local_network.gaes: gaes,
                         self.local_network.td: td,
                         self.local_network.LSTMState: initial_lstm_state
        })

        if self.thread_index is 0:
            #print('value: {}'.format(self.local_network.value))
            #print('policy: {}'.format(self.local_network.policy))
            #print('total loss: {}'.format(self.local_network.total_loss))
            print('episode reward: %d' % self.episode_reward)


