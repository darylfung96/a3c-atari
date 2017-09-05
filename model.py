import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple
import numpy as np




class A3CLSTM(object):
    def __init__(self, action_size, thread_index, device='/CPU:0', gamma=0.99, tau=1.0):

        self._action_size = action_size
        self._thread_index = thread_index
        self._device = device
        self.gamma = gamma
        self.tau = tau


        # in case if we have GPU, we can change the device
        # we also declare the variable scope for this
        scope_name = 'AC3net_' + str(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            self.s = tf.placeholder("float", shape=[None, 47, 47, 1])

            """ convolution layer """
            # conv1 layer
            self.w_conv1, self.b_conv1 = self._conv_weight_variables([5, 5, 1, 32])
            test = self._conv2d(self.s, self.w_conv1, 2) + self.b_conv1
            conv_layer1 = tf.nn.elu(self._conv2d(self.s, self.w_conv1, 2) + self.b_conv1)

            #conv2 layer
            self.w_conv2, self.b_conv2 = self._conv_weight_variables([5, 5, 32, 32])
            conv_layer2 = tf.nn.elu(self._conv2d(conv_layer1, self.w_conv2, 2) + self.b_conv2)

            self.w_conv3, self.b_conv3 = self._conv_weight_variables([5, 5, 32, 32])
            conv_layer3 = tf.nn.elu(self._conv2d(conv_layer2, self.w_conv3, 2) + self.b_conv3)
            self.conv_layer33 = conv_layer3

            """ fully connected layer """
            flatten_layer1 = tf.reshape(conv_layer3, [-1, 288])

            self.w_fc1, self.b_fc1 = self._fc_weight_variables([288, 256])
            f_connect1 = tf.nn.elu(tf.matmul(flatten_layer1, self.w_fc1) + self.b_fc1)

            f_connect1_reshaped = tf.reshape(f_connect1, [1, -1, 256])
            self.step_size = tf.placeholder("float", [1])

            """ LSTM """
            self.lstm_cell = BasicLSTMCell(256, state_is_tuple=True)
            self.cell_state = tf.placeholder(tf.float32, [1, 256])
            self.hidden_state = tf.placeholder(tf.float32, [1, 256])
            self.LSTMState = LSTMStateTuple(self.cell_state, self.hidden_state)
            lstm_output, self.lstm_current_state = tf.nn.dynamic_rnn(self.lstm_cell, f_connect1_reshaped,
                                                                self.step_size, initial_state=self.LSTMState,
                                                                scope=scope)

            # variables for policy
            self.w_fc2, self.b_fc2 = self._fc_weight_variables([256, self._action_size])
            #variables for value
            self.w_fc3, self.b_fc3 = self._fc_weight_variables([256, 1])

            #reshape output from lstm
            lstm_output = tf.reshape(lstm_output, [-1, 256])

            # get values for policy
            self.policy = tf.nn.softmax(tf.matmul(lstm_output, self.w_fc2) + self.b_fc2)
            self.log_policy = tf.log(tf.clip_by_value(self.policy, 1e-20, 1.0))

            #get values for value
            self.value = tf.matmul(lstm_output, self.w_fc3) + self.b_fc3


            # intialize input for LSTM to 0
            # self.lstm_state_output = LSTMStateTuple(tf.zeros([1, 256]), tf.zeros([1, 256]))
            # We can replace the upper one with this
            self.reset_lstm_state()


            """ loss variables declaration """
            self.value_loss = 0
            self.policy_loss = 0
            self.gae = 0
            self.R = 0

            self.rewards = tf.placeholder("float", [None])
            self.values = tf.placeholder("float", [None])
            self.td = tf.placeholder("float", [None])
            self.deltas = tf.placeholder("float", [None])
            self.gaes = tf.placeholder("float", [None, 1])
            self.a = tf.placeholder("float", [None, self._action_size])
            self.discounted_rewards = tf.placeholder("float", [None, 1])

            self.is_first = True # this will help get value gradient for the first time at the end of the step
            # calculate loss function
            self.loss_calculate_scaffold()

            # when we create BasicLSTMCell, these variables are automatically created for us
            # we want to pass this to w_lstm, and b_lstm
            # reuse the variables automatically created in BasicLSTMCell
            scope.reuse_variables()
            self.w_lstm = tf.get_variable('basic_lstm_cell/kernel')
            self.b_lstm = tf.get_variable('basic_lstm_cell/bias')

            self.reset_loss()

    def reset_lstm_state(self):
        self.lstm_state_output = LSTMStateTuple(np.zeros([1, 256]), np.zeros([1, 256]))


    def reset_loss(self):
        self.policy_loss = 0
        self.value_loss = 0
        self.total_loss = 0
        self.gae = 0
        self.R = 0

    def loss_calculate_scaffold(self):
        with tf.device(self._device):

            #self.R = self.value

            # values pass in here should be reversed #

            #policy loss
            #TODO fix this value loss
            ######3 old value loss ##########
            #self.R = self.R * self.gamma + self.rewards R is calculated for us in thread
            # R = R * gamma + reward ( last R is either 0 or value)
            # advantage = R - value
            #value loss will be 0.5 * advantage.
            #self.value_loss = tf.cumsum(0.5 * tf.square(self.td)) + 0.5 * self.value
            #################################
            value_loss = 0.5 * tf.nn.l2_loss(self.discounted_rewards - self.value)

# we need gaes, deltas, rewards, values, td, states

#TODO write the policy loss function
            # get policy_loss
            #delta = self.rewards[i] + self.gamma * self.values[i+1] - self.values[i]
            #self.gae = self.gae * self.gamma * self.tau + delta
            log_policy = tf.log(self.policy + 1e-6)  # add a constant to prevent NaN
            entropies = -tf.multiply(log_policy, self.policy)
            # calculate policy loss
            td = self.discounted_rewards - self.value

            log_action = tf.multiply(log_policy, self.a) * td
            self.policy_loss = -tf.reduce_sum(log_action - 0.01 * entropies)
            #self.policy_loss = self.policy_loss - self.log_policy * self.gae - 0.01 * entropies
            """ end of for loops """

            self.total_loss = self.policy_loss + 0.5 * self.value_loss


    def get_policy_value(self, sess, current_s):
        policy_output, value_output, self.lstm_state_output = sess.run([self.policy, self.value, self.lstm_current_state], feed_dict={self.s: current_s, self.step_size: [1],
                                                                     self.cell_state: self.lstm_state_output[0],
                                                                     self.hidden_state: self.lstm_state_output[1]})
        return (policy_output[0], value_output[0])


    def sync_from(self, global_network, name=None):
        shared_variables = global_network.get_vars()
        local_variables = self.get_vars()

        sync_operations = []

        with tf.device(self._device):
            with tf.name_scope(name, 'A3CNetwork', []) as name:
                for local_v, shared_v in zip(shared_variables, local_variables):
                    operation = tf.assign(local_v, shared_v)
                    sync_operations.append(operation)
                return tf.group(*sync_operations, name=name)

    # we want to get_var because we will need to apply gradient to these
    def get_vars(self):
        return [self.w_conv1, self.b_conv1,
                self.w_conv2, self.b_conv2,
                self.w_conv3, self.b_conv3,
                self.w_lstm, self.b_lstm,
                self.w_fc1, self.b_fc1,
                self.w_fc2, self.b_fc2,
                self.w_fc3, self.b_fc3]





    # create variables
    def _fc_weight_variables(self, weight_shape):
        fan_in = weight_shape[0]
        fan_out = weight_shape[1]

        value_bound = np.sqrt(6.0 / (fan_in + fan_out))

        bias_shape = [weight_shape[1]]

        weightObtained = tf.Variable(tf.random_uniform(weight_shape, minval=-value_bound, maxval=value_bound))
        biasObtained = tf.Variable(tf.zeros(bias_shape))

        return weightObtained, biasObtained

    # width, height, input, output
    def _conv_weight_variables(self, weight_shape):
        width = weight_shape[0]
        height = weight_shape[1]

        input_channel = weight_shape[2]
        output_channel = weight_shape[3]

        fan_in = width * height * input_channel
        fan_out = width * height * output_channel

        value_bound = np.sqrt(6.0 / (fan_in + fan_out))

        # we want to increase the bias dimension by 1 : 1x?
        bias_shape = [output_channel]

        weightObtained = tf.Variable(tf.random_uniform(weight_shape, minval=-value_bound, maxval=value_bound))
        biasObtained = tf.Variable(tf.zeros(bias_shape))

        return weightObtained, biasObtained

    def _conv2d(self, inputs, weights, strides):
        return tf.nn.conv2d(inputs, weights, strides=[1, strides, strides, 1], padding='VALID')




