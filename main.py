import threading
import tensorflow as tf



import signal
import random
import math
import os
import time


from env import Env
from rmsprop_applier import RMSApplier
from model import A3CLSTM
from single_thread import SingleThread

create_env = Env('Assault-v0')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
allow_soft_placement=True))

number_actions = create_env.number_actions()
global_network = A3CLSTM(number_actions, 0)
grad_applier = RMSApplier()
sess.run(tf.global_variables_initializer())

testing = SingleThread(1, global_network, 0.01, grad_applier, 100, number_actions, create_env)
sess.run(tf.global_variables_initializer())


testing.process(sess)