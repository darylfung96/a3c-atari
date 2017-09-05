import threading
import tensorflow as tf



import signal
import random
import math
import os
import time

THREAD_SIZE = 1
SAVE_MODEL_RATE = 50    # Save model after this number of step

from env import Env
from rmsprop_applier import RMSApplier
from model import A3CLSTM
from single_thread import SingleThread

ENV_NAME = 'Assault-v0'
create_env = Env('Assault-v0')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
allow_soft_placement=True))

number_actions = create_env.number_actions()
global_network = A3CLSTM(number_actions, -1)
grad_applier = RMSApplier()
#sess.run(tf.global_variables_initializer())


threads = []

# Create worker threads
for i in range(THREAD_SIZE):
    singleThread = SingleThread(sess, i, global_network, 0.01, grad_applier, 100, number_actions, ENV_NAME)
    threads.append(singleThread)

# restore model #
#if os.path.isfile('my_model.ckpt'):
#    tf.train.Saver().restore(sess, 'my_model.ckpt')
#else:
#    print('model not found.')

# initialize global variables #
sess.run(tf.global_variables_initializer())



####### declare summary ###########
score = tf.placeholder(tf.int32)
tf.summary.scalar('Average reward', score)
summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./summary/train', graph=sess.graph)

# the step for the summary

global_step = 0

def train(thread_index):
    global global_step
    while True:
        threads[thread_index].process(sess, summary_op, train_writer, score, global_step)
        global_step += 1


thread_run = []

# create threads to run the worker threads
for i in range(THREAD_SIZE):
    thread_run.append(threading.Thread(target=train, args=(i,)))

for thread in thread_run:
    thread.start()
    time.sleep(1) # sleep for a bit before starting another thread



# wait for thread to finish
for thread in thread_run:
    thread.join()


# todo save model weights

