from spinup import td3
import tensorflow as tf
import gym


env_fn = lambda : gym.make('InvertedPendulum-v2')

ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='.', exp_name='exp_1')

td3(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)