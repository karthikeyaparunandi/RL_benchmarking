from spinup import td3
import tensorflow as tf
import gym


env_fn = lambda : gym.make('Fish-v2')

#ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='.', exp_name='exp_1')

td3(env_fn=env_fn,  steps_per_epoch=5000, epochs=600, act_noise=0.1, logger_kwargs=logger_kwargs)
