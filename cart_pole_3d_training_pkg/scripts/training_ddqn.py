#!/usr/bin/env python

'''
    Training code made by Ricardo Tellez <rtellez@theconstructsim.com>
    Based on many other examples around Internet
    Visit our website at www.theconstruct.ai
'''
import sys
import gym
import numpy
import time
import numpy as np
from gym import wrappers
from std_msgs.msg import Float64
# ROS packages required
import rospy
import rospkg

# import our training environment
import cart_pole_3d_env
from rl_algorithms.ddqn import DQNAgent


if __name__ == '__main__':

    rospy.init_node('cart_pole_3d_gym', anonymous=True, log_level=rospy.WARN)

    reward_publisher = rospy.Publisher('/cart_pole_3d/reward', Float64, queue_size=1)

    # Create the Gym environment
    env = gym.make('CartPole3D-v0')
    rospy.loginfo("Gym environment done")

    # Set the logging system
    # Where we define all of output training to stored.
    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('cart_pole_3d_training_pkg')
    # outdir = pkg_path + '/training_results'
    # env = wrappers.Monitor(env, outdir, force=True)
    # rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0) #define last time step.

    # Loads parameters from the ROS param server
    Alpha = rospy.get_param("/cart_pole_3d/alpha")
    Epsilon = rospy.get_param("/cart_pole_3d/epsilon")
    Gamma = rospy.get_param("/cart_pole_3d/gamma")
    epsilon_discount = rospy.get_param("/cart_pole_3d/epsilon_discount")
    nepisodes = rospy.get_param("/cart_pole_3d/nepisodes")
    nsteps = rospy.get_param("/cart_pole_3d/nsteps")
    running_step = rospy.get_param("/cart_pole_3d/running_step")

    batch_size = 32
    state_size = 3
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    start_time = time.time()
    highest_reward = 0
    scores = []
    episodes = []
    # Starts the main training loop: the one about the episodes to do
    for episode in range(nepisodes):
        rospy.logwarn(">>>>>>>>>> START EPISODE ==>" + str(episode)+ "   <<<<<<<<<<<<<")
        done = False
        score = 0
        step = 0
        # rospy.logwarn("length:  " + str(len(agent.memory)))
        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = np.reshape(observation, [1, state_size])

        episode_time = rospy.get_rostime().to_sec()
        # for each episode, we test the robot for nsteps
        while not done:
            # rospy.logwarn("############### Start Step=>" + str(step))
            step +=1
            # Pick an action based on the current state
            action = agent.act(state)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)
            next_state = np.reshape(observation, [1, state_size])

            # Make the algorithm learn based on the results
            # rospy.logwarn("############### state we were=>" + str(state))
            # rospy.logwarn("############### action that we took=>" + str(action))
            # rospy.logwarn("############### reward that action gave=>" + str(reward))
            # rospy.logwarn("############### State in which we will start next step=>" + str(next_state))
            agent.memorize(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                loss = agent.replay_improved(batch_size)
                # rospy.logwarn("episode: " + str(episode) +" --- loss:" + str(loss))

            if done:
                scores.append(step)
                rospy.logwarn("episode: " + str(episode) +" --- frames:" + str(step))
                break
        # save the model
        if episode % 20 == 0:
            rospy.logwarn("save model...")
            agent.save('cart_pole_3d_ddqn.h5')

    env.close()
