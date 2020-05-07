import gym
import rospy
import time
import numpy as np
import math
import copy
from gym import utils, spaces
import numpy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState, Imu
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection

from gym.utils import seeding
from gym.envs.registration import register

from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion

reg = register(
    id='CartPole3D-v0',
    entry_point='cart_pole_3d_env:CartPole3DEnv',
    timestep_limit=1000,
)


class CartPole3DEnv(gym.Env): #class inherit from gym.Env.

    def __init__(self): # We initialize and define init function.
        number_actions = rospy.get_param('/cart_pole_3d/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        self.state_size = 3

        self._seed()

        # get configuration parameters
        self.init_cart_vel = rospy.get_param('/cart_pole_3d/init_cart_vel')

        # Actions
        self.cart_speed_fixed_value = rospy.get_param('/cart_pole_3d/cart_speed_fixed_value')

        self.start_point = Point()
        self.start_point.x = rospy.get_param("/cart_pole_3d/init_cube_pose/x")
        self.start_point.y = rospy.get_param("/cart_pole_3d/init_cube_pose/y")
        self.start_point.z = rospy.get_param("/cart_pole_3d/init_cube_pose/z")

        # Done
        self.max_angle = rospy.get_param('/cart_pole_3d/max_angle')
        self.max_distance = rospy.get_param('/cart_pole_3d/max_distance')

        # Rewards
        self.keep_pole_up_reward = rospy.get_param("/cart_pole_3d/keep_pole_up_reward")
        self.end_episode_points = rospy.get_param("/cart_pole_3d/end_episode_points")

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()
        self.controllers_list = ['joint_state_controller',
                                 'cart_joint_velocity_controller'
                                 ]
        self.controllers_object = ControllersConnection(namespace="cart_pole_3d",
                                                        controllers_list=self.controllers_list)
        """
        Namespace of robot is the namespace in front of your controller.
        Controllers_list => that we want to reset at each time that we call the reset controllers.
          This case we just have two: joint_state_controller, inertia_wheel_roll_joint_velocity_controller
        We need to create a function which gets retrieves those controllers.
        """

        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self.check_all_sensors_ready()

        rospy.Subscriber("/cart_pole_3d/joint_states", JointState, self.joints_callback)
        rospy.Subscriber("/cart_pole_3d/odom", Odometry, self.odom_callback)
        # rospy.Subscriber("/cart_pole_3d/imu", Imu, self.imu_callback)

        self._cart_velocity_publisher = rospy.Publisher('/cart_pole_3d/cart_joint_velocity_controller/command',
                                             Float64, queue_size=1)

        self.check_publishers_connection()
        self.gazebo.pauseSim()

    def _seed(self, seed=None):  # overriden function
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):  # overriden function

        self.gazebo.unpauseSim()
        self.set_action(action)
        self.gazebo.pauseSim()
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {}
        reward = self.compute_reward(obs, done)
        simplified_obs = self.convert_obs_to_state(obs)

        return simplified_obs, reward, done, info

    def _reset(self):  # We are using a virtual function defined in the gym infrastructure.

        """
        Everytime we start episode, robot will be the same place and configurations. Because otherwise it won't
        learn correctly and we can't iterated. => basic stuff for Reinforcement learning.
        :return:
        """
        self.gazebo.unpauseSim()
        """
        why we need to unpauseSim because resetting controllers and for checking the sensors, we need the simulation
        to be running because otherwise we don't have any sensory data and we don't have access to the controller reset
        functions services they won't work and tell you to hit play. => it is very important.
        """
        self.controllers_object.reset_controllers()
        self.check_all_sensors_ready()
        self.set_init_pose()
        #initialized robot
        self.gazebo.pauseSim()
        self.gazebo.resetSim()
        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self.check_all_sensors_ready()
        self.gazebo.pauseSim()
        self.init_env_variables()
        obs = self._get_obs()
        simplified_obs = self.convert_obs_to_state(obs)

        return simplified_obs

    def init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.total_distance_moved = 0.0
        self.current_y_distance = self.get_y_dir_distance_from_start_point(self.start_point)
        self.cart_current_speed = rospy.get_param('/cart_pole_3d/init_cart_vel')

    def _is_done(self, observations):

        # MAximum distance to travel permited in meters from origin
        max_upper_distance = self.max_distance
        max_lower_distance = -self.max_distance
        max_angle_allowed = self.max_angle

        if (observations[1] > max_upper_distance or observations[1] < max_lower_distance):
            rospy.logerr("Cart Too Far==>" + str(observations[1]))
            done = True
        elif(observations[2] > max_angle_allowed or observations[2] < -max_angle_allowed):
            rospy.logerr("Pole excceed allowed angle =>" + str(observations[2]))
            done = True
        else:
            # rospy.loginfo("Cart NOT Too Far==>" + str(observations[1]))
            # rospy.loginfo("Pole still in range==>" + str(observations[2]))
            done = False

        return done

    def set_action(self, action):

        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0:  # Move left
            self.cart_current_speed = self.cart_speed_fixed_value
        elif action == 1:  # Move right
            self.cart_current_speed = -1*self.cart_speed_fixed_value

        # We clamp Values to maximum
        rospy.logdebug("cart_current_speed before clamp==" + str(self.cart_current_speed))
        self.cart_current_speed = numpy.clip(self.cart_current_speed,
                                          -1 * self.cart_speed_fixed_value,
                                          self.cart_speed_fixed_value)
        rospy.logdebug("cart_current_speed after clamp==" + str(self.cart_current_speed))

        self.move_joints(self.cart_current_speed)

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        MyCubeSingleDiskEnv API DOCS
        :return:
        """

        # We get the angle of the pole_angle
        pole_angle = round(self.joints.position[1],3)

        # We get the distance from the origin
        y_distance = self.get_y_dir_distance_from_start_point(self.start_point)

        # We get the current speed of the cart
        current_cart_speed_vel = self.get_cart_velocity()

        cube_observations = [
            round(current_cart_speed_vel, 1),
            round(y_distance, 1),
            round(pole_angle, 1)
        ]

        return cube_observations

    def get_cart_velocity(self):
        # We get the current joint roll velocity
        roll_vel = self.joints.velocity[1]
        return roll_vel

    def get_y_linear_speed(self):
        # We get the current joint roll velocity
        y_linear_speed = self.odom.twist.twist.linear.y
        return y_linear_speed

    def get_y_dir_distance_from_start_point(self, start_point):
        """
        Calculates the distance from the given point and the current position
        given by odometry. In this case the increase or decrease in y.
        :param start_point:
        :return:
        """
        y_dist_dir = self.odom.pose.pose.position.y - start_point.y

        return y_dist_dir

    def get_pole_angle(self):
        return self.imu.orientation.y

    def compute_reward(self, observations, done):
        speed = observations[0]
        distance = observations[1]
        angle = observations[2]

        # if not done:
        #     # positive for keeping the pole up
        #     reward_keep_pole_up = self.keep_pole_up_reward
        #     # nigative Reinforcement
        #     reward_distance = distance * -2
        #
        #     # positive reward on angle
        #     reward_angle = 10 / ((abs(angle) * 1)+1.1)
        #
        #     reward = reward_angle + reward_distance + reward_keep_pole_up
        #
        #     rospy.loginfo("Reward_distance=" + str(reward_distance))
        #     rospy.loginfo("Reward_angle= " + str(reward_angle))
        # else:
        #     reward = -1 * self.end_episode_points
        #
        # return reward
        # rospy.loginfo("pole_angle for reward==>" + str(angle))
        delta = 0.7 - abs(angle)
        reward_pole_angle = math.exp(delta*10)

        # If we are moving to the left and the pole is falling left is Bad
        # rospy.logwarn("pole_vel==>" + str(speed))
        pole_vel_sign = numpy.sign(speed)
        pole_angle_sign = numpy.sign(angle)
        # rospy.logwarn("pole_vel sign==>" + str(pole_vel_sign))
        # rospy.logwarn("pole_angle sign==>" + str(pole_angle_sign))

        # We want inverted signs for the speeds. We multiply by -1 to make minus positive.
        # global_sign + = GOOD, global_sign - = BAD
        base_reward = 500
        if speed != 0:
            global_sign = pole_angle_sign * pole_vel_sign * -1
            reward_for_efective_movement = base_reward * global_sign
        else:
            # Is a particular case. If it doesnt move then its good also
            reward_for_efective_movement = base_reward

        reward = reward_pole_angle + reward_for_efective_movement

        # rospy.logwarn("reward==>" + str(reward)+"= r_pole_angle="+str(reward_pole_angle)+",r_movement= "+str(reward_for_efective_movement))
        return reward

    def joints_callback(self, data):
        self.joints = data

    def odom_callback(self, data):
        self.odom = data


    def check_all_sensors_ready(self):
        self.check_joint_states_ready()
        self.check_odom_ready()
        # self.check_imu_ready()
        rospy.logdebug("ALL SENSORS READY")

    def check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message("/cart_pole_3d/joint_states", JointState, timeout=1.0)
                # check response from this topic for 1 second. if don't received respone, it mean not ready.
                # Assure data channels are working perfectly.
                rospy.logdebug("Current cart_pole_3d/joint_states READY=>" + str(self.joints))

            except:
                rospy.logerr("Current cart_pole_3d/joint_states not ready yet, retrying for getting joint_states")
        return self.joints

    def check_odom_ready(self):
        self.odom = None
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/cart_pole_3d/odom", Odometry, timeout=1.0)
                rospy.logdebug("Current /cart_pole_3d/odom READY=>" + str(self.odom))

            except:
                rospy.logerr("Current /cart_pole_3d/odom not ready yet, retrying for getting odom")

        return self.odom

    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while (self._cart_velocity_publisher.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _cart_velocity_publisher yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_base_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")

    def move_joints(self, cart_speed):
        joint_speed_value = Float64()
        joint_speed_value.data = cart_speed
        # rospy.logwarn("cart Velocity >>>>>>>" + str(joint_speed_value))
        self._cart_velocity_publisher.publish(joint_speed_value)

    def set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_joints(self.init_cart_vel)

        return True

    def convert_obs_to_state(self, observations):
        """
        Converts the observations used for reward and so on to the essentials for the robot state
        In this case we only need the orientation of the cube and the speed of the disc.
        The distance doesnt condition at all the actions
        """
        speed = observations[0]
        distance = observations[1]
        angle = observations[2]

        state_converted = [speed, distance, angle]

        return state_converted
