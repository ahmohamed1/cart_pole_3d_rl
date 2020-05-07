#!/usr/bin/env python
import time
import rospy
import math
import copy
import numpy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion


class CartPole3DRLUtils(object):

    def __init__(self):

        self.check_all_sensors_ready()

        rospy.Subscriber("/cart_pole_3d/joint_states", JointState, self.joints_callback)
        rospy.Subscriber("/cart_pole_3d/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/cart_pole_3d/imu", Imu, self.imu_callback)

        self._cart_velocity_publisher = rospy.Publisher('/cart_pole_3d/cart_joint_velocity_controller/command',
                                             Float64, queue_size=1)

        self.check_publishers_connection()

    def check_all_sensors_ready(self):
        self.cart_joints_data = None
        while self.cart_joints_data is None and not rospy.is_shutdown():
            try:
                self.cart_joints_data = rospy.wait_for_message("/cart_pole_3d/joint_states", JointState, timeout=1.0)
                rospy.loginfo("Current /cart_pole_3d/joint_states READY=>" + str(self.cart_joints_data))

            except:
                rospy.logerr("Current /cart_pole_3d/joint_states not ready yet, retrying for getting joint_states")

        self.cart_odom_data = None
        while self.cart_odom_data is None and not rospy.is_shutdown():
            try:
                self.cart_odom_data = rospy.wait_for_message("/cart_pole_3d/odom", Odometry, timeout=1.0)
                rospy.loginfo("Current /cart_pole_3d/odom READY=>" + str(self.cart_odom_data))

            except:
                rospy.logerr("Current /cart_pole_3d/odom not ready yet, retrying for getting odom")

        self.cart_imu_data = None
        while self.cart_imu_data is None and not rospy.is_shutdown():
            try:
                self.cart_imu_data = rospy.wait_for_message("/cart_pole_3d/imu", Imu, timeout=1.0)
                rospy.loginfo("Current /cart_pole_3d/imu READY==>" + str(self.cart_imu_data))

            except:
                rospy.logerr("Current /cart_pole_3d/imu not ready yet, retrying for getting imu")

        rospy.loginfo("ALL SENSORS READY")

    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while (self._cart_velocity_publisher.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.loginfo("No susbribers to _cart_velocity_publisher yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.loginfo("_base_pub Publisher Connected")

        rospy.loginfo("All Publishers READY")

    def joints_callback(self, data):
        self.joints = data

    def odom_callback(self, data):
        self.odom = data

    def imu_callback(self, data):
        self.imu = data

    # Reinforcement Learning Utility Code
    def move_joints(self, cart_speed):
        joint_speed_value = Float64()
        joint_speed_value.data = cart_speed
        rospy.loginfo("cart Velocity>>" + str(joint_speed_value))
        self._cart_velocity_publisher.publish(joint_speed_value)

    def get_cart_state(self):
        # We convert from quaternions to euler
        orientation_list = [self.odom.pose.pose.orientation.x,
                            self.odom.pose.pose.orientation.y,
                            self.odom.pose.pose.orientation.z,
                            self.odom.pose.pose.orientation.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        data = self.joints
        pole_angle = round(data.position[0], 3)
        # We get the distance from the origin
        start_position = Point()
        start_position.x = 0.0
        start_position.y = 0.0
        start_position.z = 0.0

        distance = self.get_distance_from_point(start_position,
                                                self.odom.pose.pose.position)

        cart_state = [
            round(self.joints.velocity[0], 1),
            round(distance, 1),
            round(pole_angle,1)
        ]

        return cart_state

    def observation_checks(self, cube_state):

        # MAximum distance to travel permited in meters from origin
        max_upper_distance = 0.5
        max_lower_distance = -0.5
        max_angle_allowed = 0.2

        if (cube_state[1] > max_upper_distance or cube_state[1] < max_lower_distance):
            rospy.logerr("Cube Too Far==>" + str(cube_state[1]))
            done = True
        elif(cube_state[2] > max_angle_allowed or cube_state[2] < -max_angle_allowed):
            rospy.logerr("Pole excceed allowed angle =>" + str(cube_state[2]))
            done = True
        else:
            rospy.loginfo("Cart NOT Too Far==>" + str(cube_state[1]))
            rospy.loginfo("Pole still in range==>" + str(cube_state[2]))
            done = False

        return done

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def get_reward_for_observations(self, state):

        # We reward it for lower speeds and distance traveled

        speed = state[0]
        distance = state[1]
        angle = state[2]

        # nigative Reinforcement
        reward_distance = distance * -2

        # positive reward on angle
        reward_angle = 10 / (abs(angle) * 1)

        reward = reward_angle + reward_distance

        rospy.loginfo("Reward_distance=" + str(reward_distance))
        rospy.loginfo("Reward_angle= " + str(reward_angle))

        return reward


def cube_rl_systems_test():
    rospy.init_node('cart_pole_3d_rl_systems_test_node', anonymous=True, log_level=rospy.INFO)
    cart_rl_utils_object = CartPole3DRLUtils()

    for i in range(10):
        rospy.loginfo("Moving to Speed==>50")
        cart_rl_utils_object.move_joints(cart_speed=50.0)
        time.sleep(0.5)
        rospy.loginfo("Moving to Speed==>-150")
        cart_rl_utils_object.move_joints(cart_speed=-50.0)
        time.sleep(0.5)
        rospy.loginfo("Moving to Speed==>0.0")
        cart_rl_utils_object.move_joints(cart_speed=0.0)
        time.sleep(0.5)

        cart_state = cart_rl_utils_object.get_cart_state()
        done = cart_rl_utils_object.observation_checks(cart_state)
        reward = cart_rl_utils_object.get_reward_for_observations(cart_state)

    rospy.loginfo("Done==>" + str(done))
    rospy.loginfo("Reward==>" + str(reward))


if __name__ == "__main__":
    cube_rl_systems_test()
