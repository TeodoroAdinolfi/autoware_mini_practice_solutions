#!/usr/bin/env python3
import rospy

from autoware_mini.msg import Path
from geometry_msgs.msg import PoseStamped
from autoware_mini.msg import VehicleCmd
from shapely.geometry import LineString, Point
from shapely import prepare, distance
from tf.transformations import euler_from_quaternion
from math import sin
import numpy as np

class PurePursuitFollower:
    def __init__(self):
        
        # Parameters
        self.path_linestring = None
        self.lookahead_distance = rospy.get_param("~lookahead_distance")
        self.wheel_base = rospy.get_param("/vehicle/wheel_base")

        # Publishers
        self.vehicle_cmd_pub = rospy.Publisher('/control/vehicle_cmd',VehicleCmd,queue_size=10)
        # Subscribers
        rospy.Subscriber('path', Path, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg : Path):
        # convert waypoints to shapely linestring
        self.path_linestring = LineString([(w.position.x, w.position.y) for w in msg.waypoints])
        # prepare path - creates spatial tree, making the spatial queries more efficient
        prepare(self.path_linestring)

    def current_pose_callback(self, msg : PoseStamped):
        # print("X: ",msg.pose.position.x, "Y: ", msg.pose.position.y)
        vehicle_cmd = VehicleCmd()
        vehicle_cmd.header.stamp = msg.header.stamp
        vehicle_cmd.header.frame_id = "base_link"
        
        current_pose = Point([msg.pose.position.x, msg.pose.position.y])
        if (self.path_linestring is not None):
            d_ego_from_path_start = self.path_linestring.project(current_pose)
            print("ego:", d_ego_from_path_start)
            target_lookahead_distance_on_path = d_ego_from_path_start + self.lookahead_distance
            lookahead_point = self.path_linestring.interpolate(target_lookahead_distance_on_path)
            _, _, heading = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            lookahead_heading = np.arctan2(lookahead_point.y - current_pose.y, lookahead_point.x - current_pose.x)
            self.lookahead_distance = distance(current_pose,lookahead_point)
            steering_anle = np.arctan( (2*self.wheel_base*sin(lookahead_heading - heading))/self.lookahead_distance)
            vehicle_cmd.ctrl_cmd.steering_angle = steering_anle
            vehicle_cmd.ctrl_cmd.linear_velocity = 10.0

        self.vehicle_cmd_pub.publish(vehicle_cmd)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()