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
from scipy.interpolate import interp1d

class PurePursuitFollower:
    def __init__(self):
        
        # Parameters
        self.path_linestring = None
        self.distance_to_velocity_interpolator = None
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
        # Create a distance-to-velocity interpolator for the path
        # collect waypoint x and y coordinates
        waypoints_xy = np.array([(w.position.x, w.position.y) for w in msg.waypoints])
        # Calculate distances between points
        distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xy, axis=0)**2, axis=1)))
        # add 0 distance in the beginning
        distances = np.insert(distances, 0, 0)
        # Extract velocity values at waypoints
        velocities = np.array([w.speed for w in msg.waypoints])
        self.distance_to_velocity_interpolator = interp1d(distances, velocities, kind='linear')
        self.distance_to_velocity_interpolator.bounds_error = True
        self.distance_to_velocity_interpolator.fill_value = 0.0


    def current_pose_callback(self, msg : PoseStamped):
        vehicle_cmd = VehicleCmd()
        vehicle_cmd.header.stamp = msg.header.stamp
        vehicle_cmd.header.frame_id = "base_link"
        
        current_pose = Point([msg.pose.position.x, msg.pose.position.y])
        if (self.path_linestring is not None and self.distance_to_velocity_interpolator is not None):
            d_ego_from_path_start = self.path_linestring.project(current_pose)
            target_lookahead_distance_on_path = d_ego_from_path_start + self.lookahead_distance
            lookahead_point = self.path_linestring.interpolate(target_lookahead_distance_on_path)
            _, _, heading = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            lookahead_heading = np.arctan2(lookahead_point.y - current_pose.y, lookahead_point.x - current_pose.x)
            self.lookahead_distance = distance(current_pose,lookahead_point)
            steering_anle = np.arctan( (2*self.wheel_base*sin(lookahead_heading - heading))/self.lookahead_distance)
            vehicle_cmd.ctrl_cmd.steering_angle = steering_anle
            vehicle_cmd.ctrl_cmd.linear_velocity = self.distance_to_velocity_interpolator(d_ego_from_path_start)
            self.vehicle_cmd_pub.publish(vehicle_cmd)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()