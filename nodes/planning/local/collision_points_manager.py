#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from ros_numpy import msgify
from autoware_mini.msg import Path, DetectedObjectArray,Waypoint
from sensor_msgs.msg import PointCloud2

from shapely import LineString
from shapely import Point

DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('vx', np.float32),
    ('vy', np.float32),
    ('vz', np.float32),
    ('distance_to_stop', np.float32),
    ('deceleration_limit', np.float32),
    ('category', np.int32)
])

class CollisionPointsManager:

    def __init__(self):

        # parameters
        self.safety_box_width = rospy.get_param("safety_box_width")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")

        # variables
        self.detected_objects = None
        self.goal_waypoint = None

        # Lock for thread safety
        self.lock = threading.Lock()

        # publishers
        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('global_path',Path,self.global_path_callback,queue_size=1)

    def global_path_callback(self, msg: Path):
        self.goal_waypoint = msg.waypoints[-1]

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def path_callback(self, msg : Path):
        with self.lock:
            detected_objects = self.detected_objects
        collision_points = np.array([], dtype=DTYPE)

        if(detected_objects is None):
            rospy.logwarn("%s - detected objects is none", rospy.get_name())
            return

        point_cloud2 = PointCloud2()

        if(len(msg.waypoints)> 0):

            self.path_linestring = LineString([(w.position.x, w.position.y) for w in msg.waypoints])
            self.local_path_buffered  = self.path_linestring.buffer(distance=self.safety_box_width/2,cap_style='flat')
            shapely.prepare(self.local_path_buffered)
            
            last_path_waypoint = msg.waypoints[-1]
            last_path_point = Point(last_path_waypoint.position.x,last_path_waypoint.position.y)
            last_path_point_buffered = last_path_point.buffer(0.5)
                
            if (len(detected_objects) > 0): 
                for object in detected_objects:
                    pol = shapely.Polygon(np.array(object.convex_hull).reshape(-1,3)[:, :2])
                    if (shapely.intersects(self.local_path_buffered,pol)):
                        intersection_points = shapely.get_coordinates(shapely.intersection(self.local_path_buffered,pol))
                        for x, y in intersection_points:
                            object_speed = np.linalg.norm([object.velocity.x, object.velocity.y, object.velocity.z])
                            collision_points = np.append(collision_points, np.array([(x, y, object.centroid.z, object.velocity.x, object.velocity.y, object.velocity.z, self.braking_safety_distance_obstacle, np.inf, 3 if object_speed < self.stopped_speed_limit else 4)], dtype=DTYPE))
                        
            
            if(shapely.intersects(last_path_point_buffered,Point(self.goal_waypoint.position.x, self.goal_waypoint.position.y))):
                collision_points = np.append(collision_points, np.array([(self.goal_waypoint.position.x, self.goal_waypoint.position.y, self.goal_waypoint.position.z,
                                                                            0.0, 0.0, 0.0,  # no velocity because its a static point
                                                                            self.braking_safety_distance_goal, np.inf,
                                                                            1)],  # category 1 = goal
                                                                            dtype=DTYPE))
     
        point_cloud2 = msgify(PointCloud2,collision_points)
        point_cloud2.header = msg.header
        self.local_path_collision_pub.publish(point_cloud2)
            

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('collision_points_manager')
    node = CollisionPointsManager()
    node.run()