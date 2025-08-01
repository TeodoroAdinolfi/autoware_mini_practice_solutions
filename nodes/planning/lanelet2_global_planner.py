#!/usr/bin/env python3
import rospy

import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest
from lanelet2.geometry import distance

from shapely.geometry import LineString, Point

from geometry_msgs.msg import PoseStamped
from autoware_mini.msg import Waypoint
from autoware_mini.msg import Path

from math import sqrt


class Lanelet2GlobalPlanner:

    def __init__(self):
        # Parameters
        self.speed_limit = rospy.get_param('~speed_limit')
        lanelet2_map_path = rospy.get_param('~lanelet2_map_path')
        self.output_frame = rospy.get_param('~/planning/lanelet2_global_planner/output_frame')
        self.distance_to_goal_limit = rospy.get_param('~/planning/lanelet2_global_planner/distance_to_goal_limit')
        self.coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        self.use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        self.utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        self.utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")


        self.lanelet2_map = self._load_lanelet2_map(lanelet2_map_path)
        # traffic rules
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                lanelet2.traffic_rules.Participants.VehicleTaxi)
        # routing graph
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)
        
        self.goal_point = None
        self.current_location = None
        self._goal_reached = False
        # Publischers
        self.waypoints_pub = rospy.Publisher('global_path',Path, queue_size=10)

        # Sunbscriers
        rospy.Subscriber('/move_base_simple/goal',PoseStamped,self.goal_pose_callback,queue_size=1)
        rospy.Subscriber('/localization/current_pose',PoseStamped, self.current_pose_callback, queue_size=1)

    def run(self):
        rospy.spin()


    def publish_waypoints(self, waypoints):
        path = Path()        
        path.header.frame_id = self.output_frame
        path.header.stamp = rospy.Time.now()
        path.waypoints = waypoints
        self.waypoints_pub.publish(path)

    def from_lanelet_to_sequence_of_waypoints(self, path):
        waypoints = []
        for lanelet in path:
            if 'speed_ref' in lanelet.attributes:
                speed = min(float(lanelet.attributes['speed_ref']), self.speed_limit)
            else:
                speed = self.speed_limit

            for point in lanelet.centerline:
                waypoint = Waypoint()
                waypoint.position.x = point.x
                waypoint.position.y = point.y
                waypoint.position.z = point.z
                waypoint.speed = speed/3.6
                
                if  ((not waypoints ) or (waypoint != waypoints[-1]) ):
                    waypoints.append(waypoint)
        # Checking what point is theclosest one to the end
        if (self.goal_point and len(waypoints) >=2):
            path = LineString([(w.position.x,w.position.y) for w in waypoints ])

            goal_proj = path.project(Point(self.goal_point.x, self.goal_point.y))
            nearest_point_to_goal_on_lanelet = path.interpolate(goal_proj)

            last_waypoint = Waypoint()
            last_waypoint.position.x = nearest_point_to_goal_on_lanelet.x
            last_waypoint.position.y = nearest_point_to_goal_on_lanelet.y
            last_waypoint.position.z = waypoints[-1].position.z
            last_waypoint.speed = waypoints[-1].speed
            waypoints[-1] = last_waypoint
        #
        self.publish_waypoints(waypoints)

        

    
    def goal_pose_callback(self, msg: PoseStamped):
        # loginfo message about receiving the goal point
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                            msg.pose.orientation.w, msg.header.frame_id)
        self.goal_point =  BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        self._goal_reached = False
        if self.current_location is not None:
            # get start and end lanelets
            start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
            goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]
            # find routing graph
            route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)
            if route is None:
                rospy.logwarn("%s - No route found for reaching the goal position",rospy.get_name())
                return None
            # find shortest path
            path = route.shortestPath()
            # This returns LaneletSequence to a point where a lane change would be necessary to continue
            path_no_lane_change = path.getRemainingLane(start_lanelet)
            self.from_lanelet_to_sequence_of_waypoints(path_no_lane_change)


    def current_pose_callback(self, msg: PoseStamped):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        if self.goal_point is not None and not self._goal_reached and (sqrt((self.goal_point.x - self.current_location.x)**2+(self.goal_point.y - self.current_location.y)**2) < self.distance_to_goal_limit):
            self._goal_reached = True
            self.publish_waypoints([])
            rospy.loginfo("%s - Goal reached",rospy.get_name())


    # from autoware_mini/lanelet2.py
    def _load_lanelet2_map(self, lanelet2_map_path):
        """
        :param lanelet2_map_path: name of the lanelet2 map file
        :return: lanelet2 map
        """
        # Load the map using Lanelet2
        if self.coordinate_transformer == "utm":
            projector = UtmProjector(Origin(self.utm_origin_lat, self.utm_origin_lon), self.use_custom_origin, False)
        else:
            raise ValueError('Unknown coordinate_transformer for loading the Lanelet2 map ("utm" should be used): ' + coordinate_transformer)

        lanelet2_map = load(lanelet2_map_path, projector)
        return lanelet2_map

    
if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()
