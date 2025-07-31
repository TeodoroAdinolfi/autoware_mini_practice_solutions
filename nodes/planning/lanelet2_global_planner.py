#!/usr/bin/env python3
import rospy

import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest

from geometry_msgs.msg import PoseStamped


class Lanelet2GlobalPlanner:

    def __init__(self):
        pass
        # Parameters

        # Publischers

        # Sunbscriers
        rospy.Subscriber('/move_base_simple/goal',PoseStamped,self.goal_pose_callback,queue_size=1)
        
    def run(self):
        rospy.spin()

    
    def goal_pose_callback(self, msg: PoseStamped):
        # loginfo message about receiving the goal point
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                            msg.pose.orientation.w, msg.header.frame_id)


    # from autoware_mini/lanelet2.py
    def load_lanelet2_map(lanelet2_map_path):
        """
        Load a lanelet2 map from a file and return it
        :param lanelet2_map_path: name of the lanelet2 map file
        :param coordinate_transformer: coordinate transformer
        :param use_custom_origin: use custom origin
        :param utm_origin_lat: utm origin latitude
        :param utm_origin_lon: utm origin longitude
        :return: lanelet2 map
        """

        # get parameters
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise ValueError('Unknown coordinate_transformer for loading the Lanelet2 map ("utm" should be used): ' + coordinate_transformer)

        lanelet2_map = load(lanelet2_map_path, projector)
        return lanelet2_map

    
if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()
