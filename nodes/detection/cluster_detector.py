#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import PointCloud2
from ros_numpy import numpify
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from ros_numpy import numpify, msgify
from sklearn.cluster import DBSCAN 
import numpy as np


class PointsClusterer:

    def __init__(self):
        # Parameters
        self.cluster_epsilon  = rospy.get_param("~/detection/lidar/points_clusterer/cluster_epsilon")
        self.cluster_min_size = rospy.get_param("~/detection/lidar/points_clusterer/cluster_min_size")
        # Internal Variables
        self.clusterer = DBSCAN(self.cluster_epsilon,self.cluster_min_size)
        # Subscribers
        rospy.Subscriber('points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        # Publishers

    def points_callback(self, msg : PointCloud2):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        print("Points shape", points.shape)
        labels = self.clusterer.fit_predict(points)
        print("Labels shape", labels.shape)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_clusterer')
    node = PointsClusterer()
    node.run()