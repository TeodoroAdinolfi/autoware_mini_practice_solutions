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
        # Publishers
        self.points_clustered_pub = rospy.Publisher('/detection/lidar/points_clustered',PointCloud2,queue_size=1,tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        
    def points_callback(self, msg : PointCloud2):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        print("Points shape", points.shape)
        labels = self.clusterer.fit_predict(points)
        print("Labels shape", labels.shape)
        # creating points_labeled a single tensor with filtered rows
        labels_reshaped = labels.reshape(-1, 1)
        combined_tensor = np.concatenate((points, labels_reshaped), axis=1)
        mask = combined_tensor[:, 3] != -1
        points_labeled = combined_tensor[mask]

        # convert labelled points to PointCloud2 format
        data = unstructured_to_structured(points_labeled, dtype=np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('label', np.int32)
        ]))
        # publish clustered points message
        cluster_msg = msgify(PointCloud2, data)
        cluster_msg.header.frame_id = msg.header.frame_id
        cluster_msg.header.stamp = msg.header.stamp
        self.points_clustered_pub.publish(cluster_msg)




    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_clusterer')
    node = PointsClusterer()
    node.run()