#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

rospy.init_node('publisher')
pub = rospy.Publisher('/message', String, queue_size=10)
message = rospy.get_param('~message', 'Hello World!')
read_rate = rospy.get_param('~rate',1)
print("Read ",read_rate)
rate = rospy.Rate(read_rate)

while not rospy.is_shutdown():
    pub.publish(message)
    rate.sleep()