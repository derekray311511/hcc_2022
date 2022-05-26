#!/usr/bin/env python3

import rospy
import numpy as np
import message_filters
import cv2, os
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
import tf
from tf.transformations import quaternion_matrix
import math

pub = rospy.Publisher('/object_pose', PointStamped, queue_size=10)
rospy.init_node('drone_Object', anonymous=True)
rospy.loginfo("Start D435_Object_Distance")
cv_bridge = CvBridge()

print('Try to get camera info...')

msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=None)
#     [fx'  0  cx' Tx]
# P = [ 0  fy' cy' Ty]
#     [ 0   0   1   0]
print('Get camera info')
fx = msg.P[0]
fy = msg.P[5]
cx = msg.P[2]
cy = msg.P[6]

transform_time = 0.0
transform = Odometry()

tvmonitor = np.zeros(3)
bottle = np.zeros(3)
teddy_bear = np.zeros(3)
keyboard = np.zeros(3)

tvmonitor_vec = []
bottle_vec = []
teddy_bear_vec = []
keyboard_vec = []

count = np.zeros(4, dtype=int)

def main():
    print("Start running main()")
    global count
    depth_image_sub = message_filters.Subscriber('???', ???)
    bb_sub = message_filters.Subscriber('???', ???)
    ts = message_filters.ApproximateTimeSynchronizer([???, ???], ???, ???)
    ts.registerCallback(???)
    rospy.Subscriber("???", ???, ???) 
    rospy.spin()
    

def transform_cb(msg):
    global transform_time
    global transform
    transform_time = msg.header.stamp.to_sec()
    transform = msg
    # print("Get transform time")
    # print(transform_time)

def callback(depth_img, bb):
    local_time = depth_img.header.stamp.to_sec()
    # print("Get local_time")
    # print(local_time)

    # you could set the time error (local_time - transform_time) by yourseelf    
    if abs(local_time - transform_time) < ??? and transform_time != 0:
        print("\nTime error:", local_time - transform_time)
        
        # hint: http://docs.ros.org/en/jade/api/tf/html/python/transformations.html
        # You could use "quaternion_matrix" function to find the 4x4 transform matrix
        global_transform = quaternion_matrix(???)
        global_transform[0][3] = transform.pose.pose.position.x
        global_transform[1][3] = transform.pose.pose.position.y
        global_transform[2][3] = transform.pose.pose.position.z
        # print("transform")
        # print(global_transform)
        try:
            cv_depthimage = cv_bridge.imgmsg_to_cv2(depth_img, "32FC1")
            cv_depthimage2 = np.array(cv_depthimage, dtype=np.float32)
        except CvBridgeError as e:
            print(e)

        for i in bb.bounding_boxes:
            if i.Class == "tvmonitor":
            ############################
            #  Student Implementation  #
            ############################
            




            

def getXYZ(xp, yp, zc, fx,fy,cx,cy):
    #### Definition:
    # cx, cy : image center(pixel)
    # fx, fy : focal length
    # xp, yp : index of the depth image
    # zc: depth
    inv_fx = 1.0/fx
    inv_fy = 1.0/fy
    x = (xp-cx) *  zc * inv_fx
    y = (yp-cy) *  zc * inv_fy
    z = zc
    return (x,y,z,1)


if __name__ == '__main__':
    main()
