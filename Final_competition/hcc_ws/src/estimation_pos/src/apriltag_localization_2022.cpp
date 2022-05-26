#include <stdlib.h>
#include <iostream>
#include <vector>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf/LinearMath/Vector3.h>
#include <geometry_msgs/Transform.h>
#include <nav_msgs/Odometry.h>

using namespace std;
using namespace ros;


tf::StampedTransform echo_transform;
tf::TransformListener *tf_listener;
tf::StampedTransform min_distance_trans;
tf::StampedTransform camera_transform;
tf::StampedTransform tag_transforms[6];
int id;
double min_distance = 100;
double check_time[6] = {0};
double local_time = 0;
Publisher transform_pub;


void listener(){
  // use tf_listener to get the transformation from camera_link to tag 0
  id = -1;
  min_distance = 100;
  for (int i = 0; i < 5; i++){
    string parent_id = ???;
    string child_id = ???;

    tf_listener->waitForTransform(child_id, parent_id, ros::Time::now(), ros::Duration(0.07));
    try {
 
      tf_listener->lookupTransform(parent_id, child_id, ros::Time(0), echo_transform);
      if (check_time[i] != echo_transform.stamp_.toSec()){
        std::cout << "At time " << std::setprecision(16) << echo_transform.stamp_.toSec() << std::endl;
        cout << "Frame id:" << echo_transform.frame_id_ << ", Child id:" << echo_transform.child_frame_id_ << endl;
        double yaw, pitch, roll;
        echo_transform.getBasis().getRPY(roll, pitch, yaw);
        tf::Quaternion q = echo_transform.getRotation();
        tf::Vector3 v = echo_transform.getOrigin();
        std::cout << "- Translation: [" << v.getX() << ", " << v.getY() << ", " << v.getZ() << "]" << std::endl;
        std::cout << "- Rotation: in Quaternion [" << q.getX() << ", " << q.getY() << ", "
                  << q.getZ() << ", " << q.getW() << "]" << std::endl;
        std::cout << "-  echo time:" << echo_transform.stamp_.toSec() << endl;
        std::cout << "- local time:" << local_time << endl;
            
        // hint:
        // double dist;
        // dist = ???;
        /************************************************************** 
        //                                                           //
        //                                                           //
        //                 Student Implementation                    //
        //                                                           //
        //                                                           //
        **************************************************************/

        local_time = echo_transform.stamp_.toSec(); // record the time you catch the transform
        check_time[i] = local_time;
        /*
          find the closet distance from the tag to camera_link (remember to modify the parent_id).  //
        */
      }
      
      
      // find the closet tag to localization
      /************************************************************** 
      //                                                           //
      //                                                           //
      //                 Student Implementation                    //
      //                                                           //
      //                                                           //
      **************************************************************/

    }
    catch (tf::TransformException& ex)
    {
      std::cout << "Exception thrown:" << ex.what() << std::endl;
    }
  }

  /* localize the robot position */ 

  if(id >= 0) {
    /*
        Find transformation matrix from "camera_color_optical_frame" to "origin".
    */
    tf::Transform localization_trans;
    /**************************************************************
    //                 Student Implementation                    //
    **************************************************************/

    /* publish the transform */
    nav_msgs::Odometry trans_odem;
    trans_odem.pose.pose.position.x = ???; //implement
    trans_odem.pose.pose.position.y = ???;
    trans_odem.pose.pose.position.z = ???;
    trans_odem.pose.pose.orientation.x = ???;
    trans_odem.pose.pose.orientation.y = ???;
    trans_odem.pose.pose.orientation.z = ???;
    trans_odem.pose.pose.orientation.w = ???;
    trans_odem.header.stamp = ros::Time::now();
    transform_pub.publish(???);
  }
  return;
}

int main(int argc, char** argv){
  ros::init(argc, argv, "apriltag_localization");
  ros::NodeHandle nh;
  tf_listener = new tf::TransformListener();
  
  // write the publisher
  transform_pub = nh.advertise<nav_msgs::Odometry>("???", ???);

  bool find = false;

  /* get the transform from "camera_color_optical_frame" to "camera_link" */
  string parent_id = ???;   // implememt
  string child_id = ???;
  while (!find) {
    tf_listener->waitForTransform(child_id, parent_id, ros::Time::now(), ros::Duration(0.7));
    try {
      tf_listener->lookupTransform(parent_id, child_id, ros::Time(0), camera_transform);
      cout << "Get transform from \"camera_color_optical_frame\" to \"camera_link\"!!!!!!!\n";
      find = true;
    }
    catch (tf::TransformException& ex){
      std::cout << "Exception thrown:" << ex.what() << std::endl;
    }
  }

  /* get the transform from "map_tag" to "origin" */
  for (int i = 0; i < 5; i++) {
    parent_id = ???;  // implememt
    child_id = ???;
    tf_listener->waitForTransform(child_id, parent_id, ros::Time::now(), ros::Duration(0.7));
    try {
      tf_listener->lookupTransform(parent_id, child_id, ros::Time(0), tag_transforms[i]);
      cout << "Get transform from map_tag_" << i << " to \"origin\"\n";
    }
    catch (tf::TransformException& ex){
      std::cout << "Exception thrown:" << ex.what() << std::endl;
    }    
  }

  while (ros::ok())
  {
    ros::spinOnce();
    listener();
  }
  
  return 0;
}