# Final competition #
## Step1
```
$ git clone https://github.com/banbanhank/hcc-final-competition-2021.git
```
[Github](https://github.com/derekray311511/hcc_2022/Final_competition)  
Please copy the "darknet_ros" pakage you have use in Lab8 and Lab9 into your workspace.  
![](https://imgur.com/0dVP4uY.png)  

## Step2
Please finish the template shown below:
* hcc-final-competition-2021/hcc_ws/src/estimation_pos/src/apriltag_localization.cpp
* hcc-final-competition-2021/hcc_ws/src/estimation_pos/src/drone_object.py

Note that you have to catkin_make your code each time after you change your code.
```
$ catkin_make
```
## Step3
After finish your code, you have to test the result.
Please remember you have to source your environment workspace if you open a new terminal.
```
$ source devel/setup.bash
```
### Implementation

Run ROS master:  
`$ roscore`

Set use_sim_time to true:  
`$ rosparam set use_sim_time true`  
Then, 

> open one terminal(T1)
> ```
> roslaunch estimation_pos hcc2022_final_map.launch
> ```
> open another terminal(T2)
> ```
> roslaunch estimation_pos localization_final_2022.launch
> ```
> open another terminal(T3)
> ```
> roslaunch darknet_ros yolo_v3.launch
> ```
> open another terminal(T4)  
> ```
> rosrun estimation_pos apriltag_localization_2022
> ```
> open another terminal(T5)
> ```
> rosrun estimation_pos drone_object_2022.py
> ```
> open another terminal(T6) and play the bag
> ```
> rosbag play "the bag you want to play" -r 0.1 --clock
> ```
