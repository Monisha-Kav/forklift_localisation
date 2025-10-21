# Forklift Localization Project

This repository contains the complete **Forklift Localization** system, implemented for both **hardware** and **simulation** setups.  
It is built for **Ubuntu 22.04 (Jammy Jellyfish)** and **ROS 2 Humble**.

---

## Overview

The project localizes a forklift robot using **sensor fusion** with data from:

- **IMU**
- **Wheel odometry**
- **ArUco marker detections**

The fusion is implemented through different filtering approaches — **Bayes**, **Kalman**, and **Extended Kalman Filters (EKF)** — all of which can be tested and compared across real hardware and Gazebo simulation environments.

**Exact details of the project, including methodology, design choices, and experimental results, are documented in the project report.**

---

## Repository Structure
```bash
forklift_localisation/
├── my_aruco_detector/   # Hardware setup (real robot- TURTLEBOT3 burger)
└── my_simulation/       # Simulation setup (Gazebo Classic)
```


---

## 1. Hardware Setup (`my_aruco_detector`)

This package is intended to be used **in the same workspace as your TurtleBot3 installation**.  
It integrates real IMU, ArUco marker detection, and the sensor fusion nodes for localization.

### Running the System

```bash
ros2 launch my_aruco_detector <launch_file_name>
```
Example:
```bash
ros2 launch my_aruco_detector localization_kalman.launch.py
```

### Key Contents

**Launch files:**
Found in my_aruco_detector/launch/, these include setups for Kalman, Bayes, EKF, IMU calibration, and testing variations.

**Main scripts:**
Located in my_aruco_detector/my_aruco_detector/, this folder contains:

- markwe_pose_detect.py – ArUco detection node
- forklift_display.py / new_forklift_display.py – GUI display for robot pose
- kf_full_sensor_fusion_m.py, sensor_fusion_node.py, etc. – All Kalman and Bayes filter implementations
- imu_calibration.py, accelerometer_reader_node.py – IMU handling and calibration
- Noisy accelerometer data logs and experimental data

Only a subset of these nodes are linked in the launch files, but others can be run manually to compare the performance of different filters over time.


## Simulation Setup (my_simulation)

This package replicates the real-world setup using Gazebo Classic, with virtual ArUco markers, camera, and sensor nodes.

### Running the Simulation

```bash
ros2 launch my_simulation sim_launch.launch.py
```
Then, depending on the desired filter, you can run:
```bash
ros2 launch my_simulation localization_kf_fused.launch.py
ros2 launch my_simulation localization_bayes_fused.launch.py

```

### Key Contents

**Launch files:** Preconfigured setups for Kalman and Bayes filters in simulation.

**Models folder:** Includes 3D models and textures for multiple ArUco markers and camera models.

**Worlds folder:** Contains several Gazebo world files (e.g., sim_world.world, empty_world.world).

**Simulation nodes:** Located in my_simulation/my_simulation/, including:

- Sensor fusion nodes
- Forklift display (forklift_display_sim.py)
- Camera follower node for tracking the robot
- Live ArUco detection scripts

## Filter Implementations

Inside both my_aruco_detector/my_aruco_detector/ and my_simulation/my_simulation/, you will find multiple versions of filter implementations:

**Bayesian Filters:** sensor_fusion_node_bayes.py

**Kalman Filters:** kf_full_sensor_fusion_m.py, simple_kalman_fusion.py

**Extended Kalman Filter:** real_ekf.py

Users can experiment with these versions to observe differences in localization performance and convergence over time.

## Visualization

The Forklift Display (forklift_display.py / new_forklift_display.py) provides a simple 2D visualization of the robot’s pose and environment:

- Displays the robot orientation and detected landmarks.
- Updates in real time as sensor fusion data is published.
- Supports interaction during experiments.

## Additional Materials

A separate folder (project media) contains:

- Final presentation slides (containing the details of the projec, and the implementation stages)
- Videos demonstrating localization results for different filters throughout project development.

These illustrate the performance evolution and accuracy improvements achieved through each stage.

## System Requirements

- **OS:** Ubuntu 22.04 (Jammy Jellyfish)
- **ROS 2 Distribution:** Humble Hawksbill
- **Gazebo:** Classic version
- **Python:** ≥3.10
- **Dependencies:**
  - `cv2` (OpenCV)
  - `numpy`
  - `rclpy`
  - `pygame`
  - `tf_transformations`
