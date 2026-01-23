# Design Philosophy

We follow a few design strategies in CRISP.
This page describes our choices along with their respective pros and cons.

Our most important goal is to find a **meeting point between robotics and machine learning engineers.**
On the one hand, robotics engineers are already familiar with ROS2 and experienced in control and manipulation in general.
On the other hand, machine learning engineers who might not be familiar with these tools prefer to work with python-only environments.
We try to bring the strengths of both worlds in this project.

## Reuse Existing ROS2 Infrastructure

We decided to build CRISP on top of [ROS2](https://docs.ros.org/) for several reasons:

**Pros:**

- **Large community**: Many users and developers already familiar with tools for interacting with ROS2 systems (e.g., [rviz2](https://github.com/ros2/rviz), [rqt](https://docs.ros.org/en/rolling/Concepts/Intermediate/About-RQt.html), [ros2 CLI](https://docs.ros.org/en/rolling/Tutorials/Beginner-CLI-Tools.html)).
- **Existing ecosystem**: Well-tested packages for robot drivers, sensors, and visualization that we can leverage directly. The [ROS Index](https://index.ros.org/) lists thousands of available packages.
- **Simplified setup with pixi + robostack**: We use [pixi](https://pixi.sh/) with [robostack](https://robostack.github.io/) to provide a conda-like development environment.
This removes the traditional friction of ROS2 installation (no more sourcing setup.bash files or managing system dependencies) and makes it accessible to ML engineers accustomed to Python-centric workflows.

**Cons:**

- **"Framework jail"**: ROS2 imposes certain patterns (nodes, topics, executors) that may feel constraining for simple use cases. 
    Users must work within the ROS2 paradigm even for straightforward scripts.

In all objects, we try to abstract away ROS2 details behind simple Python APIs:
```python
from crisp_py import Robot

robot = Robot(...)              # ROS2 node creation is hidden and spinned up internally to receive/send data
robot.wait_until_ready()        # Waits for ROS2 topics to be alive
print(robot.end_effector_pose)  # Direct access to data without dealing with ROS2 messages
```

## Configuration: YAML Files with Programmatic Freedom

CRISP uses YAML configuration files to define robots, sensors, and control pipelines. 
This provides a declarative way to set up common scenarios without writing code. 

**Pros:**

- **Quick iteration**: Change parameters without modifying source code.

**Cons:**

- **Less flexibility**: Complex behaviors may be hard to express in YAML rather than code.

However, we also provide full programmatic access. 
Users can instantiate objects directly in Python and modify them freely when the YAML approach becomes limiting. 
This is particularly useful for:

- Rapid prototyping and debugging
- *Custom components* that don't fit the YAML schema

Here is an example for the YAML config for an environment:
```yaml
gripper_mode: "absolute_continuous"

robot_config:
  robot_type: "franka"
  time_to_home: 2.0
  publish_frequency: 50.0
  home_config: [0.0, 0.1, 0.0,-1.94, 0.0, 2.0, 0.8]

gripper_config:
    min_value: 0.0
    max_value: 0.4
    joint_state_topic: /gripper/joint_states
    command_topic: /gripper/command

camera_configs:
  - camera_name: "primary"
    camera_frame: "primary_link"
    resolution: [256, 256]
    camera_color_image_topic: "third_person_camera/image_raw"
    camera_color_info_topic: "third_person_camera/camera_info"
  - camera_name: "wrist"
    camera_frame: "wrist_link" 
    resolution: [256, 256]
    camera_color_image_topic: "wrist_camera/color/image_rect_raw"
    camera_color_info_topic: "wrist_camera/color/camera_info"

sensor_configs:
  - sensor_type: "force_torque"
    shape: [6,]
    name: "ft_sensor"
    data_topic: "external_wrench"
```

See the [example config files](https://github.com/utiasDSL/crisp_gym/tree/main/crisp_gym/config) or [here](https://github.com/utiasDSL/crisp_py/tree/main/crisp_py/config), and [how to define your own configs](getting_started_config.md) for more details.

## Data Collection directly in LeRobot Format, no rosbags

CRISP collects data directly in [LeRobot](https://github.com/huggingface/lerobot) format at a single fixed frequency, rather than saving [ROS bags](https://docs.ros.org/en/rolling/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html) for post-processing.

The LeRobot format stores episodes as [HuggingFace datasets](https://huggingface.co/docs/datasets/), making it easy to share, version, and load data for training. Each episode contains synchronized observations (images, joint states) and actions at a consistent frequency.

**Pros:**

- **Minimizes the gap between teleoperation and policy deployment**: The data you collect is immediately ready for training without conversion steps. What you record is exactly what your policy will see.
- **Consistent timing**: A single frequency ensures synchronized observations and actions, avoiding timestamp alignment issues common with bag files.
- **Simpler pipeline**: No need to manage bag files, replay them, and transform to training formats.
- **Easy sharing**: Datasets can be pushed directly to [HuggingFace Hub](https://huggingface.co/datasets) for collaboration.

**Cons:**

- **Less data captured**: You only save what's needed at the target frequency, potentially losing high-frequency sensor data that might be useful for debugging or alternative analysis.
- **Less flexibility**: Post-hoc resampling or different observation combinations require re-collection.

The recording is handled by the [RecordingManager](https://github.com/utiasDSL/crisp_gym/tree/main/crisp_gym/record) and here is an example script showing how the recording works [record_with_leader_follower](https://github.com/utiasDSL/crisp_gym/blob/main/crisp_gym/scripts/record_lerobot_format_leader_follower.py) 
More details on recording can be found in the [getting started with the gym](getting_started_gym.md) documentation.
