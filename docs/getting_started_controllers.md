# Getting started with the ROS2 side

On the ROS2 side, we want to:

1. We want to run a low-level C++ [CRISP](https://github.com/utiasDSL/crisp_controllers) controller in your manipulator.
    The computer running the CRISP controller might need a real-time patch for the controller to run smoothly and safely. 
    You can check out the [Franka Robotics guide on how to set up a RT-patch.](https://frankarobotics.github.io/docs/installation_linux.html#setting-up-the-real-time-kernel)
    On newer Ubuntu versions, you can use [Ubuntu Pro](https://ubuntu.com/real-time) for an easy setup.
2. Start nodes for the cameras, the grippers and other sensors.

For both we provide (1) pixi ready-to-use repositories that launch the nodes or (2) docker container demos of robots test that are ready to use as well as for cameras, grippers and other sensors.
If your robot, camera, gripper or sensor is not listed below, checkout [how to setup a new robot](new_robot_setup.md) and get inspired by other examples to create your own setup.

## Start ROS2 nodes with pixi (recommended)

To start ROS2 nodes for your robot, camera or sensor we recommend using `pixi` as a ready-to-use solution to start ROS2 nodes.
Here are some  examples of repositories with `pixi` support for different hardware.

### Manipulators

Robots that we tested are:

- **FR3** (real and simulated): https://github.com/danielsanjosepro/pixi_franka_ros2
- **FER/Panda** (real): https://github.com/lvjonok/pixi_panda_ros2
- **UR5** (real): https://github.com/danielsanjosepro/crisp_ur_demo
- **IIWA** (simulated): https://github.com/danielsanjosepro/pixi_iiwa_ros2

### Grippers

Grippers that we tested are:

- **Franka Hand**: a node is included in the [fr3 pixi](https://github.com/danielsanjosepro/pixi_franka_ros2) and [panda pixi](https://github.com/lvjonok/pixi_panda_ros2) repositories which is started automatically with the robot. If the Franka Hand is not connected, the node will crash silently. Checkout [this config](https://github.com/utiasDSL/crisp_py/blob/main/crisp_py/config/grippers/gripper_franka.yaml) for using it with `crisp_py`.
- Any **Dynamixel**-based gripper: https://github.com/danielsanjosepro/dynamixel_wrapper - check README.md
- **Robotiq 2F-85**: https://github.com/danielsanjosepro/pixi_robotiq_ros2

### Cameras

Cameras that we tested are:

- Any **USB camera** (with usb_cam): https://github.com/danielsanjosepro/pixi_usbcam_ros2
- **Real Sense**: https://github.com/danielsanjosepro/pixi_realsense_ros2

### Sensors

Sensors that we tested are:

- **Anyskin** tactile sensor: https://github.com/danielsanjosepro/anyskin_ros2
- **Force/Torque** sensors: https://github.com/utiasDSL/botasys_ft_sensor_ros2

## Start ROS2 nodes with docker containers

We provide ready-to-use docker container demos for different manipulators and cameras (as an alternative to `pixi`).
New demos are welcome, in particular if tested with real hardware.
Some other manipulators that could be added to this list is [Duatic](https://github.com/Duatic/dynaarm_driver), [Universal Robots](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver) or other dual setups.

### Available Demos

| Robots | Franka Robotics FR3 | FR Dual FR3 | IIWA 14 | Kinova Gen3 | UR5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| MuJoCo simulated Hardware | ✅ | ✅ | ✅ | ✅ | ✅ [^2]|
| Real Hardware | ✅ | ✅ | ❔[^1]  | ❔[^1] |  ✅ [^2] |

[^1]: Untested, but effort interface available.
[^2]: Available at: https://github.com/danielsanjosepro/crisp_ur_demo

We also have some examples with cameras.

| Robots | Real Sense | Any Camera / Webcam | Orbecc |
| :--- | :---: | :---: | :---: |
| Camera demo | ✅ | ✅ |  ✅[^2] | 

[^2]: Available container at: https://github.com/danielsanjosepro/orbecc_container_ros2

### How to

Clone the repo
```bash
git clone git@github.com:utiasDSL/crisp_controllers_demos.git crisp_controllers_demos
cd crisp_controllers_demos
```

!!! WARNING
    Do NOT use **Docker Desktop**. Just go for the normal Docker CLI.

Start your robot or camera with:
```bash
docker compose up launch_xxx
```

Check the [docker-compose.yaml](https://github.com/utiasDSL/crisp_controllers_demos/blob/main/docker-compose.yaml) for the available launch_xxx options.

!!! WARNING
    If you work in different machines, you might want to consider using a different RMW.
    To use a different middleware just pass an extra environment variable:
    ```bash
    RMW=<zenoh|cyclone> docker compose up ...
    ```

