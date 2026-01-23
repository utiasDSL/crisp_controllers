You can use the demos to start robots and cameras in simulation or real and command them using CRISP_PY and CRISP_GYM.
For instance, you could run in one terminal shell the container for your robot and different shells the containers for
the cameras.

??? Example "Example setup in our lab"
    First we start our robots (FR3s):
    ```bash
    docker compose up launch_dual_franka
    ```
    Then we start the cameras:
    ```bash
    CAMERA_NAMESPACE=right SERIAL_NO=xxxxxxxx docker compose up launch_realsense
    ```
    ...and another camera:
    ```bash
    CAMERA_NAMESPACE=right docker compose up launch_camera
    ```
    We have a zellij/tmux session which starts all of these commands in individual panes, to be able to stop and restart them easily.
    Then, we are able interact with these containers directly using CRISP_PY or CRISP_GYM.
    
    Check "Running a demo" for more information on how to launch the demos.


## Available demos
For now, these are the available demos in this repository. New demos are welcome, in particular if tested with real hardware.
Some other manipulators that could be added to this list is [Duatic](https://github.com/Duatic/dynaarm_driver) or other dual setups.

| Robots | Franka Robotics FR3 | FR Dual FR3 | IIWA 14 | Kinova Gen3 |
| :--- | :---: | :---: | :---: | :---: |
| MuJoCo simulated Hardware | ✅ | ✅ | ✅ | ✅ |
| Real Hardware | ✅ | ✅ | ❔[^1]  | ❔[^1] |

[^1]: Untested, but effort interface available.

We also have some examples with cameras.

| Robots | Real Sense | Any Camera / Webcam | Orbecc |
| :--- | :---: | :---: | :---: |
| Camera demo | ✅ | ✅ |  ✅[^2] | 


[^2]: Available container at: https://github.com/danielsanjosepro/orbec_container_ros2


## Running a demo


1. Clone the repo
```bash
git clone git@github.com:utiasDSL/crisp_controllers_demos.git crisp_controllers_demos
cd crisp_controllers_demos
```

!!! WARNING
    Do NOT use **Docker Desktop**. Just go for the normal Docker CLI.

2. Build and start the provided container.
```bash
docker compose build
```
3. Start your robot:
```bash
docker compose up launch_iiwa
```
or
```bash
LEFT_ROBOT_IP=172.16.1.2 \
RIGHT_ROBOT_IP=172.16.0.2 \
FRANKA_FAKE_HARDWARE=true \
docker compose up launch_dual_franka
```
or
```bash
ROBOT_IP=172.16.0.2 \
FRANKA_FAKE_HARDWARE=true\
docker compose up launch_franka
```
or
```bash
docker compose up launch_kinova
```

4. Now you can publish to `/target_joint`, `/target_wrench` or `/target_pose`! Check [crisp_py](https://github.com/utiasDSL/crisp_py) examples to see how to easily use it.

!!! WARNING
    If you work in different machines (using [crisp_py](https://github.com/utiasDSL/crisp_py) or others) you might want to consider using a different RMW.
    Check [how to multi-machine setup](multi_machine_setup.md).
    To use a different middleware just pass an extra environment variable:
    ```bash
    RMW=<zenoh|cyclone> docker compose up ...
    ```



For the cameras, you can run:

```bash
docker compose up launch_camera
```

...and in case you are running a realsense:

```bash
docker compose up launch_realsense
```

Check the `docker-compose.yaml` to see how to define your own services.

## Troubleshooting

??? "Rviz does not open when launch the robots. Why?"
    Simply run `xhost +` on a terminal to enable any host to use the X. You need this because we are running the demos in a container.

??? "When executing `ros2 topic list` in a different terminal, I can not see any topics. However, the container is running. Why?"
    Probably you are using a different `ROS_DOMAIN_ID`. The default now is set to 100 but you can change it when running the container. To change it in your shell run `export ROS_DOMAIN_ID=100 && ros2 daemon stop && ros2 daemon start`.

??? "How are the manipulors being simulated?"
    We implemeted a simple `MujocoHardwareInterface` to simulate the robots. This code is heavily inspired by the simulator in <a href="https://github.com/fzi-forschungszentrum-informatik/cartesian_controllers/tree/ros2/cartesian_controller_simulation">cartesian_controllers</a>, but probably better alternatives to use mujoco as a backend simulation would be <a href="https://github.com/moveit/mujoco_ros2_control">mujoco_ros2_control</a>. One could also use gazebo. 
    The mujoco files come from the mujoco menagerie and have been slightly modified to use torque based actuators + we added some friction to the joints (to increase realism).

