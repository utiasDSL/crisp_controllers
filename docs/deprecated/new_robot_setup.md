For ROS2 users, adding our controllers to their stack should be straightforward as it works like any other ROS2 controller.
For novice ROS2 users, we prepared a more detailed guide to make sure that you get your robot ready:

1. First make sure that your ROS2 drivers for the robot you want to use allow *direct-torque control* i.e. the effort command interface is available.
Usually, you should be able to find it in the Github repository. 
If this is the case, you are good to go. 
Otherwise, you might need to use [cartesian_controllers](https://github.com/fzi-forschungszentrum-informatik/cartesian_controllers) for position controlled robots.

1. Install the ROS2 drivers on your computer. 
We recommend to work in a devcontainer to avoid installing ROS2 directly in your machine.
You can check the [demos](https://github.com/utiasDLS/crisp_controllers_demos) for inspiration.

3. Add the [controllers](https://github.com/utiasDSL/crisp_controllers) to your src folder where the ROS2 drivers for your robot have been installed:
    ```bash
    cd ~/ros2_ws  # or wherever you ws is...
    git clone https://github.com/utiasDSL/crisp_controllers.git src/crisp_controllers
    source /opt/ros/$ROS_DISTRO/setup.sh
    source /install/setup.sh
    rosdep update
    rosdep install -q --from-paths src --ignore-src -y  # (1)!
    colcon build --packages-select crisp_controllers 
    touch src/crisp_controllers/COLCON_IGNORE  # (2)! 
    source /install/setup.sh  # (3)!
    ```

    1. This line will make sure that all missing dependencies are installed
    2. By adding this file, the controller will not be built over and over
    3. Don't forget to source again to make sure that ROS2 can find the new package

4. Now, you will need to add the controller to the config file of the controller_manager.
Usually, you will find this config file in the bringup package where the launch files are located (called `<robotname>_bringup`) and are saved as `controllers.yaml`.
Check out the [FR3 config](https://github.com/utiasDSL/crisp_controllers_demos/blob/main/crisp_controllers_robot_demos/config/fr3/controllers.yaml) to get an idea how the config file looks like.
For more information on the controllers, check the [available controllers and broadcaster](controllers.md) page.

    ??? example "How to add the configuration to the config file"

        ```yaml

        /**:
        controller_manager:
            ros__parameters:
            update_rate: 1000  # Hz

            pose_broadcaster:
                type: crisp_controllers/PoseBroadcaster

            gravity_compensation:
                type: crisp_controllers/CartesianController

            cartesian_impedance_controller:
                type: crisp_controllers/CartesianController

            joint_impedance_controller:
                type: crisp_controllers/CartesianController

            # more controllers...
        /**:
        pose_broadcaster:
            ros__parameters:
            joints:
                - <TODO>

            end_effector_frame: <TODO>
            base_frame: base

        gravity_compensation:
            ros__parameters:
            joints:
                <TODO>

            end_effector_frame: <TODO>
            base_frame: base

            task:
                k_pos_x: 0.0
                k_pos_y: 0.0
                k_pos_z: 0.0
                k_rot_x: 30.0
                k_rot_y: 30.0
                k_rot_z: 30.0

            nullspace: 
                stiffness: 0.0

            use_friction: true
            use_coriolis_compensation: true
            use_local_jacobian: true

        joint_impedance_controller:
            ros__parameters:
            joints:
                <TODO>

            end_effector_frame: <TODO>
            base_frame: base

            task:
                k_pos_x: 0.0
                k_pos_y: 0.0
                k_pos_z: 0.0
                k_rot_x: 0.0
                k_rot_y: 0.0
                k_rot_z: 0.0

            max_delta_tau: 0.5

            nullspace: 
                stiffness: 5.0
                projector_type: none  # So we are directly controlling the joints!
                damping: 0.5
                max_tau: 5.0
                regularization: 1.0e-06
                weights:
                <TODO>

            use_friction: true
            use_coriolis_compensation: true
            use_local_jacobian: true
            limit_error: true
            limit_torques: true


        cartesian_impedance_controller:
            ros__parameters:
            joints:
                - <TODO>

            end_effector_frame: <TODO>
            base_frame: <TODO>

            task:
                k_pos_x: 400.0
                k_pos_y: 400.0
                k_pos_z: 400.0
                k_rot_x: 30.0
                k_rot_y: 30.0
                k_rot_z: 30.0

            nullspace: 
                stiffness: 2.0

            use_friction: false
            use_coriolis_compensation: true
            use_local_jacobian: true
        ```

5. Finally add them to your launch file. 
    You will need to use the controller_manager spawner to start the controllers at launch.
    Usually, in the launch file you will find other controllers/broadcasters that are being launched.
    Just duplicate the nodes and pass the correct names to activate the controllers.
    Check out [FR3 launch file](https://github.com/utiasDSL/crisp_controllers_demos/blob/main/crisp_controllers_robot_demos/launch/franka.launch.py) for inspiration.

    ??? example "How to add the controllers to the launch file"
        Add the following nodes to the launch description:

        ```py
        ...
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["cartesian_impedance_controller", "--inactive"],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_impedance_controller", "--inactive"],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["gravity_compensation", "--inactive"],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["pose_broadcaster"],
            output="screen",
        ),
        ...
        ```

6. After launching your robot you should see that new controller are being loaded. If you get stuck somewhere in the process feel free to open an issue.
7. Finally, to use the robots in CRISP_PY, add a configuration file for the new robot and Gymnasium environments that use it.

    ??? example "New robot config example"

        ```py title="my_new_robot/robot_config.py"
        from crisp_py.robot_config import RobotConfig

        @dataclass
        class MyNewRobotConfig(RobotConfig):

            joint_names: list = field(
                default_factory=lambda: [
                    "joint1",
                    "joint2",
                    "joint3",
                    ...
                ]
            )
            home_config: list = field(
                default_factory=lambda: [
                    np.pi,
                    0.0,
                    0.0,
                    ...,
                ]
            )
            base_frame: str = "base"
            target_frame: str = "target_frame"
        ```
        You can now use this config for your robot:
        ```py title="your_test_script.py"

        from crisp_py.robot import Robot
        from my_new_robot.robot_config import MyNewRobotConfig

        my_new_robot_config = MyNewRobotConfig()
        my_new_robot = Robot(config=my_new_robot_config, namespace=...)

        my_new_robot.home()
        ...

        ```
        In a similar manner, you can add this config to an Gymnasium environment to create a Gymnasium env with this config!

8. Voila, you are good to go!

