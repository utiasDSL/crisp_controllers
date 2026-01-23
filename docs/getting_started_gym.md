# Get started with `crisp_gym`

First, clone the repository:
```sh
git clone https://github.com/utiasDSL/crisp_gym
cd crisp_gym
```
Now, you should set a few things before installing everything.
Create a file `scripts/set_env.sh` which will be sourced every time that you run a command in your environment.
The script will not be tracked by git.
In this script you need to add a environment variables:

- `ROS_DOMAIN_ID` **(Required)**: which is used to define nodes that should be able to see each other. In our [ROS2 nodes](getting_started_controllers.md) they are set to 100 as default.
- `CRISP_CONFIG_PATH` **(Optional)**: which should be the path to a config folder similar to [config path of CRISP_PY](https://github.com/utiasDSL/crisp_py/tree/main/config) or [config path of CRISP_GYM](https://github.com/utiasDSL/crisp_gym/tree/main/crisp_gym/config) but with your own custom configurations.
    If this environment variable is unset, the default configurations will be used.
    Check [how to create your own config](getting_started_config.md) guide for more information.


```sh title="scripts/set_env.sh" hl_lines="3"
export GIT_LFS_SKIP_SMUDGE=1  # (1)!
export SVT_LOG=1  # (2)!
export ROS_DOMAIN_ID=100
export CRISP_CONFIG_PATH=/path/to/config1/folder:/path/to/config2/folder  # optional
```

1. This will avoid downloading large files when cloning the repository. You can always download them later with `git lfs pull`.
2. This will remove logging from SVT codecs, which are used to create data in LeRobot format. The logs can be quite verbose.

If you want to work in a *multi-machine setup* (e.g. policy runs in a different machine as controllers/cameras), then check [how to setup multi-machine in ROS2](getting_started_multiple_machines.md).

---
Now we can install the environment:

```sh
GIT_LFS_SKIP_SMUDGE=1 pixi install -e humble-lerobot
pixi shell -e humble-lerobot
python -c "import crisp_gym"
```

You can also check that your configs are set up with:

```sh
pixi run -e humble-lerobot crisp-check-config
```

If the previous steps worked, then you are good to go.

### Teleoperation: Record data in [LeRobotFormat](https://github.com/huggingface/lerobot)

You can record data in `LeRobotFormat` to train a policy directly in [LeRobot](https://github.com/huggingface/lerobot).
You will need to use teleoperation to record data and we highly recommend using a leader-follower setup to generate episodes. 

#### Leader-follower

The leader can be controlled by a human operator and the follower will mimic its motion.
Checkout `scripts/leader_follower_teleop.py` to get an idea on how the code works.
For your specific setup you need to:

- Define your own `TeleopRobotConfig`, check [`teleop_robot_config.py`](https://github.com/utiasDSL/crisp_gym/blob/main/crisp_gym/teleop/teleop_robot_config.py).
- Define your own `ManipulatorEnvConfig`, check [`manipulator_env_config.py`](https://github.com/utiasDSL/crisp_gym/blob/main/crisp_gym/manipulator_env_config.py).

Then, to record data use:
```sh
pixi run -e humble-lerobot crisp-record-leader-follower \
   --repo-id <your_account>/<repo_name> # (1)!
```

1. Add `--help` to check other parameters to pass to the record function.

The script is interactive. It will first ask to choose the desired configuration files for the recording and then allow you to record episodes interactively.
There are two recording methods currently available:

- `keyboard` (default): It allows you to record episodes using the keyboard with the keys 
    - __r__(ecord start/stop) an episode,
    - __d__(elete episode) after recording a failed episode,
    - __s__(ave episode) after recording a successful episode,
    - __q__(uit) after finishing.
- `ros`: It uses the topic `recording_state` to catch `String` ROS2 messages to follow the same recording workflow as the keyboard. 
    With this you can implement custom recording devices to control the recording workflow

    ??? example "Using the FR3 pilot buttons of Franka Robotics as a recording device"
        In our lab, we use the buttons of the leader robot as a recording device with a fork of the [franka-buttons](https://github.com/danielsanjosepro/franka_buttons_ros2/tree/main) repository.
        The following script uses the circle, cross, check and up buttons as a record, delete, save and quit commands respectively (this is also part of the repository):
        ```py
        """Send recording commands for an episode recorder node to start, stop recording, save episodes and quit using the franka pilot buttons."""
        import rclpy
        from rclpy.node import Node

        from franka_buttons_interfaces.msg import FrankaPilotButtonEvent
        from std_msgs.msg import String


        class ButtonToRecordMessage(Node):
            """Node that subscribes to the button event and toggles the gripper when the circle button is pressed."""

            def __init__(self) -> None:
                super().__init__("button_to_record_message")

                self.create_subscription(
                    FrankaPilotButtonEvent, "franka_pilot_button_event", self.button_callback, 10
                )

                self.publisher = self.create_publisher(String, "record_transition", 10)

                # Add a cooldown to avoid multiple toggles
                self._last_toggle = self.get_clock().now()
                self._cooldown = 0.5

                self.get_logger().info("ButtonToRecordMessage node started.")

            def button_callback(self, msg: FrankaPilotButtonEvent):
                """Callback function for the button event.

                If circle pressed, then pass the command to the gripper client to toggle the gripper.
                """
                if (self.get_clock().now() - self._last_toggle).nanoseconds < self._cooldown * 1e9:
                    return

                if msg.pressed:
                    if msg.pressed[0] == "circle":
                        self.get_logger().info("Circle button pressed. Sending a record message.")
                        self.publisher.publish(String(data="record"))
                    if msg.pressed[0] == "check":
                        self.get_logger().info("Check button pressed. Sending a save episode message.")
                        self.publisher.publish(String(data="save"))
                    if msg.pressed[0] == "cross":
                        self.get_logger().info("Cross button pressed. Sending a delete episode message.")
                        self.publisher.publish(String(data="delete"))
                    if msg.pressed[0] == "up":
                        self.get_logger().info("UP button pressed. Sending a quit command message.")
                        self.publisher.publish(String(data="exit"))

                    self._last_toggle = self.get_clock().now()


        def main():
            rclpy.init()
            node = ButtonToRecordMessage()
            rclpy.spin(node)
            rclpy.shutdown()


        if __name__ == "__main__":
            main()
        ```

After this, you can visualize the episodes with rerun visualizer and LeRobot utils:
```sh
pixi run -e lerobot python -m lerobot.scripts.visualize_dataset \
        --repo-id <your_account>/<repo_name> \
        --episode-index 0
```
...or use the [online tool for visualization](https://huggingface.co/spaces/lerobot/visualize_dataset).

!!! warning
    LeRobot is subject to frequent changes. This command might change in future versions.

#### Other teleop setups

You can add further teleop options to [`teleop/`](https://github.com/utiasDSL/crisp_gym/blob/main/crisp_gym/teleop) and create 
a similar record script to [`scripts/record_lerobot_format_leader_follower.py`](https://github.com/utiasDSL/crisp_gym/blob/main/crisp_gym/scripts/record_lerobot_format_leader_follower.py)

### Train a policy

You can use LeRobot train scripts to train a policy simply by running:
```sh
pixi run -e lerobot python -m lerobot.scripts.lerobot-train \
          --dataset.repo_id=<your_account>/<repo_name> \
          --policy.type=diffusion \
          --policy.num_inference_steps=10
```

!!! warning
    LeRobot is subject to frequent changes. This command might change in future versions.

They provide the latest implementations of most major SOTA models in pytorch.
Check [LeRobot](https://github.com/huggingface/lerobot) for more information.

### Deploy policy

After training with LeRobot, you can deploy the policy with:
```sh
pixi run -e humble-lerobot crisp-deploy-policy  --policy # (1)!
```

1. The script will interactively allow you to choose a model inside `outputs/train`. If you want to explicitly pass a path you can override it with `--path`

!!! warning
    LeRobot is subject to frequent changes. This command might change in future versions.

Good job, now you can evaluate your model!

