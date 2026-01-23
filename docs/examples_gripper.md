# Gripper Examples

This page provides a simple example on how to use grippers with `crisp_py`.

## Run a gripper node

We first start a gripper node that 

- Publishes gripper the value of the gripper `JointState`, where the position value range should be known for further configuration,
- Listens to commands of type `Float32` with the same range as the joint state values to open/close the gripper,
- Optionally, a service to disable the torque of the gripper motors.

Check the [getting ros2 side ready guide](getting_started_controllers.md) to see the examples with grippers.

For...

- ... the Franka Hand, a node is directly started with the FR3 or Panda pixi robots publishes the gripper state and listens to commands automatically.
- ... Dynamixel-based grippers, you can use [dynamixel_wrapper](https://github.com/danielsanjosepro/dynamixel_wrapper). See the README.md to calibrate and get the range of the gripper.

## Access Gripper from `crisp_py`

For simple binary control of the gripper (open/close), you can use the `Gripper` class from `crisp_py` or the `make_gripper` factory function if you defined the gripper in a YAML configuration file (see [config guide](getting_started_config.md) for more information).
```python
"""Simple example to control the gripper."""
import time

from crisp_py.gripper import make_gripper

gripper = make_gripper("gripper_franka")
# or using GripperConfig directly:
gripper.config.max_delta = 0.15
gripper.wait_until_ready()

# Open and close the gripper
gripper.open()
time.sleep(2.0)
gripper.close()
time.sleep(2.0)
gripper.shutdown()
```

If using a disable torque service, you can enable/disable the torque of the gripper motors:
```python
from crisp_py.gripper import Gripper, GripperConfig

config = GripperConfig(
    min_value=1046.0,
    max_value=2065.0,
    joint_state_topic="TODO/joint_states",
    command_topic="TODO/command",
    enable_torque_service="TODO/set_torque",
)
gripper = Gripper(config=config)
gripper.wait_until_ready()

gripper.enable_torque()
gripper.set_target(0.5)  # Set to mid position
```
