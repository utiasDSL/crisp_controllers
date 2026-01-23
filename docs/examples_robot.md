# Robot Examples

This page provides a simple example on how to use robots with `crisp_py`.

## Run a robot node

We first start a robot node that 

- Publishes robot the joint states to a topic as a `JointState`, usually with the name `joint_states`,
- (Optional) Publishes robot the current pose to a topic as a `PoseStamped`, usually with the name `current_pose` (if not, `crisp_py`'s `RobotConfig` can be configured to use TF frames),
- Listens to commands of type `PoseStamped` that are published to a topic, usually with the name `target_pose` (to control the end-effector in Cartesian space),
- Listens to commands of type `JointState` that are published to a topic, usually with the name `target_joint` (to control the null space of redundant robots or for joint control),
- Optionally, listen to `WrenchStamped` messages to apply forces/torques at the end-effector.

Check the [getting ros2 side ready guide](getting_started_controllers.md) to see the examples with robots.

We will use the Franka Robotics FR3 robot as an example, but other robots can be used similarly.
You can run the FR3 robot node (simulated or real) using the [pixi_franka_ros2](https://github.com/danielsanjosepro/pixi_franka_ros2):
```bash
pixi run -e jazzy franka  # or franka-sim for simulation
```

## Access Robot from `crisp_py`

The `Robot` class from `crisp_py` that serves as a mere interface to interact with the robot, with minimal logic.
- To easily retrieve the latest joint states and end-effector pose,
- To send target poses and joint states to the robot,
- To switch between different `ros2_controllers`
- To `home()` the robot to a predefined position.
- To linearly interpolate between two poses with `move_to()` if using a cartesian controller.

What it does not do:
- Some kind of trajectory planning or advanced motion planning.
- Safety checks (e.g., joint limits, singularities, collisions, etc.)
- IK computations...

Most of the low levels of control and safety is handled by the controller or should be handled by a user of this interface.


```python
"""Simple example to control the robot."""
import time

from crisp_py.robot import make_robot

robot = make_robot("fr3")
robot.wait_until_ready()  # make sure that all topics have been received

# %% First home
robot.home()  # will activate a joint trajectory controller and home the robot

# %% Check state
print(robot.end_effector_pose)
print(robot.joint_values)

# %% Activate cartesian impedance controller
robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
# Optionally, load custom parameters for the cartesian controller
robot.cartesian_controller_parameters_client.load_param_config(
    file_path="..."
)

# %% Move to a target pose
target_pose = robot.end_effector_pose
target_pose.position.z += 0.1  # Move up 10 cm

robot.set_target(pose=target_pose)  # This will directly send the target to the robot, use move_to for linear interpolation
```

A more advanced example to move the robot around using viser (requires `viser` dependencies):
```python
import time

import numpy as np
import viser
from viser.extras import ViserUrdf
from scipy.spatial.transform import Rotation
from robot_descriptions.loaders.yourdfpy import load_robot_description

from crisp_py.robot import make_robot
from crisp_py.utils.geometry import Pose

robot = make_robot("fr3")  # Change to your robot type
robot.wait_until_ready()

robot.config.time_to_home = 2.0
robot.home()
start_pose = robot.end_effector_pose

robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")

server = viser.ViserServer()

urdf = load_robot_description("panda_description")  # Change to your robot description loader
viser_urdf = ViserUrdf(
    server,
    urdf_or_path=urdf,
    load_meshes=True,
    load_collision_meshes=False,
    collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
)

with server.gui.add_folder("Visibility"):
    show_meshes_cb = server.gui.add_checkbox(
        "Show meshes",
        viser_urdf.show_visual,
    )
    show_collision_meshes_cb = server.gui.add_checkbox(
        "Show collision meshes", viser_urdf.show_collision
    )

@show_meshes_cb.on_update
def _(_):
    viser_urdf.show_visual = show_meshes_cb.value

@show_collision_meshes_cb.on_update
def _(_):
    viser_urdf.show_collision = show_collision_meshes_cb.value

config_with_gripper = np.array([*robot.joint_values, 0.0])
viser_urdf.update_cfg(config_with_gripper)

trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
server.scene.add_grid(
    "/grid",
    width=2,
    height=2,
    position=(
        0.0,
        0.0,
        # Get the minimum z value of the trimesh scene.
        trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
    ),
)
# Add interactive transform controls for the end effector.
transform_handle = server.scene.add_transform_controls(
    "/end_effector_target",
    position=start_pose.position,
    wxyz=start_pose.orientation.as_quat(scalar_first=True),
    scale=0.3,
    line_width=3.0,
)

# Add callback for when the transform handle is moved.
@transform_handle.on_update
def update_robot_target(handle: viser.TransformControlsEvent) -> None:
    rot = Rotation.from_quat(handle.target.wxyz, scalar_first=True)
    pose = Pose(position=handle.target.position, orientation=rot)
    robot.set_target(pose=pose)


while True:
    config_with_gripper = np.array([*robot.joint_values, 0.0])
    viser_urdf.update_cfg(config_with_gripper)
    time.sleep(0.01)
```

With this, you can control the robot as shown in the following video:

<div class="video-container">
    <div class="video-item">
        <video src="../media/viser_visualization.mp4" playsinline muted loop autoplay  alt="Viser"></video>
    </div>
</div>

Have fun controlling your robot!
