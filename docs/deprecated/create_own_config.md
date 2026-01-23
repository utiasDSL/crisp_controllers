# How to create your own config

You can bring your own config to CRISP, so that you are able to create your own environments, teleoperation setup, controllers...

1. First create a config folder and give it the following structure (you do not need to add all config folders):

```bash
    my_crisp_configs
    ├── envs/
    │   ├── my_env1.yaml
    │   └── my_env2.yaml
    ├── recording/
    │   └── my_recording_manager.yaml
    ├── teleop/
    │   └── my_teleop_setup.yaml
    ├── control/
    │   ├── my_osc_controller.yaml
    │   ├── my_joint_controller.yaml
    │   └── my_cartesian_impedance_controller.yaml
    ├── my_gripper_config.yaml
    ├── my_second_gripper_config.yaml
    ├── my_robot_config.yaml
    └── my_camera_config.yaml

```
2. Then add it to your `CRISP_CONFIG_PATH`:
```bash
export CRISP_CONFIG_PATH=/path/to/my_crisp_configs
```

3. Check that the config works.

```bash
crisp-check-config  # Do this inside an environment with crisp_py or crisp_gym installed
```

This should output your config if it can be loaded properly

---

Now you can use this to create your own environments:

```python
from crisp_gym.manipulator_env import make_env

env = make_env(env_type="my_env1", namespace="my_robot_namespace_if_required")
```

Also the record and deploy scripts should be able to find your config now and allow you to load it.
