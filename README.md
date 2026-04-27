<img src="media/crisp_logo.webp" alt="CRISP Controllers Logo"/>

<a href="https://github.com/utiasDSL/crisp_controllers/actions/workflows/humble_ros2_ci.yml"><img src="https://github.com/utiasDSL/crisp_controllers/actions/workflows/humble_ros2_ci.yml/badge.svg"/></a>
<a href="https://github.com/utiasDSL/crisp_controllers/actions/workflows/jazzy_ros2_ci.yml"><img src="https://github.com/utiasDSL/crisp_controllers/actions/workflows/jazzy_ros2_ci.yml/badge.svg"/></a>
<a href="https://github.com/utiasDSL/crisp_controllers/actions/workflows/kilted_ros2_ci.yml"><img src="https://github.com/utiasDSL/crisp_controllers/actions/workflows/kilted_ros2_ci.yml/badge.svg"/></a>
<a href="https://github.com/utiasDSL/crisp_controllers/actions/workflows/rolling_ros2_ci.yml"><img src="https://github.com/utiasDSL/crisp_controllers/actions/workflows/rolling_ros2_ci.yml/badge.svg"/></a>
<a href="https://danielsanjosepro.github.io/crisp_controllers/"><img alt="Static Badge" src="https://img.shields.io/badge/docs-passing-blue?style=flat&link=https%3A%2F%2Fdanielsanjosepro.github.io%2Fcrisp_controllers%2F"></a>
<a href="https://utiasDSL.github.io/crisp_controllers#citing"><img alt="Static Badge" src="https://img.shields.io/badge/arxiv-cite-b31b1b?style=flat"></a>

CRISP is a collection of real-time, C++ controllers for compliant torque-based control for manipulators compatible with `ros2_control`, including **Cartesian Impedance Control** and **Operational Space Control**. Developed for deploying high-level learning-based policies (VLA, Diffusion, ...) and teleoperation on your manipulator. It is robot-agnostic and compatible with any manipulator offering and effort interface. Check the [project website](https://utiasdsl.github.io/crisp_controllers/) for guides, getting started, demos and more! 


> [!NOTE]
> To reduce maintenance overhead, all ROS 2 distributions are supported from a single `main` branch. The code uses compile-time macros to handle version-specific differences.

## Features
- 🐍 **Python interface** to move your ROS2 robot around without having to think about topics, spinning, and more ROS2 concepts but without loosing the powerful ROS2 API. Check [crisp_py](https://github.com/utiasDSL/crisp_py) for more information and examples.
- 🔁 **Gymnasium environment** with utilities to deploy learning-based policies and record trajectories in LeRobotFormat. Check [crisp_gym](https://github.com/utiasDSL/crisp_gym).
- ❓ **Demos** showcasing how to use the controller with FR3 of Franka Emika in single and bimanual setup. Check the [crisp_controller_demos](https://github.com/utiasDSL/crisp_controllers_demos).
- ⚙️ Dynamically and highly parametrizable: powered by the [`generate_parameter_library`](https://github.com/PickNikRobotics/generate_parameter_library) you can modify stiffness and more during operation.  
- 🤖 Operational Space Controller as well as Cartesian Impedance Controller for torque-based control.  
- 🚫 No MoveIt or complicated path-planning, just a simple C++ `ros2_controller`. Ready to use.  


______

### For Contributors

##### Updating the website

We use [zensical](https://www.zensical.org/) to generate the website from markdown. You can modify it within `docs/` in particular the `index.md`.
You can run the website locally with:
```bash
pixi run zensical serve
```
The website is automatically generated and deployed on GitHub pages with the CI. You can check the [project website](https://utiasdsl.github.io/crisp_controllers/) for guides, getting started, demos and more!
