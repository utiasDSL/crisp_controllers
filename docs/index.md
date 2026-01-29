---
hide:
  # - navigation
  # - toc
---

<img src="media/crisp_logo_dark.webp#only-dark" alt="CRISP Controllers Logo"/>
<img src="media/crisp_logo_light.webp#only-light" alt="CRISP Controllers Logo"/>

# CRISP - **C**ompliant **R**OS2 Controllers for Learn**i**ng-Ba**s**ed Manipulation **P**olicies

*Authors: [Daniel San Jose Pro](https://danielsanjosepro.github.io)[^1], [Oliver Hausdörfer](https://oliver.hausdoerfer.de/)[^1], [Ralf Römer](https://ralfroemer99.github.io)[^1], Maximilian Dösch[^1], [Martin Schuck](https://amacati.github.io/) [^1] and Angela Schoellig [^1]*.
[^1]: The authors are with Technical University of Munich, Germany; TUM School of Computation, Information and Technology, Department of Computer Engineering, Learning Systems and Robotics Lab; Munich Institute of Robotics and Machine Intelligence.

You want to deploy your learning-based policy to your manipulator, or collect teleoperation data? `CRISP` provides the tools for that.

- `CRISP` provides the `ros2_control` low-level controllers (compliant, real-time, C++, torque-based) and simple python interfaces `CRISP_PY` to interface with them. It is compatible with any manipulator offering an joint-level torque effort interface.

- `CRISP` provides also a Gymnasium environment `CRISP_GYM` to deploy learning-based `LeRobot` policies and collect data in `LeRobotDataset` format.

_If you use this work, please cite it using the [bibtex](#citing) below._

Check the [controllers (CRISP controllers) :simple-github:](https://github.com/utiasDSL/crisp_controllers), the simple [python interface (CRISP_PY) :simple-github:](https://github.com/utiasDSL/crisp_py), and a [Gymnasium wrapper (CRISP_GYM) :simple-github:](https://github.com/utiasDSL/crisp_gym) for real-world experiments.


!!! info "Aloha gripper for Manipulators"
    Check out [aloha4franka](https://tum-lsy.github.io/aloha4franka/) for the gripper used in the videos.

<!-- <div> -->
<!--   <video src="../media/videos_concatenaded.mp4" width="800" playsinline muted controls loop autoplay  alt="Robot Collection"> -->
<!--   <div>Pick and Place demo</div> -->
<!-- </div> -->
<!---->

<div class="carousel-container">
  <div class="carousel">
    <div class="carousel-slide active">
      <video src="media/videos_concatenaded.mp4" playsinline muted loop autoplay  alt="Robot Collection"></video>
      <div class="carousel-caption">End-to-end policies in action performing different contact-rich tasks.</div>
    </div>
    <div class="carousel-slide">
      <video src="media/wrist_cam.mp4" playsinline muted loop autoplay  alt="Data visualized"></video>
      <div class="carousel-caption">Recorded Trajectories in LeRobot format for different tasks using CRISP.</div>
    </div>
    <div class="carousel-slide">
      <video src="media/viser_visualization.mp4" playsinline muted loop autoplay  alt="Viser"></video>
      <div class="carousel-caption">Provided example with 1x video of the tracking capabilities of the controller.</div>
    </div>
    <div class="carousel-slide">
      <video src="media/pap_demo.mp4" playsinline muted loop autoplay  alt="Teleoperation PaP"></video>
      <div class="carousel-caption">Leader-follower teleoperation for a pick and place task.</div>
    </div>
    <div class="carousel-slide">
      <img src="media/franka.gif" alt="Franka Robot Demo">
      <div class="carousel-caption">Controllers: Franka Robot - Following a moving target while base joint follows a sine curve</div>
    </div>
    <div class="carousel-slide">
      <img src="media/kinova.gif" alt="Kinova Robot Demo">
      <div class="carousel-caption">Controllers: Kinova Robot - Simulated robot with continuous joints and nullspace control</div>
    </div>
    <div class="carousel-slide">
      <img src="media/iiwa.gif" alt="IIWA Robot Demo">
      <div class="carousel-caption">Controllers: IIWA Robot - Impedance control in simulation</div>
    </div>
  </div>

  <button class="carousel-btn carousel-btn-prev" aria-label="Previous slide">&lt;</button>
  <button class="carousel-btn carousel-btn-next" aria-label="Next slide">&gt;</button>

  <div class="carousel-indicators"></div>
</div>

<div class="carousel-controls-info">
  <p><strong>Navigation:</strong> Use arrow keys ← → or click the buttons to navigate through the gallery.</p>
</div>
<!---->
<!-- | <video src="media/pap_demo.mp4" playsinline muted controls loop autoplay width="800"/> | <video src="media/policy.mp4" playsinline controls="true" loop="true" autoplay="true" width="800"/> | -->
<!-- |:--:|:--:| -->
<!-- | Robot teleoperated using a Follower-Leader system in [CRISP_GYM :simple-github:](https://github.com/utiasDSL/crisp_gym) | Diffusion Policy trained and deployed from the same demonstrations |  -->
<!---->
<!---->
<!-- | ![Franka](media/franka.gif) | ![kinova](media/kinova.gif) | ![iiwa](media/iiwa.gif) | -->
<!-- |:--:|:--:|:--:| -->
<!-- | *Robot following a moving target, while base joint follows a sine curve* | *Simulated kinova robot with continous joints and nullspace control* | *Simulated iiwa robot* | -->
<!---->
<!-- | ![franka_eight_reduced](media/franka_eight_reduced.gif)![franka_ns_reduced](media/franka_ns_reduced.gif) | ![vicon](media/franka_teleop.gif)| -->
<!-- |:--:|:--:| -->
<!-- | *Real robot following a target and being disturbed (contact) + null space control demonstration*  | *Demonstration using a cartesian controller teleoperated using Vicon tracking system (Speed x4)*|  -->
<!---->
<!-- | <video src="media/teleoperation.mp4" controls="true" loop="true" autoplay="true" playsinline width="800"> | -->
<!-- |:--:| -->
<!-- | Teleoperation setup with activated force-torque feedback | -->
<!---->


## Why?

Learning-based controllers, such as diffusion policies, deep reinforcement learning, and vision-action-models in general, typically output low-frequency or sporadic target poses, necessitating a low-level controller to track these references smoothly, especially in contact-rich environments.
While `ROS2` frameworks like `MoveIt` offer comprehensive motion planning capabilities, they are often unnecessarily complex for tasks requiring simple, real-time pose or joint servoing.

We present a set of lightweight, torque-based Cartesian and joint-space controllers implemented in C++ for `ros2_control`, compatible with any robot exposing an effort interface—a common standard among modern manipulators.
Our controllers incorporate friction compensation, joint limit avoidance, and error clipping, and have been validated on the Franka Robotics FR3 on hardware, and on various platforms in simulation.

We provide tooling to collect data in `LeRobotDataset` format using teleoperation and deploy learning-based policies with minimal effort using `CRISP_PY` and `CRISP_GYM`.

**Why the name "CRISP"**? "CRISP" reflects our design philosophy behind the package: a concise, to-the-point implementation for easy deployment and integration in other software stacks.

## Real World Deployments

Our controllers have been tested in real hardware on the following robots:
<div class="robot-grid">
  <div class="robot-card">
    <a href="https://github.com/danielsanjosepro/pixi_franka_ros2">
        <img src="media/fr3_visual.png" alt="FR3">
        <div class="robot-name">FR3</div>
    </a>
  </div>
  <div class="robot-card">
    <a href="https://github.com/lvjonok/pixi_panda_ros2">
        <img src="media/panda_visual.png" alt="Panda">
        <div class="robot-name">Panda</div>
    </a>
  </div>

  <div class="robot-card">
    <a href="https://github.com/danielsanjosepro/crisp_ur_demo">
        <img src="media/ur10_visual.png" alt="UR5">
        <div class="robot-name">UR5</div>
    </a>
  </div>
  <!-- <div class="robot-card greyscale"> -->
  <!--   <img src="media/dynaarm_visual.png" alt="UR10"> -->
  <!--   <div class="robot-name">DynaArm</div> -->
  <!-- </div> -->
</div>

Grippers tested in real hardware:
<div class="gripper-grid">
  <div class="gripper-card">
    <a href="https://github.com/danielsanjosepro/pixi_robotiq_ros2">
        <img src="media/robotiq_visual.png">
        <div class="gripper-name">Robotiq Gripper 2F-85</div>
    </a>
  </div>
  <div class="gripper-card">
    <a href="https://github.com/TUM-lsy/aloha4franka">
        <img src="media/aloha4franka_visual.png">
        <div class="gripper-name">aloha4franka / dynamixel-based gripper</div>
    </a>
  </div>
  <div class="gripper-card">
    <a href="https://github.com/danielsanjosepro/pixi_franka_ros2">
        <img src="media/franka_hand_visual.png">
        <div class="gripper-name">Franka Hand</div>
    </a>
  </div>
</div>

<!---->
<!-- ... and in simulation on: -->
<!-- <div class="robot-grid"> -->
<!--   <div class="robot-card"> -->
<!--     <img src="media/iiwa14_visual.png" alt="IIWA"> -->
<!--     <div class="robot-name">IIWA14</div> -->
<!--   </div> -->
<!--   <div class="robot-card greyscale"> -->
<!--     <img src="media/kinova_visual.png" alt=""> -->
<!--     <div class="robot-name">Kinova</div> -->
<!--   </div> -->
<!-- </div> -->
<!---->

Many thanks community contributions:

- Lev Kozlov [@lvjonok](https://github.com/lvjonok) for testing and providing interfaces for the Panda/FER.
- Vincenzo Orlando [@VinsOrl09](https://github.com/lvjonok) for testing and providing interfaces for the UR robots.
- Linus Schwarz [@Linus-Schwarz](https://github.com/Linus-Schwarz) for testing and providing interfaces for the BOTA force-torque sensors.

Check the robots in action:

<div class="video-container">
    <div class="video-item">
      <video src="media/crisp_ur5_test.mp4" playsinline muted loop autoplay  alt="Data visualized"></video>
      <div class="hover-text">UR5 Robot using a CRISP controller to track a circular trajectory while being disturbed.</div>
    </div>
    <div class="video-item">
      <video src="media/crisp_fr3_test.mp4" playsinline muted loop autoplay  alt="Data visualized"></video>
      <div class="hover-text">End-To-End policy stacking Lego bricks with the CRISP  controllers.</div>
    </div>
</div>



## Features
- 🐍 **Python interface** to move your ROS2 robot around without having to think about topics, spinning, and more ROS2 concepts but without losing the powerful ROS2 API. Check [CRISP_PY](https://github.com/utiasDSL/crisp_py) for more information and examples.
- 🔁 **Gymnasium environment** with utilities to deploy learning-based policies and record trajectories in LeRobotFormat. Check [CRISP_GYM](https://github.com/utiasDSL/crisp_gym).
- ❓ **Demos** showcasing how to use the controller with FR3 of Franka Emika in single and bimanual setup. Check the [crisp_controller_demos](https://github.com/utiasDSL/crisp_controllers_demos).
- ⚙️ Dynamically and highly parametrizable: powered by the [`generate_parameter_library`](https://github.com/PickNikRobotics/generate_parameter_library) you can modify stiffness and more during operation.  
- 🤖 Operational Space Controller as well as Cartesian Impedance Controller for torque-based control.  
- 🚫 No MoveIt or complicated path-planning, just a simple C++ `ros2_controller`. Ready to use.  


## Citing

You can check our preprint on arXiv: [2509.06819](https://arxiv.org/abs/2509.06819)

```bibtex
@misc{pro2025crispcompliantros2,
      title={CRISP - Compliant ROS2 Controllers for Learning-Based Manipulation Policies and Teleoperation}, 
      author={Daniel San José Pro and Oliver Hausdörfer and Ralf Römer and Maximilian Dösch and Martin Schuck and Angela P. Schöllig},
      year={2025},
      eprint={2509.06819},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.06819}, 
}
```
