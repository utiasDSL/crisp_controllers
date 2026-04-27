

!!! Info 
    Please check the [controller implementation and configuration](https://github.com/utiasDSL/crisp_controllers/tree/main/src) for more details on these extra terms and how to enable them.

<div class="video-container">
    <div class="video-item">
        <video src="../media/viser_visualization.mp4" playsinline muted loop autoplay  alt="Viser"></video>
    </div>
</div>

The low-level controllers are torque-based controllers that take a `target_pose` or `target_joint` as input and compute the required torques to move the robot to that target.

## Joint control

For joint control, we use a simple PD controller to compute the desired torques based on the error between the current and target joint positions and velocities.
The torque command is computed as:

$$ \boldsymbol{\tau}_\text{cmd} = \mathbf{K}_p \mathbf{e} + \mathbf{K}_d \mathbf{\dot{e}}$$

where:

-  \( \boldsymbol{\tau} \)  is the vector of joint torques to be applied,
-  \( \mathbf{K}_p \)  is the diagonal matrix of proportional gains,
-  \( \mathbf{K}_d \)  is the diagonal matrix of derivative gains,
-  \( \mathbf{e} = \mathbf{q}_\text{target} - \mathbf{q}_\text{current} \)  is the position error,
-  \( \mathbf{\dot{e}} = \mathbf{\dot{q}}_\text{target} - \mathbf{\dot{q}}_\text{current} \)  is the velocity error, usually only \(-\mathbf{\dot{q}}_\text{current}\) is considered, as the desired velocity is often not set. This term acts as a damping term to slow down the motion.

## Cartesian control

For Cartesian Impedance control, we imagine a virtual spring-damper system between the current end-effector pose and the desired target pose.
The desired torque command $\boldsymbol{\tau}_\text{cmd}$ is computed by first calculating the desired force/torque at the end-effector $\mathcal{F}_\text{desired}$ as

$$ \boldsymbol{\tau}_\text{cmd} = \mathbf{J}^\top \mathcal{F}_\text{target} + \boldsymbol{\tau}_\text{nullspace}$$

where:

- \( \mathbf{J} \) is the robot's geometric Jacobian matrix w.r.t. the base or the world frame (can be configured).
- \( \boldsymbol{\tau}_\text{nullspace} \) is an optional nullspace torque to regulate joint positions.

### Desired end-effector force/torque
In our case the desired end-effector force/torque is computed using a PD controller in Cartesian space:

$$ \mathcal{F}_\text{desired} = \mathbf{K}_p (\mathbf{X}_\text{target}\ominus\mathbf{X}_\text{current}) - \mathbf{K}_d \mathbf{J} \mathbf{\dot{q}}_\text{current} $$

where:

- \( \mathbf{X}_\text{target}\ominus\mathbf{X}_\text{current} \) is the 6D pose error between the target and current end-effector poses, and its computation depends on whether we are representing motions w.r.t. the world frame or the base frame (can be configured).
- \( \mathbf{K}_p \) is the diagonal matrix of Cartesian proportional gains.
- \( \mathbf{K}_d \) is the diagonal matrix of Cartesian derivative gains.
- \( - \mathbf{J}\mathbf{\dot{q}} = - \mathcal{V} \) is the twist (linear and angular velocity) of the end-effector. In the controller it acts as a damping term.

### Nullspace control

The nullspace torque term allows us to regulate the joint positions while controlling the end-effector in Cartesian space.
In our implementation, it simply follows a PD control law to drive the joints towards a desired nullspace position:

$$ \boldsymbol{\tau}_\text{nullspace} = \mathbf{N} ( \mathbf{K}_{p,\text{ns}} \mathbf{e}_\text{ns} - \mathbf{K}_{d,\text{ns}} \mathbf{\dot{q}} )$$

where:

- \( \mathbf{N} \) is the nullspace projector.
- \( \mathbf{K}_{p,\text{ns}} \) is the diagonal matrix of nullspace proportional gains.
- \( \mathbf{K}_{d,\text{ns}} \) is the diagonal matrix of nullspace derivative gains.
- \( \mathbf{e}_\text{ns} = \mathbf{q}_\text{ns,desired} - \mathbf{q}_\text{current} \) is the nullspace position error.
- \( \mathbf{\dot{q}} \) is the current joint velocity. In this case, the desired nullspace velocity is assumed to be zero.

The nullspace position can be set with `robot.set_target_joint(...)` when using the Cartesian controller.
It will publish a target joint position which is interpreted as the nullspace target.


## Variable Stiffness

Both the joint and Cartesian controllers support **variable stiffness**, where the proportional gains $\mathbf{K}_p$ can be dynamically adjusted at runtime via a dedicated ROS2 topic. This enables adaptive compliance — for example, lowering stiffness during contact-rich phases and increasing it for precise positioning. The variable stiffness range is bounded by configurable minimum and maximum values.

## Admittance Control

The admittance controller adds a force-reactive layer on top of the Cartesian impedance controller, enabling compliant interaction with the environment using an external force/torque sensor.

The controller has two cascaded loops:

### Inner Loop: Admittance (Virtual Mass-Spring-Damper)

The admittance layer maintains an internal pose state \( \mathbf{x}_\text{inner} \) that evolves as a virtual mass-spring-damper driven by external forces:

$$\mathbf{M}_\text{adm} \ddot{\mathbf{x}} + \mathbf{D}_\text{adm} \dot{\mathbf{x}} + \mathbf{K}_\text{adm} (\mathbf{x}_\text{inner} - \mathbf{x}_\text{desired}) = \mathbf{F}_\text{ext}$$

where:

-  \( \mathbf{M}_\text{adm} \)  is the virtual inertia matrix (\(6 \times 6\) diagonal), controls how quickly the system responds.
-  \( \mathbf{D}_\text{adm} \)  is the virtual damping matrix (\(6 \times 6\) diagonal), controls oscillation suppression.
-  \( \mathbf{K}_\text{adm} \)  is the virtual stiffness matrix (\(6 \times 6\) diagonal), controls how strongly the system returns to \( \mathbf{x}_\text{desired} \).
-  \( \mathbf{F}_\text{ext} \)  is the external wrench from the F/T sensor topic, transformed from the sensor measurement frame to world-aligned frame using Pinocchio's `changeReferenceFrame`.
-  \( \mathbf{x}_\text{desired} \)  is the commanded target pose (from `target_pose` topic).

!!! note
    This requires an **external F/T sensor** (or the robot's built-in external wrench estimation, e.g. as provided by Franka manipulators). The URDF must include a separate frame for the F/T sensor measurement — the controller transforms the measured force from the local sensor frame to the world-aligned Pinocchio frame.

**Integration** uses semi-implicit Euler on the SE(3) manifold at each control cycle:

$$\ddot{\mathbf{x}} = \mathbf{M}_\text{adm}^{-1} \left( \mathbf{F}_\text{ext} - \mathbf{D}_\text{adm} \dot{\mathbf{x}}_\text{inner} - \mathbf{K}_\text{adm} \cdot \text{Error}(\mathbf{x}_\text{inner}, \mathbf{x}_\text{desired}) \right)$$

$$\dot{\mathbf{x}}_\text{inner} \leftarrow \dot{\mathbf{x}}_\text{inner} + \ddot{\mathbf{x}} \cdot \Delta t$$

$$\mathbf{x}_\text{inner} \leftarrow \exp_6(\dot{\mathbf{x}}_\text{inner} \cdot \Delta t) \cdot \mathbf{x}_\text{inner}$$

The pose error \( \text{Error}(\mathbf{x}_\text{inner}, \mathbf{x}_\text{desired}) \) is computed using separate \( \mathbb{R}^3 \) translational and \( SO(3) \) rotational errors (rather than a full \( SE(3) \) logarithmic map) to avoid unnatural screw motions.

### Outer Loop: Impedance

The resulting pose \( \mathbf{x}_\text{inner} \) from the admittance layer is used as the target for an outer Cartesian impedance controller, which computes the required joint torques (identical to the [Cartesian control](#cartesian-control) described above).

## Safety and extras

The actual torque commands sent to the robot are clamped to the allowed torque limits and torque rate limits defined in the config. 
We also add extra terms that can be add/enabled to the controllers so the final torque command is:

$$ \boldsymbol{\tau}_\text{final} = \text{safety}(\boldsymbol{\tau} + \boldsymbol{\tau}_\text{extra}) $$

where $\text{safety}(...)$ clamps the torques to the allowed limits and $\boldsymbol{\tau}_\text{extra}$ can include friction compensation, gravity compensation (if not already included by the robot hardware interface), coriolis compensation, and joint limit avoidance torques...

