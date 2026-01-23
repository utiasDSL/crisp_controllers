

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


## Safety and extras

The actual torque commands sent to the robot are clamped to the allowed torque limits and torque rate limits defined in the config. 
We also add extra terms that can be add/enabled to the controllers so the final torque command is:

$$ \boldsymbol{\tau}_\text{final} = \text{safety}(\boldsymbol{\tau} + \boldsymbol{\tau}_\text{extra}) $$

where $\text{safety}(...)$ clamps the torques to the allowed limits and $\boldsymbol{\tau}_\text{extra}$ can include friction compensation, gravity compensation (if not already included by the robot hardware interface), coriolis compensation, and joint limit avoidance torques...

