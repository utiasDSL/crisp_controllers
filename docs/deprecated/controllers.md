# Available controllers

## Default controllers

In [CRISP_PY](https://github.com/utiasDSL/crisp_py/tree/main/config/control) you can find some of the default configurations that we used with the FR3 of Franka Robotics.
These configurations are:

- `default_cartesian_impedance.yaml`: A simple config for a Cartesian Impedance (CI) Controller. This is our go-to controller, since it offers nice contact behavior and is the perfect balance between compliance and precision.
- `default_operational_space_controller.yaml`: A simple config for Operational Space Controller (OSC). Is a slightly stiffer controller. The contact behavior might not be as nice as for CI.
- `clipped_cartesian_impedance.yaml`: A simple config of a highly clipped error CI (CI-clipped). This offers a highly stiff controller but at the same time safe since the error is highly clipped. Useful for precise manipulation.

_Experiments using these sets of parameters can be found in our paper._

