# Get started with `crisp_py`


!!! Note
    If you want to use the gymnasium interface, CRISP_PY will be automatically installed in the gym. You can therefore check the installation of [CRISP_GYM](#4-getting-started-with-crisp_gym) directly.
    However, this section still gives you an idea on how to use CRISP_PY with your robot. We do not recommend to skip it.

!!! Warning
    `CRISP_PY` has been tested with `humble` and `jazzy` using python version `3.11`. 
    You might encounter (solvable) problems using `3.12` and above, please check this [scipy issue](https://github.com/utiasDSL/crisp_py/pull/63#discussion_r2703751488).

To use `CRISP_PY`, we recommend using [pixi](https://pixi.sh/latest/), a modern conda-like package manager.
It can be used in combination with [robostack](https://robostack.github.io/) to easily install ROS2 in any machine.
There are a few ways to get you started:

_... use in your already existing pixi project:_

To use `CRISP_PY` in an already existing pixi project, you need to make sure that `ROS2` is available.
Check the [pixi.toml](https://github.com/utiasDSL/crisp_py/blob/main/pixi.toml) of `CRISP_PY` to see how this looks like.
Then you can add `CRISP_PY` as a pypi package:
```bash
pixi add --pypi crisp-python
```
or
```bash
uv add crisp-python
```
or
```bash
pip install crisp-python
```
Double-check that everything is working by running:

```bash
python -c "import crisp_py"  # (1)!
```

1. This should not log anything if everything is fine

_... install from source:_

```bash
git clone https://github.com/utiasDSL/crisp_py
cd crisp_py
pixi install
pixi shell -e humble
python -c "import crisp_py"  # (1)!
```

1. This should not log anything if everything is fine

Now you can try to control the robot! Check out the [examples](https://github.com/utiasDSL/crisp_py/blob/main/examples) for inspiration.
Or check the examples sections in the documentation [for the robot](examples_robot.md) or for other components such as [camera](examples_camera.md).

