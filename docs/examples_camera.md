# Camera Examples

### Cameras

The cameras that we tested are:

- Any usb camera or webcam using the [ROS2 usb-cam](https://github.com/ros-drivers/usb_cam) package (see [a pixi wrapper here](https://github.com/danielsanjosepro/pixi_usbcam_ros2)),
- [Real Sense](https://github.com/IntelRealSense/realsense-ros/tree/ros2-master) which gives amazing ROS2 support,
- and [Orbbec](https://github.com/orbbec/OrbbecSDK_ROS2).

Check the [getting ros2 side ready guide](getting_started_controllers.md) to see some examples with cameras

## Run a camera node

We first start a camera node that publishes images to ROS2 topics.
Check the [getting ros2 side ready guide](getting_started_controllers.md) to see other examples with cameras.
If you use an usb camera / webcam, you can try [pixi_usbcam_ros2](https://github.com/danielsanjosepro/pixi_usbcam_ros2) and start it with:
```bash
pixi run -e jazzy usb_cam
```

## Access images from `crisp_py`

Now in the broadcasted images can be accessed from `crisp_py`.
Using the Camera class from `crisp_py`:
```python
"""Simple example for a camera. It shows the camera feed in a matplotlib window."""

import matplotlib.pyplot as plt
from crisp_py.camera import CameraConfig, Camera

camera_config = CameraConfig(
    camera_name="primary",
    camera_frame="primary_link",
    resolution=[256, 256],
    camera_color_image_topic="/image_raw",
    camera_color_info_topic="/image_raw/camera_info",
)

camera = Camera(config=camera_config, namespace="")
camera.wait_until_ready()

# Display camera feed
plt.ion()
fig, ax = plt.subplots()
ax.axis("off")

frame = camera.current_image
im = ax.imshow(frame)
while True:
    im.set_data(camera.current_image)
    plt.pause(1.0 / 30.0)
```

Or by defining a camera in a YAML configuration file:
```yaml
camera_name: "primary"
camera_frame: "primary_link"
resolution: [256, 256]
camera_color_image_topic: "/image_raw"
camera_color_info_topic: "/image_raw/camera_info"
```

Then load the configuration and use the `make_camera` factory method (assuming that you added the config path to the `CRISP_CONFIG_PATH` environment variable as described in the [configuration guide](getting_started_config.md)):

```python
"""Example showing how to load camera configuration from a YAML file."""
import cv2
from crisp_py.camera import make_camera

camera = make_camera("your_config_file_name")
camera.wait_until_ready()

# Display one frame
plt.imshow(camera.current_image)
plt.axis("off")
plt.show()
```


