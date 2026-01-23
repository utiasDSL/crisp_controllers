### Sensors

You can add further sensors (Force Torque Sensor, Tactile Sensor...) by adding a custom `Sensor` that subscribes to a topic.
`Sensor` is simply a class that gives read-only access to the data being published on a topic with a given message type.
This is particularly useful when adding further observation to the manipulator environment directly in the config:
```yaml
sensor_configs:
  - sensor_type: "force_torque"
    shape: [6,]
    # buffer_size: 30
    name: "ft_sensor"
    data_topic: "/external_wrench"
  - sensor_type: "tactile"
    shape: [...,]
    name: "tactile_sensor"
    data_topic: "/tactile_data"
  - ...
```
This sensors will directly be added to the observation space of the manipulator environment.
