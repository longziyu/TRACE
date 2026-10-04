"""Minimal ROS 2 RGB/LiDAR/odometry bridge with externally tracked pedestrians.

Goals and pedestrian poses must already be transformed into the odometry frame.
This module does not implement SLAM, object detection, tracking, or TF conversion.
PoseArray ordering is not an identity contract, so pedestrian velocities are
left zero here; use a tracked-object adapter to supply measured velocities.
"""
import json
import math
import threading
import numpy as np


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped, PoseArray, Twist
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    import cv2

    class SensorBridge(Node):
        def __init__(self):
            super().__init__("trace_sensor_bridge")
            for key, value in {"rgb_topic": "/camera/color/image_raw", "scan_topic": "/scan",
                               "odom_topic": "/odom", "goal_topic": "/trace/goal",
                               "people_topic": "/trace/people", "frame_id": "odom",
                               "sensor_max_age": 0.3, "require_people": True,
                               "scan_range": 4.0, "goal_scale": 20.0,
                               "executed_command_topic": "/trace/cmd_vel"}.items():
                self.declare_parameter(key, value)
            self.cache = {}
            self.lock = threading.Lock()
            self.bridge = CvBridge()
            self.last_command = np.zeros(2, dtype=np.float32)
            self.create_subscription(Twist, self.get_parameter("executed_command_topic").value,
                                     self.receive_command, 10)
            self.output = self.create_publisher(String, "/trace/observation", 1)
            for kind, topic, message in [("rgb", "rgb_topic", Image), ("scan", "scan_topic", LaserScan),
                                         ("odom", "odom_topic", Odometry), ("goal", "goal_topic", PoseStamped),
                                         ("people", "people_topic", PoseArray)]:
                self.create_subscription(message, self.get_parameter(topic).value,
                                         lambda msg, k=kind: self.receive(k, msg), qos_profile_sensor_data)
            self.create_timer(0.1, self.publish_observation)

        def receive(self, kind, msg):
            with self.lock:
                self.cache[kind] = msg

        def receive_command(self, msg):
            with self.lock:
                self.last_command = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)

        def publish_observation(self):
            with self.lock:
                data = dict(self.cache)
                previous_command = self.last_command.copy()
            required = {"rgb", "scan", "odom", "goal"}
            if self.get_parameter("require_people").value:
                required.add("people")
            if not required <= data.keys():
                return
            now = self.get_clock().now().nanoseconds/1e9
            sensor_keys = required-{"goal"}
            stamps = [data[k].header.stamp.sec+data[k].header.stamp.nanosec/1e9 for k in sensor_keys]
            age = now-min(stamps)
            if age < 0 or age > self.get_parameter("sensor_max_age").value:
                return
            frame_id = self.get_parameter("frame_id").value
            if any(data[k].header.frame_id != frame_id for k in ["odom", "goal"] + (["people"] if "people" in data else [])):
                self.get_logger().error("Goal, odometry, and pedestrian poses must share frame_id")
                return
            try:
                odom = data["odom"]
                pose = odom.pose.pose
                q = pose.orientation
                theta = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                origin = np.array([pose.position.x, pose.position.y])
                rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                goal_pose = data["goal"].pose.position
                goal = (np.array([goal_pose.x, goal_pose.y])-origin) @ rotation
                image = self.bridge.imgmsg_to_cv2(data["rgb"], desired_encoding="rgb8")
                image = cv2.resize(image, (64, 64)).astype(np.float32).transpose(2, 0, 1)/255
                scan = data["scan"]
                ranges = np.asarray(scan.ranges, dtype=float)
                if not len(ranges) or np.isnan(ranges).all():
                    return
                radius = self.get_parameter("scan_range").value
                good = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
                # Positive infinity represents no return; NaNs and negative values are unknown.
                known = good | np.isposinf(ranges)
                if known.mean() < 0.9:
                    return
                angles = scan.angle_min+np.arange(len(ranges))*scan.angle_increment
                lidar = np.ones(16)
                bev = np.zeros((2, 64, 64), dtype=np.float32)
                for distance, angle in zip(ranges[good], angles[good]):
                    sector = int(((angle+np.pi) % (2*np.pi))/(2*np.pi)*16) % 16
                    lidar[sector] = min(lidar[sector], distance/radius)
                    if distance <= radius:
                        pixel = np.clip((np.array([np.cos(angle), np.sin(angle)])*distance/radius*31+32).astype(int), 0, 63)
                        bev[0, pixel[1], pixel[0]] = 1
                people = np.zeros((8, 5), dtype=np.float32)
                if "people" in data:
                    positions = [(np.array([p.position.x, p.position.y])-origin) @ rotation for p in data["people"].poses]
                    positions = sorted((p for p in positions if np.linalg.norm(p) <= radius), key=np.linalg.norm)[:8]
                    for i, p in enumerate(positions):
                        people[i] = [p[0]/radius, p[1]/radius, 0, 0, 1]
                        pixel = np.clip((p/radius*31+32).astype(int), 0, 63)
                        bev[1, pixel[1], pixel[0]] = 1
                twist = odom.twist.twist
                action = [twist.linear.x, twist.angular.z]
                scale = self.get_parameter("goal_scale").value
                observation = {"rgb": image.tolist(), "bev": bev.tolist(),
                               "route": [*list(goal/scale), float(np.linalg.norm(goal)/scale), math.atan2(goal[1], goal[0])/np.pi],
                               "obstacles": lidar.tolist(), "pedestrians": people.flatten().tolist(),
                               "robot": [*action, np.sin(theta), np.cos(theta)], "previous_action": previous_command.tolist()}
                packet = {"frame": observation, "stamp": min(stamps), "sensor_age": age,
                          "min_range": float(ranges[good].min()) if good.any() else float(scan.range_max)}
                msg = String(); msg.data = json.dumps(packet); self.output.publish(msg)
            except Exception as exc:
                self.get_logger().error(f"Observation conversion failed: {exc}")

    rclpy.init(args=args)
    node = SensorBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
