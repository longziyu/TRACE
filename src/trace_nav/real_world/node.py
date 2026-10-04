"""ROS 2 controller with explicit arming, stale-input stop, and a watchdog."""
import json
import threading
import time
import numpy as np


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    from ..controller import Controller
    from ..observations import validate
    from ..explanation import AsyncExplainer, HFClaimsBackend, is_current

    class TraceNode(Node):
        def __init__(self):
            super().__init__("trace_controller")
            defaults = {"checkpoint": "", "calibration": "", "device": "cpu", "armed": False,
                        "observation_topic": "/trace/observation", "command_topic": "/trace/cmd_vel",
                        "max_input_age": 0.35, "watchdog_seconds": 0.3, "stop_distance": 0.35,
                        "language_model": "", "language_adapter": ""}
            for key, value in defaults.items():
                self.declare_parameter(key, value)
            self.controller = Controller(self.get_parameter("checkpoint").value,
                                         self.get_parameter("calibration").value,
                                         self.get_parameter("device").value)
            model_path = self.get_parameter("language_model").value
            backend = HFClaimsBackend(model_path, adapter=self.get_parameter("language_adapter").value or None) if model_path else None
            self.explainer = AsyncExplainer(backend)
            self.lock = threading.Lock()
            self.latest = None
            self.last_record = None
            self.last_complete = time.monotonic()
            self.last_submit = 0.0
            self.last_command = np.zeros(2)
            self.last_stamp = None
            self.publisher = self.create_publisher(Twist, self.get_parameter("command_topic").value, 10)
            self.records = self.create_publisher(String, "/trace/decision", 10)
            self.text = self.create_publisher(String, "/trace/explanation", 10)
            self.sensor_group = MutuallyExclusiveCallbackGroup()
            self.inference_group = MutuallyExclusiveCallbackGroup()
            self.watchdog_group = MutuallyExclusiveCallbackGroup()
            self.create_subscription(String, self.get_parameter("observation_topic").value,
                                     self.on_observation, 1, callback_group=self.sensor_group)
            self.create_timer(self.controller.cfg.dt, self.control, callback_group=self.inference_group)
            self.create_timer(0.05, self.watchdog, callback_group=self.watchdog_group)

        def on_observation(self, message):
            try:
                packet = json.loads(message.data)
                packet["frame"] = validate(packet["frame"])
                age = float(packet["sensor_age"])
                stamp = float(packet["stamp"])
                distance = float(packet["min_range"])
                if not np.isfinite([age, stamp, distance]).all() or age < 0 or distance < 0:
                    raise ValueError("Invalid sensor timing or range")
                with self.lock:
                    self.latest = (packet, time.monotonic())
            except (ValueError, KeyError, TypeError) as exc:
                self.get_logger().error(f"Rejected observation: {exc}")

        def publish_command(self, command):
            message = Twist()
            message.linear.x, message.angular.z = float(command[0]), float(command[1])
            self.publisher.publish(message)
            self.last_command = np.asarray(command).copy()

        def watchdog(self):
            with self.lock:
                latest = self.latest
                stale_input = latest is None or time.monotonic()-latest[1]+latest[0]["sensor_age"] > self.get_parameter("max_input_age").value
                stale_control = time.monotonic()-self.last_complete > self.get_parameter("watchdog_seconds").value
                if stale_input or stale_control or not self.get_parameter("armed").value:
                    self.publish_command([0, 0])

        def control(self):
            started = time.monotonic()
            with self.lock:
                latest = self.latest
            if latest is None:
                return
            packet, received = latest
            max_age = self.get_parameter("max_input_age").value
            if started-received+packet["sensor_age"] > max_age:
                return
            if self.last_stamp == packet["stamp"]:
                return
            if self.last_stamp is not None and packet["stamp"] < self.last_stamp:
                self.controller.history.clear()
            self.last_stamp = packet["stamp"]
            try:
                record = self.controller.step(packet["frame"], started)
                desired = np.asarray(record.candidates[record.selected])[0]
                override = None
                if packet["min_range"] < self.get_parameter("stop_distance").value:
                    desired = np.zeros(2); override = "laser_stop"
                if packet["frame"]["route"][2]*self.controller.cfg.goal_scale <= self.controller.cfg.goal_tolerance:
                    desired = np.zeros(2); override = "goal_reached"
                with self.lock:
                    now = time.monotonic()
                    if self.latest is not None and self.latest[0]["min_range"] < self.get_parameter("stop_distance").value:
                        desired = np.zeros(2); override = "laser_stop"
                    if now-started > self.get_parameter("watchdog_seconds").value or now-received+packet["sensor_age"] > max_age:
                        desired = np.zeros(2); override = "deadline_or_stale_input"
                    if not self.get_parameter("armed").value:
                        desired = np.zeros(2); override = "disarmed"
                    self.publish_command(desired)
                    self.last_complete = now
                payload = {"decision": record.dictionary(), "executed_command": desired.tolist(),
                           "override": override, "cycle_ms": 1000*(time.monotonic()-started)}
                message = String(); message.data = json.dumps(payload); self.records.publish(message)
                self.last_record = record
                if override is None and started-self.last_submit >= 1.0:
                    self.explainer.submit(record); self.last_submit = started
                result = self.explainer.poll()
                if result:
                    valid = override is None and is_current(result, record, self.controller,
                                                            packet["frame"]["previous_action"], started)
                    message = String()
                    message.data = json.dumps({"text": result.text, "historical": not valid,
                                               "fallback": result.fallback, "override": override})
                    self.text.publish(message)
            except Exception as exc:
                with self.lock:
                    self.publish_command([0, 0])
                self.get_logger().error(f"Controller stopped: {exc}")

        def destroy_node(self):
            self.publish_command([0, 0])
            self.explainer.close()
            return super().destroy_node()

    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        node = TraceNode(); executor.add_node(node); executor.spin()
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
