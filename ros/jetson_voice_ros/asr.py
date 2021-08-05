#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from jetson_voice_ros.msg import Audio


class AudioInputNode(Node):
    def __init__(self):
        super().__init__('audio_input', namespace='voice')
        
        # create topics
        self.audio_subscriber = self.create_subscription(Audio, 'audio_input', self.audio_listener, 10)
        
    def audio_listener(self, msg):
        self.get_logger().info("ASR recieved audio message")
        
        
def main(args=None):
    rclpy.init(args=args)
    node = ASRNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()