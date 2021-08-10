#!/usr/bin/env python3
import os
import rclpy
import numpy as np

from rclpy.node import Node
from std_msgs.msg import String

from jetson_voice import TTS
from jetson_voice.utils import audio_to_int16
from jetson_voice_ros.msg import Audio


class TTSNode(Node):
    def __init__(self):
        super().__init__('tts', namespace='voice')
        
        # create topics
        self.text_subscriber = self.create_subscription(String, 'tts_text', self.text_listener, 10)
        self.audio_publisher = self.create_publisher(Audio, 'tts_audio', 10)

        # get node parameters
        self.declare_parameter('model', 'fastpitch_hifigan')
        self.model_name = str(self.get_parameter('model').value)
        self.get_logger().info(f'model = {self.model_name}')

        # load the TTS model
        self.tts = TTS(self.model_name)

    def text_listener(self, msg):
        text = msg.data.strip()
        
        if len(text) == 0:
            return
            
        self.get_logger().info(f"running TTS on '{text}'")
        
        samples = self.tts(text)
        samples = audio_to_int16(samples)
        
        # publish message
        msg = Audio()
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.model_name

        msg.info.channels = 1
        msg.info.sample_rate = self.tts.sample_rate
        msg.info.sample_format = str(samples.dtype)
        
        msg.data = samples.tobytes()
        
        self.audio_publisher.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()