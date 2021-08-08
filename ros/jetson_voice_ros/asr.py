#!/usr/bin/env python3
import os
import rclpy
import numpy as np

from rclpy.node import Node
from std_msgs.msg import String

from jetson_voice import ASR
from jetson_voice_ros.msg import Audio


class ASRNode(Node):
    def __init__(self):
        super().__init__('asr', namespace='voice')
        
        # create topics
        self.audio_subscriber = self.create_subscription(Audio, 'audio_in', self.audio_listener, 10)
        self.transcript_publisher = self.create_publisher(String, 'transcripts', 10)
        self.partial_transcript_publisher = self.create_publisher(String, 'partial_transcripts', 10)
        
        # get node parameters
        self.declare_parameter('model', 'quartznet')
        self.model_name = str(self.get_parameter('model').value)
        self.get_logger().info(f'model = {self.model_name}')

        # load the ASR model
        self.asr = ASR(self.model_name)
        
        if self.asr.classification:
            raise ValueError(f'the ASR node does not support classification models')
        
    def audio_listener(self, msg):
        if msg.info.sample_rate != self.asr.sample_rate:
            self.get_logger().warning(f"audio has sample_rate {msg.info.sample_rate}, but ASR expects sample_rate {self.asr.sample_rate}")
            
        samples = np.frombuffer(msg.data, dtype=msg.info.sample_format)
        self.get_logger().debug(f'recieved audio samples {samples.shape} dtype={samples.dtype}') # rms={np.sqrt(np.mean(samples**2))}')
        
        results = self.asr(samples)
        
        for transcript in results:
            msg = String()
            msg.data = transcript['text']

            self.get_logger().info(f"transcript:  {transcript['text']}")

            if transcript['end']:
                self.transcript_publisher.publish(msg)
                
            self.partial_transcript_publisher.publish(msg)
                

def main(args=None):
    rclpy.init(args=args)
    node = ASRNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()