#!/usr/bin/env python3

import rclpy
import numpy as np

from rclpy.node import Node

from jetson_voice import AudioInput
from jetson_voice_ros.msg import Audio


class AudioInputNode(Node):
    def __init__(self):
        super().__init__('audio_input', namespace='voice')
        
        # create topics
        self.audio_publisher = self.create_publisher(Audio, 'audio_input', 10)
        
        # get node parameters
        self.declare_parameter('mic', '')  # input audio device ID or name
        self.declare_parameter('wav', '')  # path to wav file
        
        self.declare_parameter('sample_rate', 16000)  # sample rate (in Hz)
        self.declare_parameter('chunk_size', 16000)   # number of samples per buffer
        
        mic = self.get_parameter('mic').value
        wav = self.get_parameter('wav').value
        
        sample_rate = self.get_parameter('sample_rate').value
        chunk_size = self.get_parameter('chunk_size').value
        
        self.device_name = mic if mic != '' else wav
        
        self.get_logger().info(f'device = {self.device_name}')
        self.get_logger().info(f'sample_rate = {sample_rate}')
        self.get_logger().info(f'chunk_size = {chunk_size}')
        
        # create audio device
        self.device = AudioInput(mic=mic, wav=wav, sample_rate=sample_rate, chunk_size=chunk_size)
        self.device.open()
        
        # create a timer to check for audio samples
        self.timer = self.create_timer(chunk_size / sample_rate * 0.75, self.publish_audio)
        
    def publish_audio(self):
        self.get_logger().info('checking for audio samples')
        samples = self.device.next()
        
        if samples is None:  # TODO implement audio device reset
            self.get_logger().warning('no audio samples were returned from the audio device')
            return

        if samples.dtype == np.float32:  # convert to int16 to make the message smaller
            samples = samples.astype(np.int16)

        if samples.dtype != np.int16:  # the other voice nodes expect int16/float32
            raise ValueError(f'audio samples are expected to have datatype int16, but they were {samples.dtype}')
        
        self.get_logger().debug(f'publishing audio samples {samples.shape} dtype={samples.dtype}')
        
        # publish message
        msg = Audio()
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.device_name
        
        msg.info.channels = 1  # AudioInput is set to mono
        msg.info.sample_rate = self.device.sample_rate
        msg.info.sample_format = str(samples.dtype)
        
        msg.data = samples.tobytes()
        
        self.audio_publisher.publish(msg)
        
        
def main(args=None):
    rclpy.init(args=args)
    node = AudioInputNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()