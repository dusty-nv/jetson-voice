#!/usr/bin/env python3
import os
import rclpy
import numpy as np

from rclpy.node import Node

from jetson_voice.utils import AudioInput, audio_to_int16
from jetson_voice_ros.msg import Audio


class AudioInputNode(Node):
    def __init__(self):
        super().__init__('audio_input', namespace='voice')
        
        # create topics
        self.audio_publisher = self.create_publisher(Audio, 'audio_in', 10)
        
        # get node parameters
        self.declare_parameter('device', '')          # input audio device ID or name
        self.declare_parameter('sample_rate', 16000)  # sample rate (in Hz)
        self.declare_parameter('chunk_size', 16000)   # number of samples per buffer
        
        self.device_name = str(self.get_parameter('device').value)
        self.sample_rate = self.get_parameter('sample_rate').value
        self.chunk_size  = self.get_parameter('chunk_size').value
        
        if self.device_name == '':
            raise ValueError("must set the 'device' parameter to either an input audio device ID/name or the path to a .wav file")
        
        self.get_logger().info(f'device = {self.device_name}')
        self.get_logger().info(f'sample_rate = {self.sample_rate}')
        self.get_logger().info(f'chunk_size = {self.chunk_size}')
        
        # check if this is an audio device or a wav file
        file_ext = os.path.splitext(self.device_name)[1].lower()
        
        if file_ext == '.wav' or file_ext == '.wave':
            wav = self.device_name
            mic = ''
        else:
            wav = ''
            mic = self.device_name

        # create audio device
        self.device = AudioInput(wav=wav, mic=mic, sample_rate=self.sample_rate, chunk_size=self.chunk_size)
        self.device.open()
        
        # create a timer to check for audio samples
        self.timer = self.create_timer(self.chunk_size / self.sample_rate * 0.75, self.publish_audio)
        
    def publish_audio(self):
        samples = self.device.next()
        
        if samples is None:  # TODO implement audio device reset
            self.get_logger().warning('no audio samples were returned from the audio device')
            return

        if samples.dtype == np.float32:  # convert to int16 to make the message smaller
            samples = audio_to_int16(samples)

        if samples.dtype != np.int16:  # the other voice nodes expect int16/float32
            raise ValueError(f'audio samples are expected to have datatype int16, but they were {samples.dtype}')
        
        self.get_logger().debug(f'publishing audio samples {samples.shape} dtype={samples.dtype}') # rms={np.sqrt(np.mean(samples**2))}')
        
        # publish message
        msg = Audio()
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.device_name

        msg.info.channels = 1  # AudioInput is set to mono
        msg.info.sample_rate = self.sample_rate
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