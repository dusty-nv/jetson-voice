#!/usr/bin/env python3
import os
import rclpy
import numpy as np

from rclpy.node import Node

from jetson_voice import AudioOutput
from jetson_voice_ros.msg import Audio

from soundfile import SoundFile


class AudioOutputNode(Node):
    def __init__(self):
        super().__init__('audio_output', namespace='voice')
        
        # create topics
        self.audio_subscriber = self.create_subscription(Audio, 'audio_out', self.audio_listener, 10)
        
        # get node parameters
        self.declare_parameter('device', '')          # input audio device ID or name
        self.declare_parameter('sample_rate', 16000)  # sample rate (in Hz)
        self.declare_parameter('chunk_size', 4096)    # number of samples per buffer
        
        self.device_name = str(self.get_parameter('device').value)
        self.sample_rate = self.get_parameter('sample_rate').value
        self.chunk_size = self.get_parameter('chunk_size').value
        
        if self.device_name == '':
            raise ValueError("must set the 'device' parameter to either an input audio device ID/name or the path to a .wav file")
        
        self.get_logger().info(f'device={self.device_name}')
        self.get_logger().info(f'sample_rate={self.sample_rate}')
        self.get_logger().info(f'chunk_size={self.chunk_size}')
        
        # check if this is an audio device or a wav file
        file_ext = os.path.splitext(self.device_name)[1].lower()
        
        if file_ext == '.wav' or file_ext == '.wave':
            self.wav = SoundFile(self.device_name, mode='w', samplerate=self.sample_rate, channels=1)
            self.device = None
        else:
            self.wav = None
            self.device = AudioOutput(self.device_name, sample_rate=self.sample_rate, chunk_size=self.chunk_size)

    def audio_listener(self, msg):
        #self.get_logger().debug('recieved new audio message')
        #self.get_logger().debug(f'{msg.header}')
        #self.get_logger().debug(f'{msg.info}')
        
        if msg.info.sample_rate != self.sample_rate:
            self.get_logger().warning(f"audio has sample_rate {msg.info.sample_rate}, but audio device is using sample_rate {self.sample_rate}")
            
        samples = np.frombuffer(msg.data, dtype=msg.info.sample_format)
        
        self.get_logger().debug(f'recieved audio samples {samples.shape} dtype={samples.dtype}') # rms={np.sqrt(np.mean(samples**2))}')
        
        if self.device is not None:
            self.device.write(samples)
        else:
            self.wav.write(samples)


def main(args=None):
    rclpy.init(args=args)
    node = AudioOutputNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()