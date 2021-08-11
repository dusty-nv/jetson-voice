#!/usr/bin/env python3
import os
import rclpy

from rclpy.node import Node
from std_msgs.msg import String

from jetson_voice import IntentSlot as IntentSlotFactory
from jetson_voice_ros.msg import IntentSlot, Slot


class NLPIntentSlotNode(Node):
    def __init__(self):
        super().__init__('nlp_intent_slot', namespace='voice')
        
        # create topics
        self.query_subscriber = self.create_subscription(String, 'intent_slot_query', self.query_listener, 10)
        self.result_publisher = self.create_publisher(IntentSlot, 'intent_slot_results', 10)

        # get node parameters
        self.declare_parameter('model', 'distilbert_intent')
        self.model_name = str(self.get_parameter('model').value)
        self.get_logger().info(f'model = {self.model_name}')

        # load the IntentSlot model
        self.model = IntentSlotFactory(self.model_name)
        self.get_logger().info(f"model '{self.model_name}' ready")
        
    def query_listener(self, msg):
        text = msg.data.strip()
        
        if len(text) == 0:
            return
            
        self.get_logger().info(f"running NLP Intent/Slot query:  '{text}'")
        
        # run the model
        results = self.model(text)
        
        self.get_logger().info(f"intent: '{results['intent']}'")
        self.get_logger().info(f"score:  {results['score']}")
        
        for slot in results['slots']:
            self.get_logger().info(str(slot))

        # create message
        msg = IntentSlot()
        
        msg.query.data = text
        msg.intent.data = results['intent']
        msg.score = float(results['score'])
        
        slots = []
        
        for slot in results['slots']:
            slot_msg = Slot()
            
            slot_msg.slot.data = slot['slot']
            slot_msg.text.data = slot['text']
            slot_msg.score = float(slot['score'])
            
            slots.append(slot_msg)
            
        msg.slots = tuple(slots)
        
        # publish message
        self.result_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = NLPIntentSlotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()