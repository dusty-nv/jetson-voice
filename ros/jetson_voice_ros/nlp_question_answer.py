#!/usr/bin/env python3
import os
import rclpy

from rclpy.node import Node
from std_msgs.msg import String

from jetson_voice import QuestionAnswer as QuestionAnswerFactory
from jetson_voice_ros.msg import QuestionAnswerQuery, QuestionAnswerResult


class NLPQuestionAnswerNode(Node):
    def __init__(self):
        super().__init__('nlp_question_answer', namespace='voice')
        
        # create topics
        self.query_subscriber = self.create_subscription(QuestionAnswerQuery, 'question_answer_query', self.query_listener, 10)
        self.result_publisher = self.create_publisher(QuestionAnswerResult, 'question_answer_results', 10)

        # get node parameters
        self.declare_parameter('model', 'distilbert_qa_384')
        self.model_name = str(self.get_parameter('model').value)
        self.get_logger().info(f'model = {self.model_name}')

        # load the QA model
        self.model = QuestionAnswerFactory(self.model_name)
        self.get_logger().info(f"model '{self.model_name}' ready")
        
    def query_listener(self, msg):
        question = msg.question.data.strip()
        context = msg.context.data.strip()

        if len(question) == 0 or len(context) == 0:
            return
            
        self.get_logger().info(f"running NLP Question/Answer query:")
        self.get_logger().info(f"question:  '{question}'")
        self.get_logger().info(f"context:")
        self.get_logger().info(context)
        
        # run the model
        results = self.model((question,context))
        
        self.get_logger().info(f"answer: '{results['answer']}'")
        self.get_logger().info(f"score:  {results['score']}")

        # create message
        msg = QuestionAnswerResult()
        
        msg.question.data = question
        msg.answer.data = results['answer']
        msg.score = results['score']
        
        # publish message
        self.result_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = NLPQuestionAnswerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()