#!/usr/bin/env python3
# coding: utf-8

from jetson_voice import config, ConfigArgParser

parser = ConfigArgParser(description='test')
args = parser.parse_args()

print('in-module config', config)
