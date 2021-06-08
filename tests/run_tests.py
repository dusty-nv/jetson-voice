#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import json
import logging
import argparse
import datetime
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--log-dir', default='', type=str, help='directory to save log files under')
parser.add_argument('--tests', default='data/tests/tests.json', type=str, help='path to config file of tests')
parser.add_argument('--model', default='', type=str, help='if specified, only run tests that use this model')
parser.add_argument('--module', default='', type=str, help='if specified, only run tests that use this module')
parser.add_argument('--config', default='', type=str, help='if specified, only run tests that use this test config')
parser.add_argument('--generate', action='store_true', help='generate the expected outputs')

args = parser.parse_args()

if args.log_dir == '':
    args.log_dir = os.path.join('data/tests/logs', datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

print(args)

# wrapper for launching test processes
def run_test(module, model, config, args=None, log_dir=None):
    config = os.path.join('data/tests', config)
    cmd = f"python3 tests/{module} --model {model} --config {config}"
    
    if args:
        cmd += ' ' + args
       
    print("\nrunning test:\n\t$", cmd, "\n")  

    if log_dir:
        tee = f"tee {os.path.join(log_dir, os.path.splitext(os.path.basename(module))[0])}_{model}.txt"
        cmd = f"mkfifo pipe; {tee} < pipe & {cmd} > pipe; code=$?; rm pipe; exit $code" # https://stackoverflow.com/a/1221844

    results = subprocess.run(cmd, shell=True)
    
    if results.returncode == 0:
        status = 'PASSED'
    elif results.returncode == 127:
        status = 'GENERATED'
    else:
        status = 'FAILED'
        
    print(f"\n{status} TEST {module} ({model}) - return code {results.returncode}\n")
    return status
    
# load the config containing all the tests
with open(args.tests) as config_file:
    test_config = json.load(config_file)

# filter the tests if requested
def filter_test(test):
    if args.model != '' and args.model != test['model']:
        return False
        
    if args.module != '' and args.module != test['module']:
        return False
        
    if args.config != '' and args.config != test['config']:
        return False
        
    return True
        
test_config = [test for test in test_config if filter_test(test)]

# run the tests
for test in test_config:
    test_args = test.get('args', '')
    
    if args.generate:
        test_args += ' --generate'
        
    status = run_test(test['module'], test['model'], test['config'], test_args, args.log_dir)
    
    # if the test needed to generate the expected outputs, run it again
    if status == 'GENERATED':
        print('generated expected outputs, running test again...')
        status = run_test(test['module'], test['model'], test['config'], test.get('args'), args.log_dir)
     
    test['status'] = status

# test summary
passed = 0

print('')
print('----------------------------------------------------')
print(' TEST SUMMARY')
print('----------------------------------------------------')

for test in test_config:
    test_str = f"{test['module']} ({test['model']})"
    print(f"{test_str:<40} {test['status']}")
    
    if test['status'] == 'PASSED':
        passed += 1
        
print(f"\npassed {passed} of {len(test_config)} tests")
print(f"saved logs to {args.log_dir}")
