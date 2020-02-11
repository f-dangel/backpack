"""
Run all example files.
Example files are identified by the pattern 'example_*.py'.
"""
import glob
import os
import subprocess

HERE = os.path.dirname(os.path.realpath(__file__))
PATTERN = os.path.join(HERE, r"example_*.py")
FILES = glob.glob(PATTERN)

for example in FILES:
    print("\nRunning {}".format(example))

    exit_code = subprocess.call(["python", example])
    crash = exit_code != 0

    if crash:
        raise RuntimeError("Error running {}".format(example))
