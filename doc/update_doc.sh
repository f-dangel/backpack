#!/bin/bash

# Call sphinx-apidoc to refresh the files
sphinx-apidoc -f -o source/ \
                    ../bpexts/ \
                    ../bpexts/*_test.py \
                    ../bpexts/*/*_test.py \
                    ../bpexts/*/*/*_test.py \
                    ../bpexts/*/*/*/*_test.py
