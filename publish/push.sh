#!/bin/bash

for file in /tmp/ecoscope/release/artifacts/**/*.conda; do
    rattler-build upload prefix -c ecoscope-workflows "$file" || true
done
