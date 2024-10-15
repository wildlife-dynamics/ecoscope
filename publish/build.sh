#!/bin/bash
echo "Building recipe: release/ecoscope"

mkdir -p /tmp/ecoscope/release/artifacts

rattler-build build \
--recipe $(pwd)/publish/recipes/release/ecoscope.yaml \
--output-dir /tmp/ecoscope/release/artifacts \
--channel https://prefix.dev/ecoscope-workflows \
--channel conda-forge
    