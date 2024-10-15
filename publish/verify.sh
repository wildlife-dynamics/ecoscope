#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <tag>"
    exit 1
fi

echo "Verifying all release recipe versions match tag: $1"

for file in ./publish/recipes/release/*.yaml; do
    context_version=$(yq '.context.version' $file)
    if [ "$context_version" == "\"$1\"" ]; then
        echo $file .context.version matches tag "$1"
    else
        echo $file has .context.version $context_version, which does not match tag \"$1\"
        exit 1
    fi
done
