#!/bin/bash
TARGET="forknetv5"
TAG="smerkd/$TARGET"

# deploys the docker container
echo "Building..."
docker built -t "$TARGET" .

echo "Tagging..."
docker tag "$TARGET:latest" "$TAG:latest"

echo "Uploading..."
# two options second is better for slower internet as we compress layers beforehand into a single zip
# OPTION 1
# docker push "$TAG:latest"  # UNCOMMENT WHEN USING OPTION 1

# OPTION 2
docker save "registry.hub.docker.com/$TAG:latest"| gzip > "$TARGET\_latest.tar.gz"  # UNCOMMENT WHEN USING OPTION 2
push-docker-image "$TARGET\_latest.tar.gz"  # UNCOMMENT WHEN USING OPTION 2 (see: https://github.com/holidaycheck/push-docker-image)
