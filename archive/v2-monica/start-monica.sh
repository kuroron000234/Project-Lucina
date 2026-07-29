#!/bin/bash
# 1. \u53e4\u3044\u30b3\u30f3\u30c6\u30ca\u3092\u524a\u9664
docker rm -f koushi-desktop

# 2. \u30ed\u30b0\u30a4\u30f3\u60c5\u5831\u3092\u4fdd\u5b58\u3059\u308b\u30d5\u30a9\u30eb\u30c0\u3092Cyborg\u5074\u306b\u4f5c\u6210
mkdir -p ~/Monica/desktop-data

# 3. \u30c7\u30fc\u30bf\u3092\u4fdd\u5b58\u3059\u308b\u3088\u3046\u306b\u6307\u5b9a\u3057\u3066\u8d77\u52d5
docker run -d \
  --name koushi-desktop \
  --restart always \
  -p 6080:80 \
  --shm-size=2g \
  -e VNC_PASSWORD=koushi \
  -v ~/Monica/desktop-data:/home/ubuntu \
  dorowu/ubuntu-desktop-lxde-vnc

echo "\u8d77\u52d5\u5b8c\u4e86\uff01\u3053\u308c\u3067\u30ed\u30b0\u30a4\u30f3\u60c5\u5831\u3082\u4fdd\u5b58\u3055\u308c\u308b\u308f\u3088\u3002"