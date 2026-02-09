#!/bin/bash

mkdir -p data/ld/EUR/

wget -O data/ld/EUR.tar.gz https://github.com/shz9/viprs/releases/download/v0.1.2/EUR.tar.gz
tar -xzf data/ld/EUR.tar.gz -C data/ld/EUR

rm -rf data/ld/EUR.tar.gz
