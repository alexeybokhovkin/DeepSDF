#!/usr/bin/env bash

mkdir -p data/
cd data/

# Downloading data
mkdir -p raw_data/
wget -P raw_data/ http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip