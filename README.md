# Halimeda
Halimeda coverage estimation

This repository aims to provide the necessary tools and knowledge to operate our halimeda coverage estimation algorithm.

# Installation

To clone the repository do:

git clone --recursive https://github.com/srv/Halimeda.git
git submodule update --init --recursive
git submodule foreach -q --recursive 'branch="$(git config -f $toplevel/.gitmodules submodule.$name.branch)"; git checkout $branch'

or:
git clone https://github.com/srv/Halimeda.git
cd object/yolov5
git submodule init
git submodule update
cd ..
cd SS_Halimeda
git submodule init
git submodule update

# Object detection enviroment installation

cd object_detection

conda create -n <environment-name> --file object_req.txt

## Download object detection model:

https://zenodo.org/record/7611869#.Y_xsFSbMJD8


