#!/bin/sh
sudo apt-get install python3-opencv python3-matplotlib -y
sudo -H pip3 install scikit-learn pandas -y

pip3 install --user -r requirements.txt
