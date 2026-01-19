# Yer A Wizard - Raspberry Pi Wand Tracker

A computer vision project for Raspberry Pi that tracks a retro-reflective wand and uses a machine learning model to recognize gestures and trigger actions.

# Prerequisites & Hardware Requirements

Raspberry Pi: Pi 4 or Pi 5 recommended.

Operating System: Raspberry Pi OS (64-bit) Bookworm (Debian 12).

Note: 64-bit is mandatory for TensorFlow 2.x support on Python 3.11.

Camera: Raspberry Pi V2.1 NoIR Camera Module.

Wand: Retro-reflective tip (IR-reflective) with an IR light ring around the camera.

# System Setup

Before installing the Python environment, update your system and install hardware-specific libraries for the camera and math acceleration.

## Update and Upgrade system
sudo apt update && sudo apt upgrade -y

## Install system dependencies for OpenCV and Libcamera
sudo apt install python3-opencv python3-libcamera libatlas-base-dev libopenblas-dev -y

## (Optional) Enable SSH and VNC
sudo raspi-config


# Python Environment Setup

We use a virtual environment with --system-site-packages to bridge the gap between Python and the Pi's hardware-accelerated drivers.

## Navigate to project folder
cd ~/.../yerawizard

## Create the environment
python3 -m venv --system-site-packages venv
source venv/bin/activate

## Install Machine Learning libraries
pip install --upgrade pip setuptools
pip install -r requirements.txt


# Running the Tracker

Ensure your camera is connected and recognized (libcamera-hello --list-cameras).

source venv/bin/activate
python3 wand_tracker.py


# Troubleshooting Tips

ImportError: libopenblas.so.0: This occurs if system math libraries are missing. Run sudo apt install libopenblas-dev.

ModuleNotFoundError: 'imp': This usually means you are on Python 3.13 (Trixie). Ensure you are on Bookworm (Python 3.11) or install setuptools.

TensorFlow Version: If using an .h5 model, ensure you are on a 64-bit OS. 32-bit (armv7l) does not support modern TensorFlow 2.13+ wheels.

# Project Structure

wand_tracker.py: Main execution script for CV and gesture recognition. Also used for recording new spells and changing the affect of each spell.
nn_trainer.py: Trains the neural network

venv/: Virtual environment (excluded from Git).
