# FPGA-Accelerated Neural Network for Deltabot Control

Fast neural network inference on FPGA for real-time control of a non-linear deltabot system at Diamond Light Source.

## Project Overview

### The Problem
At Diamond Light Source, we operate a deltabot (3-axis delta robot) for precision positioning:
- **Hardware**: Voice coil actuators with flexures
- **Controller**: PandABox (Zynq-7000 XC7Z030 FPGA)
- **Challenge**: Thermal drift in voice coils creates significant non-linearity
- **Requirements**: 
  - High-speed control loop (125 MHz FPGA clock)
  - Accurate position control despite non-linear dynamics

PandABox website: https://ohwr.org/projects/pandabox/

### The Solution
Deploy a neural network directly on FPGA hardware to learn and compensate for the non-linear voltage-to-position mapping. 

The neural network is converted into hardware logic and packaged as an AXI-compatible IP block that integrates directly into the PandABox FPGA fabric
