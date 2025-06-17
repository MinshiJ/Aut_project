# Master Thesis: Evaluation of Tolerable Ranges in High-Speed Assembly Tasks in Dynamic Simulations of Compliant Mechanisms


## Set up
Donwload MuJoCo from the offical website. Set up a python environment with the python version 3.10.14. Download the packages from the requirements.txt with the specified versions. Please note that the installed version of the package and software from MuJoCo need to be the exact same. If necessary, the mujoco/bin directory has to be added to the path of the python environment.


## Usage

This MuJoCo Simulation is created to simulate compliant finray-grippers in different assembly tasks with varying starting positions. This way, the tolerable ranges of the compliance before an assembly fails can be evaluated for different design parameters. 

## Repository Layout

The sim.py contains the simulation control. The robot_15.xml and scene_15.xml contain the model for the straight gripper design. The robot_chamf.xml and scene_chamf.xml contain the model for the chamfered gripper design. 
The asset directory contains all model assets for the simulation. It needs to be in the same directory as the other files. 
The requirements.txt shows the packages installed for a working environment in anaconda.navigator and spyder.










