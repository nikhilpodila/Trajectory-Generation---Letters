# Trajectory generation - Letters
    Author: Nikhil Podila
	
## Code test environment:

The code was tested in Python (version 3.7.4) on Windows 10, installed using Anaconda distribution. 

The following are lines from the "requirements.txt" file of the project. <br>
It contains all the required dependencies/libraries/packages with the format: [package_name]==[version_number]
```
matplotlib==3.1.1
numpy==1.16.5
pybullet==2.1.3
scipy==1.3.1
scikit_learn==0.23.2
```

## Getting started

This code is used to draw three letters - "S", "L" and "V" using a Kuka Iiwa robot on the y-z plane at x = -0.4 of the simulation.

Results and simulation can be viewed by running either of the following commands on the command prompt or terminal in this directory:
```
python experiment.py S
```
(This displays letter "S")
or 
```
python experiment.py L
```
(This displays letter "L")
or
```
python experiment.py V
```
(This displays letter "V")
