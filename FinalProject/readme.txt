Files
Parallel_Final_Project_Paper.pdf
- Paper

nbody.cu
- Contains the N-body simulation implementation in C++ and CUDA

CMakeLists.txt
- Contains info to use cmake

nbody.py
- The Python implementation of the N-body simulation

all_positions.txt 1-4
- The saved positions for different scenarios
1. All masses are 0.2
2. Two masses are 5, rest are 0.01 * i
3. All masses are different: 0.2*i

simulation.py
- Creates the visualization of the particles and their motions
- Currently set to run using all_positions_1.txt

graph.py
- Creates a graph of the runtime comparisons of Python vs. C++/CUDA
- Graph is already created: time_comp.py

nbody_unopt.py
- Doesn't use numpy arrays
- Not important to the final product, not used


Variables are currently set:
N = 100
tEnd = 10


To run nbody.cu
Using cMake to compile nbody.cu
create build directory or use existing build directory and run make
cd into build directory
run cmake ..
run make
Run program using ./nbody


To run nbody.py
run python3 nbody.py


To run the simulation
Run simulation.py 
This will run the simulation with saved positions from when all masses are 0.2


Based off of Python version by Philip Mocz
https://medium.com/swlh/create-your-own-n-body-simulation-with-python-f417234885e9
https://github.com/pmocz/nbody-python

