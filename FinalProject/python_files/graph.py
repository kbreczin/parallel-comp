import matplotlib.pyplot as plt 

cuda = [252371, 466737, 815599, 1149932, 1532900]
python = [1167188.883, 7302161.932, 30958075.05, 68233203.89, 124217505.9]

yaxis = [10, 25, 50, 75, 100]

plt.plot(cuda, yaxis, marker = 'o', label='C++/CUDA')
plt.plot(python, yaxis, marker = 'o', label='Python')
plt.legend()
plt.title('Time to Run N-Body Simulation: Python vs. C++/CUDA')
plt.xlabel('Time in Microseconds')
plt.ylabel('Number of Particles')

plt.show()
