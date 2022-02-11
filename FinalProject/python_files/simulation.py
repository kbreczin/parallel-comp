import matplotlib.pyplot as plt
import numpy as np
import csv

N = 100
numSteps = 10/0.01
numSteps = int(numSteps)
all_data = np.zeros((N, numSteps, 3))
# positions = np.array
i = 0
j = 0

# Change file to test different positions
with open('all_positions_1.txt') as csvfile:
	read = csv.reader(csvfile, delimiter = ',')
	for row in read:
		# coord = [0]*3
		if 'Particle' in row[0] and '1 ' not in row[0]:
			i+=1
			j = 0
		elif 'Particle' not in row[0]:
			all_data[i][j][0] = float(row[0])
			all_data[i][j][1] = float(row[1])
			all_data[i][j][2] = float(row[2])
			j += 1

# all_data.append(positions)
# all_data = np.array(all_data)

csvfile.close()

fig = plt.figure(figsize=(4, 5), dpi = 80)
grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
ax1 = plt.subplot(grid[0:2,0])
# ax2 = plt.subplot(grid[2,0])

# needs to be an numpy array for multi-dimensional slicing

np.random.seed(17)
saved_positions = np.random.randn(10, 10, 3)
all_data = np.array(all_data)
# print(all_data)
'''
Array format: [ [[x, y], [x, y], [x, y]], [[x, y], [x, y], [x, y]], [[x, y], [x, y], [x, y]]]
'''
# :,i, 0
# [:,i, 1]

for i in range(numSteps):
	plt.sca(ax1)
	plt.cla()
	plt.scatter(all_data[:, i, 0], all_data[:, i, 1], s=10, color='blue')
	ax1.set(xlim=(-10, 10), ylim=(-10, 10))
	ax1.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
	ax1.set_yticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])

	plt.pause(0.001)

plt.xlabel('time')
plt.ylabel('energy')

plt.show()

