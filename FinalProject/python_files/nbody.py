import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt 


# Step 2: Calculate the acceleration for each coordinate of each particle
def getAccels(positions, mass, G, N):

	accelerations = np.zeros((N, 3))
	# Go through the entire list of particles
	for i in range(N):
		# The ith particle's new acceleration
		ax = 0
		ay = 0
		az = 0
		# Need to know the force of each particle on current particle?
		# Sum these forces to get the acces
		for j in range(N):

			if j != i:
				# x coordinate in 0th position
				dx = positions[j][0] - positions[i][0]
				# y coordinate in 1st position
				dy = positions[j][1] - positions[i][1]
				# z coordinate in 2nd position
				dz = positions[j][2] - positions[i][2]
				# Maybe use softening for cases where particles are 
				# too close together
				# Calculate inverse thing
				magnitude3 = (math.sqrt((dx**2 + dy**2 + dz**2 + 0.1**2)))**3

				ax += mass[j] * (dx/magnitude3)
				ay += mass[j] * (dy/magnitude3)
				az += mass[j] * (dz/magnitude3)

		# Multiply with gravitational constant G
		ax = ax * G 
		ay = ay * G 
		az = az * G 
		accelerations[i][0] = ax
		accelerations[i][1] = ay
		accelerations[i][2] = az

	return accelerations



# Step 3: Move the particle, which means update velocities and positions
# Use a leapfrog integration, "kick-drift-kick"
# We want to update the position of the particle 
# velocity at i+1/2 = v at i-1/2 + a*dt -> 
# each particle receives a half step "kick" for each dt
# then full-step drift ->
# position is updated as position at i + dt*vi
# Finish with another half-step kick
# a second order system and is able to preserve total energy of the system
# Use because acceleration doesn't depend on velocity 


def main():
	# Number of particles
	N = 75
	# Time step
	dt = 0.01
	# Initial start time
	t = 0
	# End time
	tEnd = 10
	G = 6.67 * 10**(-11)

	t0 = time.time()

	velocities = np.random.randn(N, 3)
	positions = np.random.randn(N, 3)
	initPositions = positions

	# Initialize mass, all the same
	mass = 25 * np.ones((N, 1))/N
	mass[0] = 10

	# Get initial accelerations
	accelerations = getAccels(positions, mass, G, N)

	# Number of steps
	numSteps = int(tEnd/dt)

	fig = plt.figure(figsize=(4, 5), dpi = 80)
	grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
	ax1 = plt.subplot(grid[0:2,0])
	ax2 = plt.subplot(grid[2,0])

	for i in range(numSteps):
		# print("i: ", i)
		# First half step 
		velocities += accelerations * (dt/2.0)

		# Update the positions, the drift
		positions += velocities*dt 

		# Get new accelerations
		accelerations = getAccels(positions, mass, G, N)

		# Get new velocities with new accelerations
		# Second half step
		velocities += accelerations * (dt/2)

		

		plt.sca(ax1)
		plt.cla()
		# xx = pos
		plt.scatter(positions[0, 0], positions[0, 1], s=10, color = 'orange')
		plt.scatter(positions[1:,0], positions[1:,1], s=10, color='blue')
		ax1.set(xlim=(-10, 10), ylim=(-10, 10))
		ax1.set_aspect('equal', 'box')
		ax1.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
		ax1.set_yticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])

		plt.pause(0.001)


		# Update the time
		t += dt


	tEnd = time.time()
	print("Total time: ", tEnd - t0)


main()







