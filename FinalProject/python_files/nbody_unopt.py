# Unoptimized nbody
import math
import time
import numpy as np

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



def main():
	# Number of particles
	N = 100
	# Time step
	dt = 0.01
	# Initial start time
	t = 0
	# End time
	tEnd = 5
	G = 6.67 * 10**(-11)

	t0 = time.time()

	vel = np.random.randn(N, 3)
	posit = np.random.randn(N, 3)

	velocities = [0] * N 
	for i in range(N):
		coordinates = [0, 0, 0]
		coordinates[0] = vel[i][0]
		coordinates[1] = vel[i][1]
		coordinates[2] = vel[i][2]
		velocities[i] = coordinates

	positions = [0] * N
	for i in range(N):
		coordinates = [0, 0, 0]
		coordinates[0] = posit[i][0]
		coordinates[1] = posit[i][1]
		coordinates[2] = posit[i][2]
		positions[i] = coordinates

	initPositions = positions

	print("x \t \t y \t \t z")
	print(positions[0][0], positions[0][1], positions[0][2])

	# Initialize mass, all the same
	mass = 20 * np.ones((N, 1))/N

	# Get initial accelerations
	accelerations = getAccels(positions, mass, G, N)

	# Number of steps
	numSteps = int(tEnd/dt)

	for i in range(numSteps):
		# print("i: ", i)
		# First half step 
		for j in range(N):
			velocities[j][0] = velocities[j][0] + accelerations[j][0] * (dt/2.0)
			velocities[j][1] = velocities[j][1] + accelerations[j][1] * (dt/2.0)
			velocities[j][2] = velocities[j][2] + accelerations[j][2] * (dt/2.0)


		# Update the positions, the drift
		for j in range(N):
			positions[j][0] = positions[j][0] + velocities[j][0] * dt
			positions[j][1] = positions[j][1] + velocities[j][1] * dt
			positions[j][2] = positions[j][2] + velocities[j][2] * dt

		# Get new accelerations
		accelerations = getAccels(positions, mass, G, N)

		# Get new velocities with new accelerations
		# Second half step
		for j in range(N):
			velocities[j][0] = velocities[j][0] + accelerations[j][0] * (dt/2)
			velocities[j][1] = velocities[j][1] + accelerations[j][1] * (dt/2)
			velocities[j][2] = velocities[j][2] + accelerations[j][2] * (dt/2)
		velocities += accelerations * (dt/2)

		# Update the time
		t += dt

	print("x \t \t y \t \t z")
	print(positions[0][0], positions[0][1], positions[0][2])
	
	tEnd = time.time()
	print("Total time: ", tEnd - t0)


main()







