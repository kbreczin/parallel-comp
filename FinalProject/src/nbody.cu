#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
// #include "utils.h"
#include <iostream>
#include <math.h>
#include <chrono>
#include <fstream>
#include <iostream>

#define N_THREADS 100
#define N_BLOCKS 1


__global__ void init_rand_kernel(curandState *state) {
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
 curand_init(0, idx, 0, &state[idx]);
}


__global__ void initialize_kernel(int N, float* positions, float* velocities, float* masses, curandState *state, float* savedPositions, int numSteps){
  // Initializes the velocities and positions of each particle

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // randNeg is used to determine whether to multiply by positive or negative 2
  for(int i=0; i < N; i++){
    // Initialize positions and velocities between -2 and 2
    float randX = curand_uniform(&state[tid]);
    float randVal = curand_uniform(&state[tid]);
    int randNeg = (int)(randVal * 10);
    int scale = 10;
    if(randNeg > 5){
      scale = -10;
    }
    positions[0 + 3*i] = scale * randX;
    savedPositions[(i * 3 * numSteps) + (0 * 3) + 0] = scale * randX;

    float randY = curand_uniform(&state[tid]);
    randVal = curand_uniform(&state[tid]);
    randNeg = (int)(randVal * 10);
    scale = 10;
    if(randNeg > 5){
      scale = -10;
    }
    positions[1 + 3*i] = scale * randY;
    savedPositions[(i * 3 * numSteps) + (0 * 3) + 1] = scale * randY;


    float randZ = curand_uniform(&state[tid]);  
    randVal = curand_uniform(&state[tid]);
    randNeg = (int)(randVal * 10);
    scale = 10;
    if(randNeg > 5){
      scale = -10;
    }
    positions[2 + 3*i] = scale * randZ;
    savedPositions[(i * 3 * numSteps) + (0 * 3) + 2] = scale * randZ;


    // Initialize velocities
    float randXV = curand_uniform(&state[tid]);
    randVal = curand_uniform(&state[tid]);
    randNeg = (int)(randVal * 10);
    scale = 10;
    if(randNeg > 5){
      scale = -10;
    }
    velocities[0 + 3*i] = scale * randXV;

    float randYV = curand_uniform(&state[tid]);
    randVal = curand_uniform(&state[tid]);
    randNeg = (int)(randVal * 10);
    scale = 10;
    if(randNeg > 5){
      scale = -10;
    }
    velocities[1 + 3*i] = scale * randYV;

    float randZV = curand_uniform(&state[tid]);
    randVal = curand_uniform(&state[tid]);
    randNeg = (int)(randVal * 10);
    scale = 10;
    if(randNeg > 5){
      scale = -10;
    }
    velocities[2 + 3*i] = scale * randZV;

  }
}

__global__ void calculateAccels_kernel(int N, float *accelerations, float *positions, float *mass, float G){
  
  // Calculates the accelerations for each particle
  // Only calculate one particle per thread
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Each thread only calculates the new acceleration for its specific particle
  float ax = 0;
  float ay = 0;
  float az = 0;

  // Go through each of the other particles
  for(int j = 0; j < N; j++){
    if(j != tid){
      // Find the different between each x, y, z component of two particles
      float dx = positions[0 + j*3] - positions[0 + tid*3];
      float dy = positions[1 + j*3] - positions[1 + tid*3];
      float dz = positions[2 + j*3] - positions[2 + tid*3];

      // Calculate the magnitude cubed with softening equal to 0.1
      float magnitudeCube = pow(sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2) + pow(0.1, 2)), 3);
      ax = ax + mass[j] * (dx/magnitudeCube);
      ay = ay + mass[j] * (dy/magnitudeCube);
      az = az + mass[j] * (dz/magnitudeCube);
    }
  }
  // Multiply with gravitational constant
  ax = ax * G;
  ay = ay * G;
  az = az * G;
  // Update accelerations
  accelerations[0 + tid*3] = ax;
  accelerations[1 + tid*3] = ay;
  accelerations[2 + tid*3] = az;

}


__global__ void calculate1_kernel(int N, float *positions, float *velocities, float* accelerations, float* savedPositions, float dt, int i, int numSteps){
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // First half step of kick-drift-kick
  // Only update the specific thread's velocity and position
  // Can't calculate acceleration here because positions need to be updated for all
  // threads and particles
  velocities[0 + tid*3] = velocities[0 + tid*3] + accelerations[0 + tid*3] * (dt/2.0);
  velocities[1 + tid*3] = velocities[1 + tid*3] + accelerations[1 + tid*3] * (dt/2.0);
  velocities[2 + tid*3] = velocities[2 + tid*3] + accelerations[2 + tid*3] * (dt/2.0);

  positions[0 + tid*3] = positions[0 + tid*3] + velocities[0 + tid*3] * dt;
  positions[1 + tid*3] = positions[1 + tid*3] + velocities[1 + tid*3] * dt;
  positions[2 + tid*3] = positions[2 + tid*3] + velocities[2 + tid*3] * dt;

  // index = (z * xMax * yMax) + (y * xMax) + x
  // xMax is the size of the innermost array
  // yMax is the size of the middle array
  savedPositions[(tid * 3 * numSteps) + (i * 3) + 0] = positions[0 + tid*3];
  savedPositions[(tid * 3 * numSteps) + (i * 3) + 1] = positions[1 + tid*3];
  savedPositions[(tid * 3 * numSteps) + (i * 3) + 2] = positions[2 + tid*3];
  
  
}

__global__ void calculate2_kernel(int N, float * positions, float *velocities, float *accelerations, float dt){
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // The second half step
  // Calculate new velocity
  velocities[0 + tid*3] = velocities[0 + tid*3] + accelerations[0 + tid*3] * (dt/2);
  velocities[1 + tid*3] = velocities[1 + tid*3] + accelerations[1 + tid*3] * (dt/2);
  velocities[2 + tid*3] = velocities[2 + tid*3] + accelerations[2 + tid*3] * (dt/2);

}

curandState* init_rand() {
  curandState *d_state;
  cudaMalloc(&d_state, N_BLOCKS * N_THREADS * sizeof(curandState));
  init_rand_kernel<<<N_BLOCKS, N_THREADS>>>(d_state);
  return d_state;
}


float* simulate(int N, float dt, float t, float tEnd, float G){

    curandState* d_state = init_rand();
    float* positions;
    float* d_positions;

    float* velocities;
    float* d_velocities;

    float* accelerations;
    float* d_accelerations;

    float* masses;
    float* d_masses;

    float* savedPositions;
    float* d_savedPositions;

    int numSteps = tEnd/dt;

  // Allocate memory 
    positions = (float*)malloc((N * 3) * sizeof(float));
    velocities = (float*)malloc((N * 3) * sizeof(float));
    accelerations = (float*)malloc((N * 3) * sizeof(float));
    masses = (float*)malloc(N * sizeof(float));
    savedPositions = (float*)malloc((N * (3 * numSteps)) * sizeof(float));

    cudaMalloc(&d_positions, (N * 3) * sizeof(float));
    cudaMalloc(&d_velocities, (N * 3) * sizeof(float));
    cudaMalloc(&d_accelerations, (N * 3) * sizeof(float));
    cudaMalloc(&d_masses, N * sizeof(float));
    cudaMalloc(&d_savedPositions, (N * 3 * numSteps) * sizeof(float));


    // Initialize masses for each particle
    masses[0] = 5;
    masses[1] = 5;
    for(int i = 2; i < N; i++){
      masses[i] = 0.2;
    }

    // Copy masses to device
    cudaMemcpy(d_masses, masses, N * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "masses at 0: " << masses[0] << std::endl;

    // Initialize starting conditions
    initialize_kernel<<<N_BLOCKS, N_THREADS>>>(N, d_positions, d_velocities, d_masses, d_state, d_savedPositions, numSteps);

    // Need to copy back to host function then back to device? Or will it be fine?
    cudaMemcpy(positions, d_positions, (N * 3) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, d_velocities, (N * 3) * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(masses, d_masses, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(savedPositions, d_savedPositions, (N * 3 * numSteps) * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<< "x\t y\t z" << std::endl;
    std::cout<< positions[0 + 0*3] << "\t" << positions[1 + 0*3] << "\t" << positions[2 + 0*3] << std::endl;
    std::cout<< savedPositions[0] << "\t" << savedPositions[1] << "\t" << savedPositions[2] << std::endl;


    calculateAccels_kernel<<<N_BLOCKS, N_THREADS>>>(N, d_accelerations, d_positions, d_masses, G);

    cudaMemcpy(accelerations, d_accelerations, (N * 3) * sizeof(float), cudaMemcpyDeviceToHost);


    // Have the for loop start here then send to device function
    // Calculate one particle per thread
    // 0 is the initial position
    for(int i = 1; i < numSteps; i++){
      // std::cout << "i" << i << std::endl;
      // First part of leapfrog
      calculate1_kernel<<<N_BLOCKS, N_THREADS>>>(N, d_positions, d_velocities, d_accelerations, d_savedPositions, dt, i, numSteps);
      cudaDeviceSynchronize();

      // Recalculate accelerations
      calculateAccels_kernel<<<N_BLOCKS, N_THREADS>>>(N, d_accelerations, d_positions, d_masses, G);
      cudaDeviceSynchronize();

      // Second part of leapfrog timestep
      calculate2_kernel<<<N_BLOCKS, N_THREADS>>>(N, d_positions, d_velocities, d_accelerations, dt);
      // Wait until all threads have finished working before going on to next step
      cudaDeviceSynchronize();
    }

    cudaMemcpy(positions, d_positions, (N * 3) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(savedPositions, d_savedPositions, (N * 3 * numSteps) * sizeof(float), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < N; i++){
    //   std::cout<< "-------------" << i << "-------------" << std::endl;
    //   std::cout<< "x\t y\t z" << std::endl;
    //   std::cout<< positions[0 + i*3] << "\t" << positions[1 + i*3] << "\t" << positions[2 + i*3] << std::endl;
    // }

    std::cout<< "x\t y\t z" << std::endl;
    std::cout<< positions[0 + 0*3] << "\t" << positions[1 + 0*3] << "\t" << positions[2 + 0*3] << std::endl;

    cudaFree(d_state);
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_masses);
    cudaFree(d_accelerations);

    free(positions);
    free(velocities);
    // free(masses);
    free(accelerations);

    return savedPositions;
}



int main(int argc, char** argv) {

  // Initialize constant variables
  int N = 100;
  float dt = 0.01;
  float t = 0;
  float tEnd = 10;
  float G = pow(6.67 * 10, -11);


  // To calculate the total runtime
  auto start = std::chrono::high_resolution_clock::now();

  float* savedPositions = simulate(N, dt, t, tEnd, G);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
  std::cout<< "Done in " << duration.count() << " microseconds" << std::endl;

  // The number of steps taken by each particle 
  int numSteps = tEnd/dt;

  // Save positions from scenario one
  std::ofstream myFile1;
  myFile1.open("../python_files/all_positions_1-2.txt");

  // Track keeps track of the placement for each x, y, z component position
  // Also keeps track of when the next particle's positions start
  int trackY = 0;
  int trackZ = 0;
  myFile1 << "Particle 1 \n";
  for(int i = 0; i < N*numSteps; i++){

    if(trackY == numSteps){
      trackZ++;
      trackY = 0;
      myFile1 << "Particle ";
      myFile1 << trackZ+1;
      myFile1 << "\n";
    }
    myFile1 << savedPositions[(trackZ * 3 * numSteps) + (trackY * 3) + 0];
    myFile1 << ", ";
    myFile1 << savedPositions[(trackZ * 3 * numSteps) + (trackY * 3) + 1];
    myFile1 << ", ";
    myFile1 << savedPositions[(trackZ * 3 * numSteps) + (trackY * 3) + 2];
    myFile1 << "\n";

    trackY++;
    
  }

  myFile1.close();

  // First file: all 0.2
  // Second file: 1 and 2 masses are 5, all rest are 0.01*i
  // Third file: All masses are 0.2 * i
  free(savedPositions);

}