#define _USE_MATH_DEFINES
#define G 6.67408e-11
#define EPSILON2 (0.005*0.005)
#define DELTAT 0.1

#include "init_particles.h"
#include <iostream>

int main() {
    
    
    long seed = 12; 
    double side = 3; 
    long ncside = 9;
    long long n_part = 10; 

    particle_t particles_arr[n_part];

    init_particles(seed, side, ncside, n_part, particles_arr);
    

    std::cout << "Hello, World!" << std::endl;
    return 0;
}   


