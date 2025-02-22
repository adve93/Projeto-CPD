#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#define G 6.67408e-11
#define EPSILON2 (0.005*0.005)
#define DELTAT 0.1
#include "init_particles.h"

////////////////Professor Function declarations////////
unsigned int seed;
void init_r4uni(int input_seed)
{
    seed = input_seed + 987654321;
}
double rnd_uniform01()
{
    int seed_in = seed;
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return 0.5 + 0.2328306e-09 * (seed_in + (int) seed);
}
double rnd_normal01()
{
    double u1, u2, z, result;
    do {
        u1 = rnd_uniform01();
        u2 = rnd_uniform01();
        z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
        result = 0.5 + 0.15 * z;      // Shift mean to 0.5 and scale
    } while (result < 0 || result >= 1);
    return result;
}

void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par)
{
    double (*rnd01)() = rnd_uniform01;
    long long i;

    if(seed < 0) {
        rnd01 = rnd_normal01;
        seed = -seed;
    }
    
    init_r4uni(seed);

    for(i = 0; i < n_part; i++) {
        par[i].x = rnd01() * side;
        par[i].y = rnd01() * side;
        par[i].vx = (rnd01() - 0.5) * side / ncside / 5.0;
        par[i].vy = (rnd01() - 0.5) * side / ncside / 5.0;

        par[i].m = rnd01() * 0.01 * (ncside * ncside) / n_part / G * EPSILON2;
    }
}
///////////////////////////////////////////////////////

////////////////Main Code//////////////////////////////


int main() {
    
    
    long seed = 1;
    double side = 3; //Side total 
    long ncside = 3; //Number of cells per side
    long long n_part = 10; //Number of particles
    particle_t particles_arr[n_part]; //Array of particles initizalized by teacher's function
    const double cell_side = side / ncside; //Side of each cell
    com_t com[ncside*ncside]; //1D flattened Array of center of mass of each cell

    init_particles(seed, side, ncside, n_part, particles_arr); //Initialize particles

    particle_cell(particles_arr, n_part, cell_side); //Calculate cell of each particle
    
    calculate_com(particles_arr, com, n_part, ncside); //Calculate center of mass of each cell
    
    print_com(com, ncside);

    return 0;
}   

//Function to run at the beginning of every time step to calculate center of mass of each cell
void calculate_com(particle_t *par, com_t *com, long long n_part, long ncside) {
    int cell_index;

    // Reset center of mass data for each cell
    for (long i = 0; i < ncside * ncside; i++) {
        com[i].x = 0;
        com[i].y = 0;
        com[i].m = 0;
    }

    // Accumulate mass and weighted positions
    for (long long i = 0; i < n_part; i++) {
        cell_index = par[i].y_cell * ncside + par[i].x_cell; // Flatten 2D index to 1D array

        com[cell_index].x += par[i].x * par[i].m;
        com[cell_index].y += par[i].y * par[i].m;
        com[cell_index].m += par[i].m;
    }

    // Compute center of mass for each cell
    for (long i = 0; i < ncside * ncside; i++) {
        if (com[i].m > 0) { // If that cell has particles
            com[i].x /= com[i].m;
            com[i].y /= com[i].m;
        }
    }
}

void particle_cell(particle_t *par,long long n_part, double cell_size) {
    long long i;

    for(i = 0; i < n_part; i++) {
        
        par[i].x_cell = (int) (par[i].x/cell_size);
        par[i].y_cell = (int) (par[i].y/cell_size);

    }
}

//Test function to print center of mass of each cell
void print_com(com_t *com, long ncside) {
    for(long i = 0; i < ncside * ncside; i++) {
        printf("COM %d: X:%f Y:%f M:%f\n", i, com[i].x, com[i].y, com[i].m);
    }
}

//Test function to print particle data
void print_particles(particle_t *par, long long n_part) {
    for(long long i = 0; i < n_part; i++) {
        printf("Particle %d: X:%f Y:%f VX:%f VY:%f M:%f X_CELL:%d Y_CELL:%d\n", i, par[i].x, par[i].y, par[i].vx, par[i].vy, par[i].m, par[i].x_cell, par[i].y_cell);
    }
}

