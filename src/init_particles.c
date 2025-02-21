#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include "init_particles.h"
#define G 6.67408e-11
#define EPSILON2 (0.005*0.005)
#define DELTAT 0.1


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

int main() {
    
    
    long seed = 1; 
    double side = 3; 
    long ncside = 100;
    long long n_part = 10; 
    particle_t particles_arr[n_part];
    int ncells = side*side; 
    com_t com[ncells];

    init_particles(seed, side, ncside, n_part, particles_arr);
    particle_cell(particles_arr, n_part);

    for(int i = 0; i < n_part; i++) {
        printf("Particle %d: X:%f Y:%f M:%f\n", i, particles_arr[i].x, particles_arr[i].y, particles_arr[i].m);
    }
    
    calculate_com(particles_arr, com, n_part);

    for(int i = 0; i < ncells; i++) {
        printf("COM %d: X:%f Y:%f M:%f\n", i, com[i].x, com[i].y, com[i].m);
    }   

    return 0;
}   

void calculate_com(particle_t *par, com_t *com, long long n_part) {
    int ind;
    for (long long i = 0; i < n_part; i++) {
        ind = (par[i].x_cell*3) + (par[i].y_cell);
        com[ind].x += par[i].x_position;
        com[ind].y += par[i].y_position;
        com[ind].m += par[i].m;
    }
}

void particle_cell(particle_t *par,long long n_part) {
    long long i;

    for(i = 0; i < n_part; i++) {
        
        par[i].x_cell = (int) par[i].x;
        par[i].y_cell = (int) par[i].y;

        par[i].x_position = par[i].x - par[i].x_cell;
        par[i].y_position = par[i].y - par[i].y_cell;
    }
}

