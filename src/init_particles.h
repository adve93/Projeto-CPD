#ifndef PARTICLES
#define PARTICLES

#include <math.h>

typedef struct  {

    double x, y;
    double vx, vy;
    double m;
    int x_cell, y_cell;
} particle_t;


typedef struct  {

    double x, y;
    double m;
} com_t;


void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);
void calculate_com(particle_t *par, com_t *com, long long n_part, long ncside);
void particle_cell(particle_t *par, long long n_part, double cell_size);
void print_com(com_t *com, long ncside);
void print_particles(particle_t *par, long long n_part);
void update_velocities(particle_t *par, com_t *com, long long n_part, long ncside, double side);

#endif