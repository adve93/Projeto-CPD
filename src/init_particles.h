#ifndef PARTICLES
#define PARTICLES

#include <math.h>

typedef struct  {

    double x, y;
    double vx, vy;
    double m;
    int x_cell, y_cell;
    double x_position, y_position;
} particle_t;


typedef struct  {

    double x, y;
    double m;
} com_t;


void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);
void calculate_com(particle_t *par, com_t *com, long long n_part);
void info_particle(particle_t *par);
void particle_cell(particle_t *par, long long n_part);

#endif