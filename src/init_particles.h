#ifndef PARTICLES
#define PARTICLES

#include <math.h>

//Particle structure
typedef struct  {

    double x, y; //Position
    double vx, vy; //Velocity
    double ax, ay; //Acceleration
    double m; //Mass
    int x_cell, y_cell;
    int removed;
} particle_t;

//Center of mass structure
typedef struct  {

    double x, y; //Position
    double m; //Mass
} com_t;


void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);
void calculate_com(particle_t *par, com_t *com, long long n_part, long ncside);
void particle_cell(particle_t *par, long long n_part, double cell_size);
void calculate_forces(particle_t *par, com_t *com, long long n_part, long ncside, double side);
void update_velocities(particle_t *par, long long n_part);
void update_positions(particle_t *par, long long n_part, double side);
void print_com(com_t *com, long ncside);
void print_particles(particle_t *par, long long n_part);
void print_forces(particle_t *par, long long n_part);
void detect_collisions(particle_t *par, long long *n_part, long long *collision_count);

#endif