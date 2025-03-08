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

//Cell structure
typedef struct {
    int *indices;   // Dynamic array of particle indices in this cell
    int count;      // Number of particles in this cell
    int capacity;   // Capacity of the indices array
} cell_t;

void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);
void free_cell_lists(cell_t *cells, long ncside);
void calculate_com(particle_t *par, com_t *com, long long n_part, long ncside);
void calculate_forces(particle_t *par, com_t *com, cell_t *cells, long long *n_part, long ncside, double side);
void update_positions_and_velocities(particle_t *par, long long n_part, double side);
void detect_collisions(cell_t *cells, particle_t *par, long ncside, long long *n_part, long long *collision_count);
void run_time_step(particle_t *par, long long *n_part, com_t *com, long ncside, double side, double cell_side, long long *collision_count);

#endif