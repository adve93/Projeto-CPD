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

//Cell structure
typedef struct {
    int *indices;   // Dynamic array of particle indices in this cell
    int count;      // Number of particles in this cell
    int capacity;   // Capacity of the indices array
    double x, y;    //Position of com
    double m;       //Mass of com
} cell_t;

void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);
cell_t* assign_particles_and_build_cells(particle_t *par, long long n_part, long ncside, double cell_size);
void free_cell_lists(cell_t *cells, long ncside);
void calculate_forces(particle_t *par, cell_t *cells, long long *n_part, long ncside, double side);
void update_positions_and_velocities(particle_t *par, long long n_part, double side);
void detect_collisions(cell_t *cells, particle_t *par, long ncside, long long *n_part, long long *collision_count, double side);
void run_time_step(particle_t *par, long long *n_part, long ncside, double side, double cell_side, long long *collision_count);
void print_particles(particle_t *par, long long n_part);
void print_cells(cell_t *cells, long ncside);
void detect_collisions_2(cell_t *cells, particle_t *par, long ncside, long long *n_part, long long *collision_count, double side);

#endif