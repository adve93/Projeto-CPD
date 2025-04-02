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
    int new_x_cell, new_y_cell; // New cell indices
    int cell_pos;
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
cell_t *init_cells(particle_t *par, long total_cells, long long n_part, long ncside, double cell_side);
void run_time_steps(particle_t *par, cell_t *cells, long long *n_part, long ncside, double side, double cell_side, long long *collision_count, long total_cells, long long time_steps);
void free_cell_particle_lists(particle_t *par, cell_t *cells, long ncside);
void print_particles(particle_t *par, long long n_part);
void print_cells(cell_t *cells, long ncside);
void print_forces(particle_t *par, long long n_part);

#endif