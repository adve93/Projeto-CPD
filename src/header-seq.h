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
    double x, y;    // Position of com
    double m;       // Mass of com
} cell_t;

cell_t* init_cells(long total_cells, long long n_part, long ncside, double inv_cell_size, particle_t *par);
void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);
void assign_particles_and_build_cells(particle_t *par, long long n_part, long ncside, double cell_size, double inv_cell_size, long total_cells, cell_t *cells);
void free_cell_lists(cell_t *cells, long ncside, long total_cells);
void calculate_forces(particle_t *par, cell_t *cells, long long *n_part, long ncside, double side, long total_cells);
void update_positions_and_velocities(particle_t *par, cell_t *cells, long long n_part, long ncside, double side, double inv_cell_size, long total_cells);
void detect_collisions(cell_t *cells, particle_t *par, long ncside, long long *n_part, long long *collision_count, long total_cells, long long timestep);
void run_time_step(particle_t *par, long long *n_part, long ncside, double side, double cell_side, double inv_cell_side, long total_cells, long long *collision_count, long long timestep, cell_t *cells);
void print_particles(particle_t *par, long long n_part);
void print_cells(cell_t *cells, long ncside);

#endif