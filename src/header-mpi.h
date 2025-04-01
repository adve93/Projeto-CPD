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
    int global_id;
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
void print_process_cell_assignment(int rank, int size, long ncside, int side, int start_row, int end_row);
void print_local_particles(int rank, int size, particle_t *par, long long n_part, int inv_cell_side);
void get_local_domain(int rank, int size, int ncside, int *start_row, int *end_row);
void initialize_and_distribute_cells(int rank, int size, long ncside, cell_t *local_cells);
void initialize_and_distribute_particles(int rank, int size, long ncside, double side, long long n_part_total, particle_t *local_particles, int inv_cell_side, long long local_n_part);
void exchange_ghost_cells(cell_t *cells, int start_row, int end_row, int rank, int size, MPI_Comm comm, cell_t *ghost_upper, cell_t *ghost_lower, int ncside);
void build_com(particle_t *par, long long n_part, long ncside, double cell_size, double inv_cell_size, long total_cells, cell_t *cells);
void print_cells(cell_t *cells, long ncside, int rank);

void free_cell_lists(cell_t *cells, long ncside, long total_cells);
void calculate_forces(particle_t *par, cell_t *cells, long long *n_part, long ncside, double side, long total_cells);
void update_positions_and_velocities(particle_t *par, cell_t *cells, long long n_part, long ncside, double side, double inv_cell_size, long total_cells);
void detect_collisions(cell_t *cells, particle_t *par, long ncside, long long *n_part, long long *collision_count, long total_cells, long long timestep);
void run_time_step(particle_t *par, long long *n_part, long ncside, double side, double cell_side, double inv_cell_side, long total_cells, long long *collision_count, long long timestep);
void print_particles(particle_t *par, long long n_part);


#endif