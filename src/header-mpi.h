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
void exchange_ghost_cells(int start_row, int end_row, int rank, int size,  int total_local_cells, int ncside, int truesize, MPI_Comm comm, cell_t *local_cells, cell_t *ghost_upper, cell_t *ghost_lower);
void build_com(particle_t *par, long long n_part, long ncside, double cell_size, double inv_cell_size, long total_cells, cell_t *cells);
void print_cells(cell_t *cells, long ncside, int rank);
void calculate_forces(particle_t *par, cell_t *cells, long long n_part, long ncside, double side, int start_row, int end_row, long total_cells, cell_t *ghost_lower, cell_t *ghost_upper);
void update_positions_and_velocities(particle_t *par, particle_t *to_remove, cell_t *cells, int start_row, int end_row, long *ghost_par_count, long long n_part, long ncside, double side, double inv_cell_size, long total_cells, int rank);
void exchange_particles(int rank, int size, int local_rows, int ncside, int truesize, MPI_Comm comm, particle_t **local_particles, particle_t *ghost_particles, long ghost_par_count, long long *local_n_part);
void detect_collisions(cell_t *local_cells, particle_t *local_particles, long ncside, long long *local_n_part, long long *local_collision_count, long local_total_cells, long long t);

void free_cell_lists(cell_t *cells, long ncside, long total_cells);
void run_time_step(particle_t *par, long long *n_part, long ncside, double side, double cell_side, double inv_cell_side, long total_cells, long long *collision_count, long long timestep);
void print_particles(particle_t *par, long long n_part);


#endif