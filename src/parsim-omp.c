#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "header.h"

#define _USE_MATH_DEFINES
#define G 6.67408e-11
#define EPSILON2 (0.005*0.005)
#define DELTAT 0.1
#define M_PI 3.14159265358979323846

///////////////////////////////////////
// Random and Initialization Functions
///////////////////////////////////////

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

void init_particles(long userseed, double side, long ncside, long long n_part, particle_t *par)
{
    double (*rnd01)() = rnd_uniform01;
    long long i;

    if(userseed < 0) {
        rnd01 = rnd_normal01;
        userseed = -userseed;
    }
    
    init_r4uni(userseed);

    for(i = 0; i < n_part; i++) {
        par[i].x = rnd01() * side;
        par[i].y = rnd01() * side;
        par[i].vx = (rnd01() - 0.5) * side / ncside / 5.0;
        par[i].vy = (rnd01() - 0.5) * side / ncside / 5.0;

        par[i].m = rnd01() * 0.01 * (ncside * ncside) / n_part / G * EPSILON2;
    }
}

//////////////////////////////////////
// Main Function
//////////////////////////////////////

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <seed> <side> <grid_size> <num_particles> <time_steps>\n", argv[0]);
        return 1;
    }

    double exec_time;
    long seed = atol(argv[1]);              // Seed
    double side = atof(argv[2]);            // Domain side length
    long ncside = atol(argv[3]);            // Number of cells per side
    long long n_part = atoll(argv[4]);      // Number of particles
    long long time_steps = atoll(argv[5]);  // Number of time steps
    
    particle_t *particles_arr = malloc(n_part * sizeof(particle_t));
    if (!particles_arr) {
        fprintf(stderr, "Memory allocation failed for particles.\n");
        return 1;
    }
    const double cell_side = side / ncside;
    init_particles(seed, side, ncside, n_part, particles_arr);
    exec_time = -omp_get_wtime();

    long long collision_count = 0;
    for (long long t = 0; t < time_steps; t++) {
        // printf("Time step %lld\n", t);
        run_time_step(particles_arr, &n_part, ncside, side, cell_side, &collision_count, t);
    }
    // print_particles(particles_arr, n_part);

    exec_time += omp_get_wtime();

    printf("%.3f %.3f\n", particles_arr[0].x, particles_arr[0].y);
    printf("%lld\n", collision_count);
    fprintf(stderr, "%.1fs\n", exec_time);

    free(particles_arr);
    return 0;
}

///////////////////////////////////////
// Run time pipeline
///////////////////////////////////////

// Run one simulation time step using spatial partitioning
void run_time_step(particle_t *par, long long *n_part, long ncside, double side, double cell_side, long long *collision_count, long long timestep) {
    // Print particle positions
    //print_particles(par, *n_part);
    // 1. Combined cell assignment and cell list build.
    cell_t *cells = assign_particles_and_build_cells(par, *n_part, ncside, cell_side);
    // Print COM for each cell
    //print_cells(cells, ncside);
    // 3. Compute forces using spatial partitioning: same-cell and adjacent cells
    calculate_forces(par, cells, n_part, ncside, side);
    // 4. Update positions and velocities
    update_positions_and_velocities(par, *n_part, side, cell_side, cells, ncside);
    // 5. Detect collisions (check only within the same cell)
    detect_collisions(cells, par, ncside, n_part, collision_count, side, timestep);
    // Free cell lists for this time step
    free_cell_lists(cells, ncside);
}

///////////////////////////////////////
// Spatial Partitioning Structures and Functions
///////////////////////////////////////

cell_t* assign_particles_and_build_cells(particle_t *par, long long n_part, long ncside, double cell_size) {
    long total_cells = ncside * ncside;
    double inv_cell_size = 1.0 / cell_size;
    cell_t *cells = malloc(total_cells * sizeof(cell_t));
    if (!cells) {
        fprintf(stderr, "Memory allocation failed for cell lists.\n");
        exit(1);
    }

    int *cell_counts = calloc(total_cells, sizeof(int));
    if (!cell_counts) {
        fprintf(stderr, "Memory allocation failed for cell counts.\n");
        exit(1);
    }

    // Phase 1: Count particles per cell (parallel)
    #pragma omp parallel for
    for (long long i = 0; i < n_part; i++) {
        int x_cell = (int)(par[i].x * inv_cell_size);
        int y_cell = (int)(par[i].y * inv_cell_size);
        int cell_index = y_cell * ncside + x_cell;
        #pragma omp atomic
        cell_counts[cell_index]++;
    }

    // Initialize cells
    for (long i = 0; i < total_cells; i++) {
        cells[i].capacity = cell_counts[i] > 10 ? cell_counts[i] : 10;
        cells[i].count = 0;
        cells[i].indices = malloc(cells[i].capacity * sizeof(int));
        if (!cells[i].indices) {
            fprintf(stderr, "Memory allocation failed for cell indices.\n");
            exit(1);
        }
        cells[i].x = 0;
        cells[i].y = 0;
        cells[i].m = 0;
    }

    // Phase 2: Assign particles to cells (parallel)
    #pragma omp parallel for
    for (long long i = 0; i < n_part; i++) {
        int x_cell = (int)(par[i].x * inv_cell_size);
        int y_cell = (int)(par[i].y * inv_cell_size);
        par[i].x_cell = x_cell;
        par[i].y_cell = y_cell;
        int cell_index = y_cell * ncside + x_cell;
        int local_index;
        #pragma omp atomic capture
        local_index = cells[cell_index].count++;
        cells[cell_index].indices[local_index] = i;
    }

    // Phase 3: Compute center of mass in sequential order
    for (long i = 0; i < total_cells; i++) {
        for (int j = 0; j < cells[i].count; j++) {
            int idx = cells[i].indices[j];
            cells[i].x += par[idx].x * par[idx].m;
            cells[i].y += par[idx].y * par[idx].m;
            cells[i].m += par[idx].m;
        }
        if (cells[i].m > 0) {
            cells[i].x /= cells[i].m;
            cells[i].y /= cells[i].m;
        }
    }

    free(cell_counts);
    return cells;
}

void free_cell_lists(cell_t *cells, long ncside) {
    long total_cells = ncside * ncside;
    for (long i = 0; i < total_cells; i++) {
        free(cells[i].indices);
    }
    free(cells);
}

///////////////////////////////////////
// Simulation Helper Functions
///////////////////////////////////////


// Calculate forces on each particle using spatial partitioning,
void calculate_forces(particle_t *par, cell_t *cells, long long *n_part, long ncside, double side) {
    int num_threads = omp_get_max_threads();
    double **local_ax = malloc(num_threads * sizeof(double*));
    double **local_ay = malloc(num_threads * sizeof(double*));
    for (int t = 0; t < num_threads; t++) {
        local_ax[t] = calloc(*n_part, sizeof(double));
        local_ay[t] = calloc(*n_part, sizeof(double));
    }

    for (long long i = 0; i < *n_part; i++) {
        par[i].ax = 0;
        par[i].ay = 0;
    }

    #pragma omp parallel for schedule(dynamic)
    for (long cell = 0; cell < ncside * ncside; cell++) {
        int thread_id = omp_get_thread_num();
        cell_t current = cells[cell];
        for (int a = 0; a < current.count; a++) {
            int i = current.indices[a];
            for (int b = a + 1; b < current.count; b++) {
                int j = current.indices[b];
                double dx = par[j].x - par[i].x;
                double dy = par[j].y - par[i].y;
                double dist2 = dx * dx + dy * dy;
                double inv_r = 1.0 / sqrt(dist2);
                double force = G * (par[i].m * par[j].m) / dist2;
                double fx = force * dx * inv_r;
                double fy = force * dy * inv_r;
                local_ax[thread_id][i] += fx / par[i].m;
                local_ay[thread_id][i] += fy / par[i].m;
                local_ax[thread_id][j] -= fx / par[j].m;
                local_ay[thread_id][j] -= fy / par[j].m;
            }
        }
    }

    #pragma omp parallel for
    for (long long i = 0; i < *n_part; i++) {
        int thread_id = omp_get_thread_num();
        int x_cell = par[i].x_cell;
        int y_cell = par[i].y_cell;
        for (int dxc = -1; dxc <= 1; dxc++) {
            for (int dyc = -1; dyc <= 1; dyc++) {
                if (dxc == 0 && dyc == 0) continue;
                int nx = (x_cell + dxc + ncside) % ncside;
                int ny = (y_cell + dyc + ncside) % ncside;
                int neighbor_index = ny * ncside + nx;
                if (cells[neighbor_index].m == 0) continue;
                double dx_cm = cells[neighbor_index].x - par[i].x;
                double dy_cm = cells[neighbor_index].y - par[i].y;
                int diff_x = nx - x_cell;
                if (diff_x > ncside/2) diff_x -= ncside;
                if (diff_x < -ncside/2) diff_x += ncside;
                if (diff_x > 0 && dx_cm < 0) dx_cm += side;
                else if (diff_x < 0 && dx_cm > 0) dx_cm -= side;
                int diff_y = ny - y_cell;
                if (diff_y > ncside/2) diff_y -= ncside;
                if (diff_y < -ncside/2) diff_y += ncside;
                if (diff_y > 0 && dy_cm < 0) dy_cm += side;
                else if (diff_y < 0 && dy_cm > 0) dy_cm -= side;
                double dist2_cm = dx_cm * dx_cm + dy_cm * dy_cm;
                double inv_r_cm = 1.0 / sqrt(dist2_cm);
                double force_cm = G * (par[i].m * cells[neighbor_index].m) / dist2_cm;
                double fx_cm = force_cm * dx_cm * inv_r_cm;
                double fy_cm = force_cm * dy_cm * inv_r_cm;
                local_ax[thread_id][i] += fx_cm / par[i].m;
                local_ay[thread_id][i] += fy_cm / par[i].m;
            }
        }
    }

    for (long long i = 0; i < *n_part; i++) {
        for (int t = 0; t < num_threads; t++) {
            par[i].ax += local_ax[t][i];
            par[i].ay += local_ay[t][i];
        }
    }

    for (int t = 0; t < num_threads; t++) {
        free(local_ax[t]);
        free(local_ay[t]);
    }
    free(local_ax);
    free(local_ay);
}

// Combined function to update positions and velocities in one loop.
void update_positions_and_velocities(particle_t *par, long long n_part, double side, double cell_size, cell_t *cells, long ncside) {
    double inv_cell_size = 1.0 / cell_size;
    long total_cells = ncside * ncside;

    // **Step 1: Reset cell counts**
    // Prepare the global cells array for new assignments by resetting counts.
    #pragma omp parallel for
    for (long i = 0; i < total_cells; i++) {
        cells[i].count = 0;
    }

    // **Step 2: Allocate temporary array for counting particles per cell**
    int *cell_counts = calloc(total_cells, sizeof(int));
    if (!cell_counts) {
        // Handle allocation failure if necessary
        return;
    }

    // **Step 3: Update positions and count particles per cell**
    #pragma omp parallel for
    for (long long i = 0; i < n_part; i++) {
        // Update position using velocity and acceleration
        par[i].x += par[i].vx * DELTAT + 0.5 * par[i].ax * DELTAT * DELTAT;
        par[i].y += par[i].vy * DELTAT + 0.5 * par[i].ay * DELTAT * DELTAT;

        // Apply periodic boundary conditions to keep particles within [0, side)
        par[i].x = fmod(par[i].x, side);
        if (par[i].x < 0) par[i].x += side;
        par[i].y = fmod(par[i].y, side);
        if (par[i].y < 0) par[i].y += side;

        // Compute new cell coordinates
        int new_x_cell = (int)floor(par[i].x * inv_cell_size);
        int new_y_cell = (int)floor(par[i].y * inv_cell_size);
        // Clamp to valid cell indices
        if (new_x_cell >= ncside) new_x_cell = ncside - 1;
        if (new_y_cell >= ncside) new_y_cell = ncside - 1;
        int new_cell_index = new_y_cell * ncside + new_x_cell;

        // Store cell coordinates in particle for later use
        par[i].x_cell = new_x_cell;
        par[i].y_cell = new_y_cell;

        // Atomically increment the count for this cell
        #pragma omp atomic
        cell_counts[new_cell_index]++;
    }

    // **Step 4: Reallocate cells[c].indices based on new counts**
    for (long c = 0; c < total_cells; c++) {
        free(cells[c].indices); // Free previous allocation
        cells[c].capacity = cell_counts[c];
        cells[c].count = cell_counts[c];
        if (cells[c].capacity > 0) {
            cells[c].indices = malloc(cells[c].capacity * sizeof(int));
            if (!cells[c].indices) {
                // Handle allocation failure if necessary
                free(cell_counts);
                return;
            }
        } else {
            cells[c].indices = NULL;
        }
    }

    // **Step 5: Initialize offsets for assigning particle indices**
    int *cell_offset = calloc(total_cells, sizeof(int));
    if (!cell_offset) {
        // Handle allocation failure if necessary
        free(cell_counts);
        return;
    }

    // **Step 6: Assign particles to cells**
    #pragma omp parallel for
    for (long long i = 0; i < n_part; i++) {
        int cell_index = par[i].y_cell * ncside + par[i].x_cell;
        int pos;
        // Atomically get the next available position in the cell's indices array
        #pragma omp atomic capture
        pos = cell_offset[cell_index]++;
        cells[cell_index].indices[pos] = i;
    }

    // **Step 7: Clean up temporary arrays**
    free(cell_counts);
    free(cell_offset);

    // **Step 8: Update velocities**
    #pragma omp parallel for
    for (long long i = 0; i < n_part; i++) {
        par[i].vx += par[i].ax * DELTAT;
        par[i].vy += par[i].ay * DELTAT;
    }
}

void detect_collisions(cell_t *cells, particle_t *par, long ncside, long long *n_part, long long *collision_count, double side, long long timestep) {
    int *marked_for_removal = calloc(*n_part, sizeof(int)); // Track particles for deletion
    if (!marked_for_removal) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    long long local_collision_count = 0; // Each thread will maintain its own counter

    // **Parallelize the outer loop over cells**
    #pragma omp parallel for reduction(+:local_collision_count) schedule(dynamic)
    for (long cell = 0; cell < ncside * ncside; cell++) {
        cell_t current = cells[cell];
        if (current.count < 2) continue; // No collision possible

        for (int i = 0; i < current.count; i++) {
            int idx_i = current.indices[i];
            if (marked_for_removal[idx_i]) continue;

            int group_size = 1;
            int counted_this_group = 0;

            for (int j = i + 1; j < current.count; j++) {
                int idx_j = current.indices[j];
                if (marked_for_removal[idx_j]) continue;

                double dx = par[idx_j].x - par[idx_i].x;
                double dy = par[idx_j].y - par[idx_i].y;
                double dist2 = dx * dx + dy * dy;

                if (dist2 < EPSILON2) {
                    marked_for_removal[idx_j] = 1;
                    group_size++;

                    if (group_size == 2) {
                        marked_for_removal[idx_i] = 1;
                    }

                    if (!counted_this_group) {
                        local_collision_count++;
                        counted_this_group = 1;
                    }

                    // Check for a third colliding particle
                    for (int k = j + 1; k < current.count; k++) {
                        int idx_k = current.indices[k];
                        if (marked_for_removal[idx_k]) continue;

                        double dx_k = par[idx_k].x - par[idx_j].x;
                        double dy_k = par[idx_k].y - par[idx_j].y;
                        double dist2_k = dx_k * dx_k + dy_k * dy_k;

                        if (dist2_k < EPSILON2) {
                            marked_for_removal[idx_k] = 1;
                            group_size++;
                            if (group_size == 3) break;
                        }
                    }
                    break;
                }
            }
        }
    }

    // **Atomic update to global collision count**
    #pragma omp atomic
    *collision_count += local_collision_count;

    // **Particle Removal (Sequential Step)**
    long long new_n_part = 0;
    for (long long i = 0; i < *n_part; i++) {
        if (!marked_for_removal[i]) {
            par[new_n_part++] = par[i];
        }
    }
    *n_part = new_n_part;

    free(marked_for_removal);
}


void print_particles(particle_t *par, long long n_part) {
    for (int i = 0; i < n_part; i++) {
             printf("Particle %d: mass=%.6f x=%.6f y=%.6f vx=%.6f vy=%.6f\n X_cell: %d Y_cell: %d\n", 
                     i, par[i].m, par[i].x, par[i].y,
                     par[i].vx, par[i].vy, par[i].x_cell, par[i].y_cell);
        }
}

void print_cells(cell_t *cells, long ncside) {
    for (int i = 0; i < ncside * ncside; i++) {
        printf("Cell %d x: %.6f y: %.6f m: %.6f\n", i, cells[i].x, cells[i].y, cells[i].m);
    }
}


void print_forces(particle_t *par, long long n_part) {
    for (long long i = 0; i < n_part; i++) {
        printf("Particle %lld: AX:%f AY:%f \n", i, par[i].ax, par[i].ay);
    }
}