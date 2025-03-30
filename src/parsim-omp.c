#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include "header-omp.h"

#define _USE_MATH_DEFINES
#define G 6.67408e-11
#define EPSILON2 (0.005*0.005)
#define DELTAT 0.1

// Global variables for thread management
int max_threads = 0;

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

    // Set max threads once
    max_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    
    // Allocate memory for particles and cells
    particle_t *particles_arr = malloc(n_part * sizeof(particle_t));
    if (!particles_arr) {
        fprintf(stderr, "Memory allocation failed for particles.\n");
        return 1;
    }

    init_particles(seed, side, ncside, n_part, particles_arr);

    // Calculate cell side length
    const double cell_side = side / ncside;
    // Calculate total number of cells
    long total_cells = ncside * ncside;

    // Alocate memory for cells
    cell_t *cells = init_cells(particles_arr, total_cells, n_part, ncside, cell_side);

    exec_time = -omp_get_wtime();

    long long collision_count = 0;

    run_time_steps(particles_arr, cells, &n_part, ncside, side, cell_side, &collision_count, total_cells, time_steps);

    exec_time += omp_get_wtime();

    printf("%.3f %.3f\n", particles_arr[0].x, particles_arr[0].y);
    printf("%lld\n", collision_count);
    fprintf(stderr, "%.1fs\n", exec_time);

    free_cell_particle_lists(particles_arr, cells, ncside);
    return 0;
}

cell_t *init_cells(particle_t *par, long total_cells, long long n_part, long ncside, double cell_side)
{
    cell_t *cells = malloc(total_cells * sizeof(cell_t));
    if (!cells)
    {
        fprintf(stderr, "Memory allocation failed for cell lists.\n");
        exit(1);
    }
    for (long i = 0; i < total_cells; i++)
    {
        cells[i].capacity = 1000;
        cells[i].count = 0;
        cells[i].indices = malloc(cells[i].capacity * sizeof(int));
        if (!cells[i].indices)
        {
            fprintf(stderr, "Memory allocation failed for cell indices.\n");
            exit(1);
        }
        cells[i].x = 0;
        cells[i].y = 0;
        cells[i].m = 0;
    }

    for (long long i = 0; i < n_part; i++)
    {
        int x_cell = (int)(par[i].x / cell_side);
        int y_cell = (int)(par[i].y / cell_side);
        par[i].x_cell = x_cell;
        par[i].y_cell = y_cell;
        int cell_index = y_cell * ncside + x_cell;

        if (cells[cell_index].count == cells[cell_index].capacity)
        {
            cells[cell_index].capacity *= 2;
            cells[cell_index].indices = realloc(cells[cell_index].indices, cells[cell_index].capacity * sizeof(int));
            if (!cells[cell_index].indices)
            {
                fprintf(stderr, "Memory reallocation failed for cell indices.\n");
                exit(1);
            }
        }

        cells[cell_index].indices[cells[cell_index].count++] = i;
    }
    return cells;
}

///////////////////////////////////////
// Run time pipeline
///////////////////////////////////////

void run_time_steps(particle_t *par, cell_t *cells, long long *n_part, long ncside, double side, double cell_side, long long *collision_count, long total_cells, long long time_steps) {

    // Allocate marked_for_removal with initial size (won't need resizing since *n_part only decreases)
    int *marked_for_removal = calloc(*n_part, sizeof(int));
    if (!marked_for_removal) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    // Parallel region for all time steps
    #pragma omp parallel
    {

        for (long long t = 0; t < time_steps; t++) {
            // Reset cells
            #pragma omp for
            for (long i = 0; i < total_cells; i++) {
                cells[i].x = 0;
                cells[i].y = 0;
                cells[i].m = 0;

                for (long long j = 0; j < cells[i].count; j++) {
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

            // Compute forces between particles in the same cell
            #pragma omp for schedule(dynamic)
            for (long cell = 0; cell < total_cells; cell++) {
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
                        #pragma omp atomic
                        par[i].ax += fx / par[i].m;
                        #pragma omp atomic
                        par[i].ay += fy / par[i].m;
                        #pragma omp atomic
                        par[j].ax -= fx / par[j].m;
                        #pragma omp atomic
                        par[j].ay -= fy / par[j].m;
                    }
                }
            }

            // Compute forces between particles and adjacent cells' COM
            #pragma omp for
            for (long long i = 0; i < *n_part; i++) {
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
                        // Adjust for periodic boundaries
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
                        #pragma omp atomic
                        par[i].ax += fx_cm / par[i].m;
                        #pragma omp atomic
                        par[i].ay += fy_cm / par[i].m;
                    }
                }
            }

            // Update positions and velocities
            #pragma omp for
            for (long long i = 0; i < *n_part; i++) {

                // Store previous cell index
                int prev_x_cell = par[i].x_cell;
                int prev_y_cell = par[i].y_cell;
                int prev_cell_index = prev_y_cell * ncside + prev_x_cell;
                
                par[i].x += par[i].vx * DELTAT + 0.5 * par[i].ax * DELTAT * DELTAT;
                par[i].y += par[i].vy * DELTAT + 0.5 * par[i].ay * DELTAT * DELTAT;
                
                par[i].x = fmod(par[i].x, side);
                if (par[i].x < 0) par[i].x += side;
                par[i].y = fmod(par[i].y, side);
                if (par[i].y < 0) par[i].y += side;

                // Compute new cell index
                int new_x_cell = (int)(par[i].x / cell_side);
                int new_y_cell = (int)(par[i].y / cell_side);
                if (new_x_cell >= ncside) new_x_cell = ncside - 1;
                if (new_y_cell >= ncside) new_y_cell = ncside - 1;
                int new_cell_index = new_y_cell * ncside + new_x_cell;

                // Check if the particle has moved to a different cell
                if (new_cell_index != prev_cell_index) {
                    #pragma omp critical
                    {
                        int *indices = cells[prev_cell_index].indices;
                        int count = cells[prev_cell_index].count;
                        for (int j = 0; j < count; j++) {
                            if (indices[j] == i) {
                                indices[j] = indices[count - 1];
                                cells[prev_cell_index].count--;
                                break;
                            }
                        }
                    }
                    par[i].x_cell = new_x_cell;
                    par[i].y_cell = new_y_cell;
                    #pragma omp critical
                    {
                        if (cells[new_cell_index].count == cells[new_cell_index].capacity) {
                            cells[new_cell_index].capacity *= 2;
                            cells[new_cell_index].indices = realloc(cells[new_cell_index].indices, cells[new_cell_index].capacity * sizeof(int));
                            if (!cells[new_cell_index].indices) {
                                fprintf(stderr, "Memory reallocation failed for cell indices.\n");
                                exit(1);
                            }
                        }
                        cells[new_cell_index].indices[cells[new_cell_index].count++] = i;
                    }
                }
                // Update velocities
                par[i].vx += par[i].ax * DELTAT;
                par[i].vy += par[i].ay * DELTAT;
                // Reset acceleration for the next time step
                par[i].ax = 0;
                par[i].ay = 0;
            }

            #pragma omp for
            for (long long i = 0; i < *n_part; i++) {
                marked_for_removal[i] = 0;
            }

            // Step 1 & 2: Detect collisions and compact cell index lists
            #pragma omp for
            for (long cell = 0; cell < total_cells; cell++) {
                if (cells[cell].count < 2) continue;

                // Detect collisions and mark particles
                for (int i = 0; i < cells[cell].count; i++) {
                    int idx_i = cells[cell].indices[i];
                    if (marked_for_removal[idx_i]) continue;

                    int collision_detected = 0;
                    for (int j = i + 1; j < cells[cell].count; j++) {
                        int idx_j = cells[cell].indices[j];
                        if (marked_for_removal[idx_j]) continue;

                        double dx = par[idx_j].x - par[idx_i].x;
                        double dy = par[idx_j].y - par[idx_i].y;
                        double dist2 = dx * dx + dy * dy;

                        if (dist2 < EPSILON2) {
                            marked_for_removal[idx_i] = 1;
                            marked_for_removal[idx_j] = 1;
                            if (!collision_detected) {
                                #pragma omp atomic
                                (*collision_count)++;
                                collision_detected = 1;
                            }
                        }
                    }
                }

                int new_count = 0;
                for (int i = 0; i < cells[cell].count; i++) {
                    int idx = cells[cell].indices[i];
                    if (!marked_for_removal[idx]) {
                        cells[cell].indices[new_count++] = idx;
                    }
                }
                cells[cell].count = new_count;
            }

            // Step 3: Compact the particle array
            #pragma omp single
            {
                long long original_n = *n_part;
                long long new_n_part = 0;
                for (long long i = 0; i < original_n; i++) {
                    if (!marked_for_removal[i]) {
                        par[new_n_part] = par[i];
                        marked_for_removal[i] = new_n_part; // Store new index
                        new_n_part++;
                    } else {
                        marked_for_removal[i] = -1; // Mark as removed
                    }
                }
                *n_part = new_n_part;
            }

            // Step 4: Update cell index lists with new indices
            #pragma omp for
            for (long cell = 0; cell < total_cells; cell++) {
                for (int i = 0; i < cells[cell].count; i++) {
                    int old_idx = cells[cell].indices[i];
                    cells[cell].indices[i] = marked_for_removal[old_idx];
                }
            }
        }
    }
    free(marked_for_removal);
}

// Free memory allocated for particles and cells
void free_cell_particle_lists(particle_t *par, cell_t *cells, long ncside) {
    long total_cells = ncside * ncside;
    # pragma omp parallel for
    for (long i = 0; i < total_cells; i++) {
        free(cells[i].indices);
    }
    free(cells);
    free(par);
}

// Print functions for debugging
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