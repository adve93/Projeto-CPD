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

// Combined function: assign each particle to a cell and build cell lists.
cell_t* assign_particles_and_build_cells(particle_t *par, long long n_part, long ncside, double cell_size) {
    long total_cells = ncside * ncside;
    cell_t *cells = malloc(total_cells * sizeof(cell_t));
    if (!cells) {
        fprintf(stderr, "Memory allocation failed for cell lists.\n");
        exit(1);
    }
    // Initialize each cell with a small capacity.
    for (long i = 0; i < total_cells; i++) {
        cells[i].capacity = 10;
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
    // Loop over particles once: assign cell indices and add to corresponding cell list.
    double inv_cell_size = 1.0 / cell_size; 
    for (long long i = 0; i < n_part; i++) {
        int x_cell = (int)(par[i].x * inv_cell_size );
        int y_cell = (int)(par[i].y * inv_cell_size );
        par[i].x_cell = x_cell;
        par[i].y_cell = y_cell;
        int cell_index = y_cell * ncside + x_cell;

        if (cells[cell_index].count == cells[cell_index].capacity) {
            cells[cell_index].capacity *= 2;
            cells[cell_index].indices = realloc(cells[cell_index].indices, cells[cell_index].capacity * sizeof(int));
            if (!cells[cell_index].indices) {
                fprintf(stderr, "Memory reallocation failed for cell indices.\n");
                exit(1);
            }
        }
    
        cells[cell_index].indices[cells[cell_index].count++] = i;
        cells[cell_index].x += par[i].x * par[i].m;
        cells[cell_index].y += par[i].y * par[i].m;
        cells[cell_index].m += par[i].m;
    }

    for (long i = 0; i < total_cells; i++) {
        if (cells[i].m > 0) {
            cells[i].x /= cells[i].m;
            cells[i].y /= cells[i].m;
        }
    }

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
    double half_side = side / 2.0;

    // Reset accelerations
    for (long long i = 0; i < *n_part; i++) {
        par[i].ax = 0;
        par[i].ay = 0;
    }

    // 1. Compute same-cell interactions using each unique pair only once.
    for (long cell = 0; cell < ncside * ncside; cell++) {
        cell_t current = cells[cell];
        for (int a = 0; a < current.count; a++) {
            int i = current.indices[a];
            for (int b = a + 1; b < current.count; b++) {
                int j = current.indices[b];

                double dx = par[j].x - par[i].x;
                double dy = par[j].y - par[i].y;

                double dist2 = dx * dx + dy * dy;

                //if (dist2 < EPSILON2) continue;

                double inv_r = 1.0 / sqrt(dist2);
                double force = G * ((par[i].m * par[j].m) / dist2);
                double fx = force * dx * inv_r; // Actual force component in x direction
                double fy = force * dy * inv_r; // Actual force component in y direction

                par[i].ax += fx / par[i].m;
                par[i].ay += fy / par[i].m;
                par[j].ax -= fx / par[j].m;
                par[j].ay -= fy / par[j].m;

                // Debug print for same-cell interactions
                //printf("P%d/P%d mag: %.6f fx: %.6f fy: %.6f\n", i, j, force, fx, fy);
            }
        }
    }

    // 2. Compute forces from neighboring cell COMs
    for (long long i = 0; i < *n_part; i++) {
        int x_cell = par[i].x_cell;
        int y_cell = par[i].y_cell;
        for (int dxc = -1; dxc <= 1; dxc++) {
            for (int dyc = -1; dyc <= 1; dyc++) {
                if (dxc == 0 && dyc == 0) continue; // Skip own cell

                // Get neighbor cell with toroidal wrapping
                int nx = (x_cell + dxc + ncside) % ncside;
                int ny = (y_cell + dyc + ncside) % ncside;
                int neighbor_index = ny * ncside + nx;
                if (cells[neighbor_index].m == 0) continue; // Skip empty cell

                // Compute raw displacements.
                double dx_cm = cells[neighbor_index].x - par[i].x;
                double dy_cm = cells[neighbor_index].y - par[i].y;

                // Compute cell offset differences (normalized to -1, 0, or 1)
                int diff_x = nx - x_cell;
                if (diff_x > ncside/2) diff_x -= ncside;
                if (diff_x < -ncside/2) diff_x += ncside;

                int diff_y = ny - y_cell;
                if (diff_y > ncside/2) diff_y -= ncside;
                if (diff_y < -ncside/2) diff_y += ncside;

                // Adjust displacements based on the cell offset.
                if (diff_x > 0 && dx_cm < 0) {
                    dx_cm += side;
                } else if (diff_x < 0 && dx_cm > 0) {
                    dx_cm -= side;
                }

                if (diff_y > 0 && dy_cm < 0) {
                    dy_cm += side;
                } else if (diff_y < 0 && dy_cm > 0) {
                    dy_cm -= side;
                }

                // Compute squared distance, force, etc.
                double dist2_cm = dx_cm * dx_cm + dy_cm * dy_cm;
                //if (dist2_cm < EPSILON2) continue;
                double inv_r_cm = 1.0 / sqrt(dist2_cm);
                double force_cm = G * ((par[i].m * cells[neighbor_index].m) / dist2_cm);
                double fx_cm = force_cm * dx_cm * inv_r_cm;
                double fy_cm = force_cm * dy_cm * inv_r_cm;

                par[i].ax += fx_cm / par[i].m;
                par[i].ay += fy_cm / par[i].m;

                //printf("P%d/C%d mag: %.6f fx: %.6f fy: %.6f\n", i, neighbor_index, force_cm, fx_cm, fy_cm);
            }
        }
    }
}

// Combined function to update positions and velocities in one loop.
void update_positions_and_velocities(particle_t *par, long long n_part, double side, double cell_size, cell_t *cells, long ncside) {
    double inv_cell_size = 1.0 / cell_size;

    // Reset cell counts before updating
    for (long i = 0; i < ncside * ncside; i++) {
        cells[i].count = 0;
    }

    for (long long i = 0; i < n_part; i++) {
        // Update positions
        par[i].x += par[i].vx * DELTAT + 0.5 * par[i].ax * DELTAT * DELTAT;
        par[i].y += par[i].vy * DELTAT + 0.5 * par[i].ay * DELTAT * DELTAT;

        // Apply toroidal wrapping
        par[i].x = fmod(par[i].x, side);
        if (par[i].x < 0) par[i].x += side;

        par[i].y = fmod(par[i].y, side);
        if (par[i].y < 0) par[i].y += side;

        // Compute new cell
        int new_x_cell = (int)floor(par[i].x * inv_cell_size);
        int new_y_cell = (int)floor(par[i].y * inv_cell_size);

        // Clamp to ensure particles don't go out of bounds
        if (new_x_cell >= ncside) new_x_cell = ncside - 1;
        if (new_y_cell >= ncside) new_y_cell = ncside - 1;

        // Compute new cell index
        int new_cell_index = new_y_cell * ncside + new_x_cell;

        // Update particle cell assignment
        par[i].x_cell = new_x_cell;
        par[i].y_cell = new_y_cell;

        // Ensure the cell has enough space
        if (cells[new_cell_index].count == cells[new_cell_index].capacity) {
            cells[new_cell_index].capacity *= 2;
            cells[new_cell_index].indices = realloc(cells[new_cell_index].indices, cells[new_cell_index].capacity * sizeof(int));
            if (!cells[new_cell_index].indices) {
                fprintf(stderr, "Memory reallocation failed for cell indices.\n");
                exit(1);
            }
        }

        // Add particle to the new cell
        cells[new_cell_index].indices[cells[new_cell_index].count++] = i;

        // Update velocities
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

    // Loop through all cells
    for (long cell = 0; cell < ncside * ncside; cell++) {
        cell_t current = cells[cell];
        if (current.count < 2) continue; // No collision possible
        // Iterate over all particles in the cell
        for (int i = 0; i < current.count; i++) {
            int idx_i = current.indices[i];
            if (marked_for_removal[idx_i]) continue; // Already removed

            int group_size = 1; // Start tracking a collision group
            int counted_this_group = 0; // Ensure we count this collision event once

            for (int j = i + 1; j < current.count; j++) {
                int idx_j = current.indices[j];
                if (marked_for_removal[idx_j]) continue;

                double dx = par[idx_j].x - par[idx_i].x;
                double dy = par[idx_j].y - par[idx_i].y;
                double dist2 = dx * dx + dy * dy;

                if (dist2 <  EPSILON2) { // Collision detected
                    marked_for_removal[idx_j] = 1;
                    group_size++;

                    if (group_size == 2) {
                        marked_for_removal[idx_i] = 1; // Mark the first particle
                    }

                    // **Ensure we count this as ONE collision event**
                    if (!counted_this_group) {
                        (*collision_count)++;
                        counted_this_group = 1; // Prevent multiple counts for the same event
                        //printf("Collision %lld: P%d/P%d\n", *collision_count, idx_i, idx_j);
                    }

                    // Check for a third colliding particle
                    for (int k = j + 1; k < current.count; k++) {
                        int idx_k = current.indices[k];
                        if (marked_for_removal[idx_k]) continue;

                        double dx_k = par[idx_k].x - par[idx_j].x;
                        double dy_k = par[idx_k].y - par[idx_j].y;
                        double dist2_k = dx_k * dx_k + dy_k * dy_k;

                        if (dist2_k <  EPSILON2) {
                            marked_for_removal[idx_k] = 1;
                            group_size++;
                            if (group_size == 3) break;  // Stop at 3-particle collision
                        }
                    }
                    break;  // Stop after finding a collision for idx_i
                }
            }
        }
    }

    // Compact the particle array (remove collided particles)
    long long new_n_part = 0;
    for (long long i = 0; i < *n_part; i++) {
        if (!marked_for_removal[i]) {
            par[new_n_part++] = par[i]; // Keep valid particles
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