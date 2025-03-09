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
void init_r4uni(int input_seed) {
    seed = input_seed + 987654321;
}

double rnd_uniform01() {
    int seed_in = seed;
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return 0.5 + 0.2328306e-09 * (seed_in + (int) seed);
}

double rnd_normal01() {
    double u1, u2, z, result;
    do {
        u1 = rnd_uniform01();
        u2 = rnd_uniform01();
        z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
        result = 0.5 + 0.15 * z; // Shift mean to 0.5 and scale
    } while (result < 0 || result >= 1);
    return result;
}

void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par) {
    double (*rnd01)() = rnd_uniform01;
    if (seed < 0) {
        rnd01 = rnd_normal01;
        seed = -seed;
    }
    init_r4uni(seed);
    for (long long i = 0; i < n_part; i++) {
        par[i].x = rnd01() * side;
        par[i].y = rnd01() * side;
        par[i].vx = (rnd01() - 0.5) * side / ncside / 5.0;
        par[i].vy = (rnd01() - 0.5) * side / ncside / 5.0;
        par[i].m  = rnd01() * 0.01 * (ncside * ncside) / n_part / G * EPSILON2;
        par[i].removed = 0;
    }
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
        int x_cell = (int)(par[i].x * inv_cell_size);
        int y_cell = (int)(par[i].y * inv_cell_size);
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

    // Zero accelerations for all particles.
    for (long long i = 0; i < *n_part; i++) {
        par[i].ax = 0.0;
        par[i].ay = 0.0;
    }

    // 1. Compute same-cell interactions using each unique pair only once.
    //    For each cell, loop over all pairs (i, j) with i < j.
    for (long cell = 0; cell < ncside * ncside; cell++) {
        cell_t current = cells[cell];
        for (int a = 0; a < current.count; a++) {
            int i = current.indices[a];
            for (int b = a + 1; b < current.count; b++) {
                int j = current.indices[b];
                double dx = par[j].x - par[i].x;
                double dy = par[j].y - par[i].y;
                // Apply toroidal adjustments
                if (dx > half_side) dx -= side;
                if (dx < -half_side) dx += side;
                if (dy > half_side) dy += side;  // using your Y convention
                if (dy < -half_side) dy -= side;
                double dist2 = dx * dx + dy * dy;
                double r = sqrt(dist2);
                double force = (G * par[i].m * par[j].m) / dist2;
                double ux = dx / r;
                double uy = dy / r;
                // Particle i receives +F, particle j receives -F
                par[i].ax += force * ux / par[i].m;
                par[i].ay += force * uy / par[i].m;
                par[j].ax -= force * ux / par[j].m;
                par[j].ay -= force * uy / par[j].m;
            }
        }
    }

    // 2. For adjacent cells, use COM approximations.
    //    For each particle, add the contributions from its eight neighbors.
    for (long long i = 0; i < *n_part; i++) {
        int x_cell = par[i].x_cell;
        int y_cell = par[i].y_cell;
        for (int dxc = -1; dxc <= 1; dxc++) {
            for (int dyc = -1; dyc <= 1; dyc++) {
                if (dxc == 0 && dyc == 0) continue; // skip own cell
                int nx = (x_cell + dxc + ncside) % ncside;
                int ny = (y_cell + dyc + ncside) % ncside;
                int neighbor_index = ny * ncside + nx;
                if (cells[neighbor_index].m == 0) continue;
                double dx_cm = cells[neighbor_index].x - par[i].x;
                double dy_cm = cells[neighbor_index].y - par[i].y;
                if (dx_cm > half_side) dx_cm -= side;
                if (dx_cm < -half_side) dx_cm += side;
                if (dy_cm > half_side) dy_cm += side;
                if (dy_cm < -half_side) dy_cm -= side;
                double dist2_cm = dx_cm * dx_cm + dy_cm * dy_cm;
                double r_cm = sqrt(dist2_cm);
                double force_cm = (G * par[i].m * cells[neighbor_index].m) / dist2_cm;
                double ux_cm = dx_cm / r_cm;
                double uy_cm = dy_cm / r_cm;
                par[i].ax += force_cm * ux_cm / par[i].m;
                par[i].ay += force_cm * uy_cm / par[i].m;
            }
        }
    }
}

// Combined function to update positions and velocities in one loop.
void update_positions_and_velocities(particle_t *par, long long n_part, double side) {
    for (long long i = 0; i < n_part; i++) {
        // Update position.
        par[i].x += par[i].vx * DELTAT;
        par[i].y += par[i].vy * DELTAT;
        // Apply toroidal wrapping.
        if (par[i].x < 0) par[i].x += side;
        if (par[i].x >= side) par[i].x -= side;
        if (par[i].y < 0) par[i].y += side;
        if (par[i].y >= side) par[i].y -= side;
        // Update velocity.
        par[i].vx += par[i].ax * DELTAT;
        par[i].vy += par[i].ay * DELTAT;
    }
}

// Collision detection: check collisions within each cell.
void detect_collisions(cell_t *cells, particle_t *par, long ncside, long long *n_part, long long *collision_count) {
    int *marked_for_removal = calloc(*n_part, sizeof(int)); // Track which particles should be removed
    int *collision_group = calloc(*n_part, sizeof(int));    // Track if a particle is part of a collision chain
    if (!marked_for_removal || !collision_group) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    for (long cell = 0; cell < ncside * ncside; cell++) {
        cell_t current = cells[cell];

        for (int i = 0; i < current.count; i++) {
            int idx_i = current.indices[i];
            if (marked_for_removal[idx_i]) continue; // Skip already removed particles

            int group_size = 1; // Start a new chain collision group
            int first_particle = idx_i; // Keep track of the first particle in the chain

            for (int j = i + 1; j < current.count; j++) {
                int idx_j = current.indices[j];
                if (marked_for_removal[idx_j]) continue;

                double dx = par[idx_j].x - par[idx_i].x;
                double dy = par[idx_j].y - par[idx_i].y;
                double dist2 = dx * dx + dy * dy;

                if (dist2 <= EPSILON2) {
                    marked_for_removal[idx_j] = 1; // Mark for removal
                    collision_group[idx_j] = 1; // Assign to a chain collision
                    group_size++; // Increase the chain size

                    if (group_size == 2) {
                        marked_for_removal[idx_i] = 1; // Mark the first particle for removal
                        collision_group[idx_i] = 1;
                    }

                    // Search for a third particle in the same cell
                    for (int k = j + 1; k < current.count; k++) {
                        int idx_k = current.indices[k];
                        if (marked_for_removal[idx_k]) continue;

                        double dx_k = par[idx_k].x - par[idx_j].x;
                        double dy_k = par[idx_k].y - par[idx_j].y;
                        double dist2_k = dx_k * dx_k + dy_k * dy_k;

                        if (dist2_k <= EPSILON2) {
                            marked_for_removal[idx_k] = 1;
                            collision_group[idx_k] = 1;
                            group_size++;

                            if (group_size == 3) {
                                break; // Stop at 3-particle collision
                            }
                        }
                    }
                    break; // Stop after finding a collision for idx_i
                }
            }

            if (group_size > 1) {
                (*collision_count)++; // Count the entire chain as ONE collision event
            }
        }
    }

    // Compact particle array (remove collided particles)
    long long new_n_part = 0;
    for (long long i = 0; i < *n_part; i++) {
        if (!marked_for_removal[i]) {
            par[new_n_part++] = par[i]; // Keep non-removed particles
        }
    }
    *n_part = new_n_part;

    free(marked_for_removal);
    free(collision_group);
}



// Run one simulation time step using spatial partitioning
void run_time_step(particle_t *par, long long *n_part, long ncside, double side, double cell_side, long long *collision_count) {
    // 1. Combined cell assignment and cell list build.
    cell_t *cells = assign_particles_and_build_cells(par, *n_part, ncside, cell_side);
    // 3. Compute forces using spatial partitioning: same-cell and adjacent cells
    calculate_forces(par, cells, n_part, ncside, side);
    // 4. Update positions and velocities
    update_positions_and_velocities(par, *n_part, side);
    // 5. Detect collisions (check only within the same cell)
    detect_collisions(cells, par, ncside, n_part, collision_count);
    // Free cell lists for this time step
    free_cell_lists(cells, ncside);
}


///////////////////////////////////////
// Main Function
///////////////////////////////////////

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <seed> <side> <grid_size> <num_particles> <time_steps>\n", argv[0]);
        return 1;
    }

    double exec_time;
    long seed = atol(argv[1]);         // Seed
    double side = atof(argv[2]);       // Domain side length
    long ncside = atol(argv[3]);       // Number of cells per side
    long long n_part = atoll(argv[4]);   // Number of particles
    long long time_steps = atoll(argv[5]); // Number of time steps
    
    particle_t *particles_arr = malloc(n_part * sizeof(particle_t));
    if (!particles_arr) {
        fprintf(stderr, "Memory allocation failed for particles.\n");
        return 1;
    }
    const double cell_side = side / ncside;
    init_particles(seed, side, ncside, n_part, particles_arr);
    exec_time = -omp_get_wtime(); // (You can also use clock() for a purely sequential timing)

    long long collision_count = 0;
    for (long long t = 0; t < time_steps; t++) {
        printf("Time step %lld\n", t);
        run_time_step(particles_arr, &n_part, ncside, side, cell_side, &collision_count);
    }

    exec_time += omp_get_wtime();

    printf("\nParticle 0: %.3f %.3f\n", particles_arr[0].x, particles_arr[0].y);
    printf("Collisions: %lld\n", collision_count);
    fprintf(stderr, "%.1fs\n", exec_time);

    free(particles_arr);
    return 0;
}

void print_particles(particle_t *par, long long n_part) {
    for (long long i = 0; i < n_part; i++) {
        printf("Particle %lld: X:%f Y:%f VX:%f VY:%f M:%f X_CELL:%d Y_CELL:%d\n",
               i, par[i].x, par[i].y, par[i].vx, par[i].vy, par[i].m, par[i].x_cell, par[i].y_cell);
    }
}
void print_forces(particle_t *par, long long n_part) {
    for (long long i = 0; i < n_part; i++) {
        printf("Particle %lld: AX:%f AY:%f \n", i, par[i].ax, par[i].ay);
    }
}
