#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "header-seq.h"

#define _USE_MATH_DEFINES
#define G 6.67408e-11
#define EPSILON2 (0.005 * 0.005)
#define DELTAT 0.1
#define pi = 4 * atan(1.0);

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
    return 0.5 + 0.2328306e-09 * (seed_in + (int)seed);
}
double rnd_normal01()
{
    double u1, u2, z, result;
    do
    {
        u1 = rnd_uniform01();
        u2 = rnd_uniform01();
        z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
        result = 0.5 + 0.15 * z; // Shift mean to 0.5 and scale
    } while (result < 0 || result >= 1);
    return result;
}

void init_particles(long userseed, double side, long ncside, long long n_part, particle_t *par)
{
    double (*rnd01)() = rnd_uniform01;
    long long i;

    if (userseed < 0)
    {
        rnd01 = rnd_normal01;
        userseed = -userseed;
    }

    init_r4uni(userseed);

    for (i = 0; i < n_part; i++)
    {
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

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        fprintf(stderr, "Usage: %s <seed> <side> <grid_size> <num_particles> <time_steps>\n", argv[0]);
        return 1;
    }

    double exec_time;
    long seed = atol(argv[1]);             // Seed
    double side = atof(argv[2]);           // Domain side length
    long ncside = atol(argv[3]);           // Number of cells per side
    long long n_part = atoll(argv[4]);     // Number of particles
    long long time_steps = atoll(argv[5]); // Number of time steps

    // Allocate memory for particles and cells
    particle_t *particles_arr = malloc(n_part * sizeof(particle_t));
    if (!particles_arr) {
        fprintf(stderr, "Memory allocation failed for particles.\n");
        return 1;
    }

    // Calculate cell side length
    const double cell_side = side / ncside;
    const double inv_cell_side = 1.0 / cell_side;
    // Calculate total number of cells
    long total_cells = ncside * ncside;

    init_particles(seed, side, ncside, n_part, particles_arr);

    cell_t *cells = init_cells(particles_arr, total_cells, n_part, ncside, cell_side);
    
    exec_time = -omp_get_wtime();

    long long collision_count = 0;
    
    for (long long t = 0; t < time_steps; t++)
    {
        run_time_step(particles_arr, &n_part, ncside, side, cell_side, inv_cell_side, total_cells, &collision_count, t, cells);
    }

    exec_time += omp_get_wtime();

    printf("%.3f %.3f\n", particles_arr[0].x, particles_arr[0].y);
    printf("%lld\n", collision_count);
    fprintf(stderr, "%.1fs\n", exec_time);

    free_cell_lists(cells, ncside, total_cells);
    free(particles_arr);
    return 0;
}

cell_t *init_cells(particle_t *par, long total_cells, long long n_part, long ncside, double cell_side)
{
    // Step 1: Count initial particles per cell
    long long *cell_counts = calloc(total_cells, sizeof(long long));
    if (!cell_counts) {
        fprintf(stderr, "Memory allocation failed for cell_counts.\n");
        free(par);
        return NULL;
    }

    for (long long i = 0; i < n_part; i++) {
        int x_cell = (int)(par[i].x / cell_side);
        int y_cell = (int)(par[i].y / cell_side);
        // Ensure indices are within bounds
        x_cell = (x_cell + ncside) % ncside;
        y_cell = (y_cell + ncside) % ncside;
        int cell_index = y_cell * ncside + x_cell;
        cell_counts[cell_index]++;
    }

    // Step 2: Allocate memory for cells
    cell_t *cells = malloc(total_cells * sizeof(cell_t));
    if (!cells) {
        fprintf(stderr, "Memory allocation failed for cells.\n");
        free(cell_counts);
        free(par);
        return NULL;
    }

    // Define a buffer for extra capacity
    const long long buffer = 1000;

    // Step 3: Set capacities and allocate indices arrays
    for (long i = 0; i < total_cells; i++) {
        cells[i].capacity = cell_counts[i] + buffer;
        cells[i].indices = malloc(cells[i].capacity * sizeof(int));
        if (!cells[i].indices) {
            fprintf(stderr, "Memory allocation failed for cell indices at cell %ld.\n", i);
            // Free previously allocated memory to avoid leaks
            for (long j = 0; j < i; j++) {
                free(cells[j].indices);
            }
            free(cells);
            free(cell_counts);
            free(par);
            return NULL;
        }
        cells[i].count = 0;  // Initialize count to 0; will be incremented later
        cells[i].x = 0;
        cells[i].y = 0;
        cells[i].m = 0;
    }

    // Step 4: Populate indices arrays
    for (long long i = 0; i < n_part; i++) {
        int x_cell = (int)(par[i].x / cell_side);
        int y_cell = (int)(par[i].y / cell_side);
        par[i].x_cell = x_cell;
        par[i].y_cell = y_cell;
        x_cell = (x_cell + ncside) % ncside;
        y_cell = (y_cell + ncside) % ncside;
        int cell_index = y_cell * ncside + x_cell;
        cells[cell_index].indices[cells[cell_index].count] = i;
        cells[cell_index].count++;  // Increment the count
    }

    // Step 5: Free temporary array and return
    free(cell_counts);
    return cells;
}

void free_cell_lists(cell_t *cells, long ncside, long total_cells)
{
    for (long i = 0; i < total_cells; i++)
    {
        free(cells[i].indices);
    }
    free(cells);
}

///////////////////////////////////////
// Run time pipeline
///////////////////////////////////////

// Run one simulation time step using spatial partitioning
void run_time_step(particle_t *par, long long *n_part, long ncside, double side, double cell_side, double inv_cell_side, long total_cells, long long *collision_count, long long timestep, cell_t *cells)
{
    // 1. Combined cell assignment and cell list build.
    assign_particles_and_build_cells(par, *n_part, ncside, cell_side, inv_cell_side, total_cells, cells);
    // 2. Compute forces using spatial partitioning: same-cell and adjacent cells
    calculate_forces(par, cells, n_part, ncside, side, total_cells);
    // 3. Update positions and velocities
    update_positions_and_velocities(par, cells, *n_part, ncside, side, inv_cell_side, total_cells);
    // 4. Detect collisions (check only within the same cell)
    detect_collisions(cells, par, ncside, n_part, collision_count, total_cells, timestep);
}

///////////////////////////////////////
// Spatial Partitioning Structures and Functions
///////////////////////////////////////

// Combined function: assign each particle to a cell and build cell lists.
void assign_particles_and_build_cells(particle_t *par, long long n_part, long ncside, double cell_size, double inv_cell_size, long total_cells, cell_t *cells)
{
    // Reset cell COMs and masses
    for (long i = 0; i < total_cells; i++)
    {
        cells[i].x = 0;
        cells[i].y = 0;
        cells[i].m = 0;

        for (long long j = 0; j < cells[i].count; j++)
        {
            int idx = cells[i].indices[j];
            if (par[idx].m != 0) // Only consider non-zero-mass particles
            {
                cells[i].x += par[idx].x * par[idx].m;
                cells[i].y += par[idx].y * par[idx].m;
                cells[i].m += par[idx].m;
            }
        }

        if (cells[i].m > 0)
        {
            cells[i].x /= cells[i].m;
            cells[i].y /= cells[i].m;
        }
    }
}

///////////////////////////////////////
// Simulation Helper Functions
///////////////////////////////////////

// Calculate forces on each particle using spatial partitioning,
void calculate_forces(particle_t *par, cell_t *cells, long long *n_part, long ncside, double side, long total_cells)
{

    // 1. Compute same-cell interactions
    for (long cell = 0; cell < total_cells; cell++)
    {
        cell_t current = cells[cell];
        for (int a = 0; a < current.count; a++)
        {
            int i = current.indices[a];
            for (int b = a + 1; b < current.count; b++)
            {
                int j = current.indices[b];
                double dx = par[j].x - par[i].x;
                double dy = par[j].y - par[i].y;
                double dist2 = dx * dx + dy * dy;
                double inv_r = 1.0 / sqrt(dist2);
                double force = G * (par[i].m * par[j].m) / dist2;
                double fx = force * dx * inv_r;
                double fy = force * dy * inv_r;
                par[i].ax += fx / par[i].m;
                par[i].ay += fy / par[i].m;
                par[j].ax -= fx / par[j].m;
                par[j].ay -= fy / par[j].m;
            }
        }
    }

    // 2. Compute forces from neighboring cell COMs
    for (long long i = 0; i < *n_part; i++)
    {
        if (par[i].m == 0) continue; // Skip zero-mass particles
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
                par[i].ax += fx_cm / par[i].m;
                par[i].ay += fy_cm / par[i].m;
            }
        }
    }
}

// Combined function to update positions and velocities in one loop.
void update_positions_and_velocities(particle_t *par, cell_t *cells, long long n_part, long ncside, double side, double inv_cell_size, long total_cells)
{
    for (long long i = 0; i < n_part; i++)
    {
        if (par[i].m == 0) continue; // Skip zero-mass particles
        int prev_x_cell = par[i].x_cell;
        int prev_y_cell = par[i].y_cell;
        int prev_cell_index = prev_y_cell * ncside + prev_x_cell;
        par[i].x += par[i].vx * DELTAT + 0.5 * par[i].ax * DELTAT * DELTAT;
        par[i].y += par[i].vy * DELTAT + 0.5 * par[i].ay * DELTAT * DELTAT;
        par[i].x = fmod(par[i].x, side);
        if (par[i].x < 0) par[i].x += side;
        par[i].y = fmod(par[i].y, side);
        if (par[i].y < 0) par[i].y += side;
        int new_x_cell = (int)floor(par[i].x * inv_cell_size);
        int new_y_cell = (int)floor(par[i].y * inv_cell_size);
        if (new_x_cell >= ncside) new_x_cell = ncside - 1;
        if (new_y_cell >= ncside) new_y_cell = ncside - 1;
        int new_cell_index = new_y_cell * ncside + new_x_cell;
        if (new_cell_index != prev_cell_index)
        {
            int *indices = cells[prev_cell_index].indices;
            int count = cells[prev_cell_index].count;
            for (int j = 0; j < count; j++)
            {
                if (indices[j] == i)
                {
                    indices[j] = indices[count - 1];
                    cells[prev_cell_index].count--;
                    break;
                }
            }
            par[i].x_cell = new_x_cell;
            par[i].y_cell = new_y_cell;
            cells[new_cell_index].indices[cells[new_cell_index].count++] = i;
        }
        par[i].vx += par[i].ax * DELTAT;
        par[i].vy += par[i].ay * DELTAT;
        // Reset accelerations for the next time step
        par[i].ax = 0;
        par[i].ay = 0;
    }
}


void detect_collisions(cell_t *cells, particle_t *par, long ncside, long long *n_part, long long *collision_count, long total_cells, long long timestep)
{
    for (long cell = 0; cell < total_cells; cell++)
    {
        if (cells[cell].count < 2) continue;

        // Detect collisions and set mass to zero
        for (int i = 0; i < cells[cell].count; i++)
        {
            int idx_i = cells[cell].indices[i];
            if (par[idx_i].m == 0) continue; // Skip already zero-mass particles

            int collision_detected = 0;
            for (int j = i + 1; j < cells[cell].count; j++)
            {
                int idx_j = cells[cell].indices[j];
                if (par[idx_j].m == 0) continue; // Skip already zero-mass particles

                double dx = par[idx_j].x - par[idx_i].x;
                double dy = par[idx_j].y - par[idx_i].y;
                double dist2 = dx * dx + dy * dy;

                if (dist2 < EPSILON2)
                {
                    par[idx_i].m = 0;  // Set mass to zero
                    par[idx_j].m = 0;  // Set mass to zero
                    if (!collision_detected)
                    {
                        (*collision_count)++;
                        collision_detected = 1;
                    }
                }
            }
        }

        // Compact cell index list to exclude zero-mass particles
        int new_count = 0;
        for (int i = 0; i < cells[cell].count; i++)
        {
            int idx = cells[cell].indices[i];
            if (par[idx].m != 0)
            {
                cells[cell].indices[new_count++] = idx;
            }
        }
        cells[cell].count = new_count;
    }
}

void print_particles(particle_t *par, long long n_part)
{
    for (int i = 0; i < n_part; i++)
    {
        printf("Particle %d: mass=%.6f x=%.6f y=%.6f vx=%.6f vy=%.6f\n X_cell: %d Y_cell: %d\n",
               i, par[i].m, par[i].x, par[i].y,
               par[i].vx, par[i].vy, par[i].x_cell, par[i].y_cell);
    }
}

void print_cells(cell_t *cells, long ncside)
{
    for (int i = 0; i < ncside * ncside; i++)
    {
        printf("Cell %d x: %.6f y: %.6f m: %.6f\n", i, cells[i].x, cells[i].y, cells[i].m);
    }
}

void print_forces(particle_t *par, long long n_part)
{
    for (long long i = 0; i < n_part; i++)
    {
        printf("Particle %lld: AX:%f AY:%f \n", i, par[i].ax, par[i].ay);
    }
}