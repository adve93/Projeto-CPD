#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "init_particles.h"

#define _USE_MATH_DEFINES
#define G 6.67408e-11
#define EPSILON2 (0.005*0.005)
#define DELTAT 0.1
#define M_PI 3.14159265358979323846

///////////////////////////////////////Professor Function declarations////////////////////////////////////////
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

void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par)
{
    double (*rnd01)() = rnd_uniform01;
    long long i;

    if(seed < 0) {
        rnd01 = rnd_normal01;
        seed = -seed;
    }
    
    init_r4uni(seed);

    for(i = 0; i < n_part; i++) {
        par[i].x = rnd01() * side;
        par[i].y = rnd01() * side;
        par[i].vx = (rnd01() - 0.5) * side / ncside / 5.0;
        par[i].vy = (rnd01() - 0.5) * side / ncside / 5.0;

        par[i].m = rnd01() * 0.01 * (ncside * ncside) / n_part / G * EPSILON2;
        par[i].removed = 0;
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////Main Code//////////////////////////////////////////////////////


int main(int argc, char *argv[]) {
    
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <seed> <side> <grid_size> <num_particles> <time_steps>\n", argv[0]);
        return 1;
    }

    double exec_time;
    long seed = atol(argv[1]);         // Seed
    double side = atof(argv[2]);       // Simulation domain side length
    long ncside = atol(argv[3]);       // Number of cells per side
    long long n_part = atoll(argv[4]);   // Number of particles
    long long time_steps = atoll(argv[5]); // Number of time steps
    
    particle_t *restrict particles_arr = (particle_t *)malloc(n_part * sizeof(particle_t));
    if (!particles_arr) {
        fprintf(stderr, "Memory allocation failed for particles.\n");
        return 1;
    }
//Vtune
    const double cell_side = side / ncside;

    com_t *restrict com = (com_t *)malloc(ncside * ncside * sizeof(com_t));
    if (!com) {
        fprintf(stderr, "Memory allocation failed for center of mass array.\n");
        free(particles_arr);
        return 1;
    }

    init_particles(seed, side, ncside, n_part, particles_arr);
    exec_time = -omp_get_wtime();

    long long collision_count = 0;
    for (long long t = 0; t < time_steps; t++) {
        printf("Time step %lld\n", t);
        // Calculate cell indices for particles
        particle_cell(particles_arr, n_part, cell_side);
        // Calculate center of mass for each cell
        calculate_com(particles_arr, com, n_part, ncside);
        // Calculate forces and update accelerations
        calculate_forces(particles_arr, com, n_part, ncside, side);
        // Update positions and velocities (order according to project spec)
        update_positions(particles_arr, n_part, side);
        update_velocities(particles_arr, n_part);
        // Detect collisions
        detect_collisions(particles_arr, &n_part, &collision_count);
    }

    printf("\nParticle 0: %.3f %.3f\n", particles_arr[0].x, particles_arr[0].y);
    printf("%lld\n", collision_count);
    exec_time += omp_get_wtime();
    fprintf(stderr, "%.1fs\n", exec_time);

    free(particles_arr);
    free(com);
    return 0;
}   

// Calculate the center of mass for each cell
void calculate_com(particle_t *restrict par, com_t *restrict com, long long n_part, long ncside) {
    long total_cells = ncside * ncside;
    for (long i = 0; i < total_cells; i++) {
        com[i].x = 0;
        com[i].y = 0;
        com[i].m = 0;
    }
    for (long long i = 0; i < n_part; i++) {
        int cell_index = par[i].y_cell * ncside + par[i].x_cell;
        com[cell_index].x += par[i].x * par[i].m;
        com[cell_index].y += par[i].y * par[i].m;
        com[cell_index].m += par[i].m;
    }
    for (long i = 0; i < total_cells; i++) {
        if (com[i].m > 0) {
            com[i].x /= com[i].m;
            com[i].y /= com[i].m;
        }
    }
}

// Determine the cell for each particle
void particle_cell(particle_t *restrict par, long long n_part, double cell_size) {
    for (long long i = 0; i < n_part; i++) {
        par[i].x_cell = (int)(par[i].x / cell_size);
        par[i].y_cell = (int)(par[i].y / cell_size);
    }
}

// Calculate forces on each particle from particles in the same cell and adjacent cells (COM)
void calculate_forces(particle_t *restrict par, com_t *restrict com, long long n_part, long ncside, double side) {
    double half_side = side / 2.0;
    for (long long i = 0; i < n_part; i++) {
        double fx = 0.0, fy = 0.0;
        int x_cell = par[i].x_cell;
        int y_cell = par[i].y_cell;
        
        // Step 1: Forces from particles in the same cell
        for (long long j = 0; j < n_part; j++) {
            if (i == j) continue;
            if (par[j].x_cell == x_cell && par[j].y_cell == y_cell) {
                double dx = par[j].x - par[i].x;
                double dy = par[j].y    - par[i].y;
                // Apply toroidal wrapping for particles
                if (dx > half_side) dx -= side;
                if (dx < -half_side) dx += side;
                if (dy > half_side) dy += side;  // Use your convention for Y
                if (dy < -half_side) dy -= side;
                double dist2 = dx * dx + dy * dy;  // EPSILON2 omitted for speed if not critical
                double force = (G * par[i].m * par[j].m) / dist2;
                double r = sqrt(dist2);
                fx += force * (dx / r);
                fy += force * (dy / r);
            }
        }
        
        // Step 2: Forces from centers of mass of adjacent cells (excluding own cell)
        for (int dxc = -1; dxc <= 1; dxc++) {
            for (int dyc = -1; dyc <= 1; dyc++) {
                if (dxc == 0 && dyc == 0) continue;
                int nx = (x_cell + dxc + ncside) % ncside;
                int ny = (y_cell + dyc + ncside) % ncside;
                int cell_index = ny * ncside + nx;
                if (com[cell_index].m == 0) continue;
                
                double dx_cm = com[cell_index].x - par[i].x;
                double dy_cm = com[cell_index].y - par[i].y;
                // Apply toroidal wrapping for centers of mass
                if (dx_cm > half_side) dx_cm -= side;
                if (dx_cm < -half_side) dx_cm += side;
                if (dy_cm > half_side) dy_cm += side;  // Use your convention for Y
                if (dy_cm < -half_side) dy_cm -= side;
                double dist2_cm = dx_cm * dx_cm + dy_cm * dy_cm;
                double force_cm = (G * par[i].m * com[cell_index].m) / dist2_cm;
                double r_cm = sqrt(dist2_cm);
                fx += force_cm * (dx_cm / r_cm);
                fy += force_cm * (dy_cm / r_cm);
            }
        }
        par[i].ax = fx / par[i].m;
        par[i].ay = fy / par[i].m;
    }
}

// Update particle velocities using precomputed accelerations
void update_velocities(particle_t *restrict par, long long n_part) {
    for (long long i = 0; i < n_part; i++) {
        par[i].vx += par[i].ax * DELTAT;
        par[i].vy += par[i].ay * DELTAT;
    }
}

// Update particle positions using updated velocities and apply toroidal wrapping
void update_positions(particle_t *restrict par, long long n_part, double side) {
    for (long long i = 0; i < n_part; i++) {
        par[i].x += par[i].vx * DELTAT;
        par[i].y += par[i].vy * DELTAT;
        if (par[i].x < 0) par[i].x += side;
        if (par[i].x >= side) par[i].x -= side;
        if (par[i].y < 0) par[i].y += side;
        if (par[i].y >= side) par[i].y -= side;
    }
}

// Collision detection: mark colliding particles and compact array
void detect_collisions(particle_t *restrict par, long long *n_part, long long *collision_count) {
    long long i, j;
    for (i = 0; i < *n_part; i++) {
        if (par[i].removed) continue;
        for (j = i + 1; j < *n_part; j++) {
            if (par[j].removed) continue;
            double dx = par[j].x - par[i].x;
            double dy = par[j].y - par[i].y;
            double dist2 = dx * dx + dy * dy;
            if (dist2 < EPSILON2) {
                par[i].removed = 1;
                par[j].removed = 1;
                (*collision_count)++;
            }
        }
    }
    long long new_n_part = 0;
    for (i = 0; i < *n_part; i++) {
        if (!par[i].removed) {
            par[new_n_part] = par[i];
            new_n_part++;   
        }
    }
    *n_part = new_n_part;
}

// The following print functions are for testing and debugging and can be omitted in the final build
void print_com(com_t *restrict com, long ncside) {
    for (long i = 0; i < ncside * ncside; i++) {
        printf("COM %ld: X:%f Y:%f M:%f\n", i, com[i].x, com[i].y, com[i].m);
    }
}

void print_particles(particle_t *restrict par, long long n_part) {
    for (long long i = 0; i < n_part; i++) {
        printf("Particle %lld: X:%f Y:%f VX:%f VY:%f M:%f X_CELL:%d Y_CELL:%d\n",
               i, par[i].x, par[i].y, par[i].vx, par[i].vy, par[i].m, par[i].x_cell, par[i].y_cell);
    }
}

void print_forces(particle_t *restrict par, long long n_part) {
    for (long long i = 0; i < n_part; i++) {
        printf("Particle %lld: AX:%f AY:%f \n", i, par[i].ax, par[i].ay);
    }
}