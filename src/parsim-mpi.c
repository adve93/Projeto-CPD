#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "header-mpi.h"
#include <string.h>

#define M_PI 3.14159265358979323846
#define _USE_MATH_DEFINES
#define G 6.67408e-11
#define EPSILON2 (0.005 * 0.005)
#define DELTAT 0.1


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
        par[i].global_id = i;
    }
}

//////////////////////////////////////
// Main Function
//////////////////////////////////////

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);   // Initialize MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc != 6) {
        if(rank == 0) {
            fprintf(stderr, "Usage: %s <seed> <side> <grid_size> <num_particles> <time_steps>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Process 0 will read the parameters and initialize the full particle array.
    long seed = atol(argv[1]);
    double side = atof(argv[2]);
    int ncside = atoi(argv[3]);
    long long n_part_total = atoll(argv[4]);
    long long time_steps = atoll(argv[5]);
    const double cell_side = side / ncside;
    const double inv_cell_side = 1.0 / cell_side;
    const long total_cells = ncside * ncside;

    cell_t *local_cells = NULL;

    ////////////////////////////////////////////////////////////
    ////////////Initialize And Distribute Cells/////////////////
    ////////////////////////////////////////////////////////////
    int start_row, end_row;
    get_local_domain(rank, size, ncside, &start_row, &end_row);
    int local_rows = end_row - start_row + 1;
    // Each process manages local_rows * ncside cells.
    long local_total_cells = local_rows * ncside;

    //print_process_cell_assignment(rank, size, ncside, side, start_row, end_row);

    // Allocate local cells (only for the rows this process owns).
    local_cells = malloc(local_total_cells * sizeof(cell_t));
    if (!local_cells) {
        fprintf(stderr, "Process %d: Memory allocation failed for local cells.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Initialize local cell arrays (similar to init_cells but only for local domain).
    for (long i = 0; i < local_total_cells; i++) {
        local_cells[i].capacity = 1000;
        local_cells[i].count = 0;
        local_cells[i].indices = malloc(local_cells[i].capacity * sizeof(int));
        if (!local_cells[i].indices) {
            fprintf(stderr, "Process %d: Memory allocation failed for cell indices.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        local_cells[i].x = 0;
        local_cells[i].y = 0;
        local_cells[i].m = 0;
    }


    ////////////////////////////////////////////////////////////
    ////////////Initialize And Distribute Particles/////////////
    ////////////////////////////////////////////////////////////

    particle_t *local_particles = NULL;
    long long local_n_part = 0;
    int truesize = 0;

    if(rank == 0) {
        truesize = size;
        // Process 0 initializes the full particle array.
        particle_t *particles_arr = malloc(n_part_total * sizeof(particle_t));
        if (!particles_arr) {
            fprintf(stderr, "Memory allocation failed for particles.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        init_particles(seed, side, ncside, n_part_total, particles_arr);

        // Decide which process will manage which particles.
        // We assign a particle to the process responsible for the cell its (x,y) falls in.
        // Will have to reorganize the particle array to match the process's local domain.
        int *send_counts = calloc(size, sizeof(int));
        for (long long i = 0; i < n_part_total; i++) {
            int x_cell = (int)(particles_arr[i].x * inv_cell_side);
            int y_cell = (int)(particles_arr[i].y * inv_cell_side);
            int owner;
            int start, end;

            for (owner = 0; owner < size; owner++) {
                get_local_domain(owner, size, ncside, &start, &end);
                if (y_cell >= start && y_cell <= end) {
                    send_counts[owner]++;
                    break;
                }
            }
        }

        // Compute displacements.
        int *displs = malloc(size * sizeof(int));
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + send_counts[i - 1];
        }
        // Allocate buffer for particles for each process.
        particle_t *all_particles = malloc(n_part_total * sizeof(particle_t));
        // Copy particles into the appropriate positions.
        // We will use a temporary array to hold the counts for each process.
        int *temp_counts = calloc(size, sizeof(int));
        for (long long i = 0; i < n_part_total; i++) {
            int x_cell = (int)(particles_arr[i].x * inv_cell_side);
            int y_cell = (int)(particles_arr[i].y * inv_cell_side);
            int owner;
            int start, end;
            for (owner = 0; owner < size; owner++) {
                get_local_domain(owner, size, ncside, &start, &end);
                if (y_cell >= start && y_cell <= end) {
                    int pos = displs[owner] + temp_counts[owner];
                    all_particles[pos] = particles_arr[i];
                    temp_counts[owner]++;
                    break;
                }
            }
        }
        // Now, send the counts to each process (including self).
        // First, send local_n_part count to each process.
        for (int i = 0; i < size; i++) {
            if (i == 0) {
                local_n_part = send_counts[0];
                local_particles = malloc(local_n_part * sizeof(particle_t));
                if (!local_particles) {
                    fprintf(stderr, "Memory allocation failed for local_particles at process 0.\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                memcpy(local_particles, all_particles, local_n_part * sizeof(particle_t));
            } else {
                if(send_counts[i] == 0) {
                    truesize--;
                }
                MPI_Send(&send_counts[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(all_particles + displs[i], send_counts[i] * sizeof(particle_t), MPI_BYTE, i, 1, MPI_COMM_WORLD);
            }
        }
        for(int i =1; i < size; i++) {
            MPI_Send(&truesize, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
        }
        free(send_counts);
        free(displs);
        free(temp_counts);
        free(all_particles);
        free(particles_arr);
    } else {
        // Other processes receive the number of particles and their data.
        MPI_Recv(&local_n_part, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_particles = malloc(local_n_part * sizeof(particle_t));
        if (!local_particles) {
            fprintf(stderr, "Process %d: Memory allocation failed for local_particles.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Recv(local_particles, local_n_part * sizeof(particle_t), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&truesize, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    for (long long i = 0; i < local_n_part; i++) {
        int x_cell = (int)(local_particles[i].x * inv_cell_side);
        int y_cell = (int)(local_particles[i].y * inv_cell_side);
        // Only assign particles that belong in this process’s rows. <- Suppostamente ja verificamos isto mas bom failsafe
        if (y_cell >= start_row && y_cell <= end_row) {
            // Map global cell (x_cell, y_cell) to local index:
            int local_row = y_cell - start_row;
            int local_cell_index = local_row * ncside + x_cell;
            local_particles[i].x_cell = x_cell;
            local_particles[i].y_cell = y_cell;
            // Make sure there is capacity.
            if (local_cells[local_cell_index].count == local_cells[local_cell_index].capacity) {
                local_cells[local_cell_index].capacity *= 2;
                local_cells[local_cell_index].indices = realloc(local_cells[local_cell_index].indices, local_cells[local_cell_index].capacity * sizeof(int));
                if (!local_cells[local_cell_index].indices) {
                    fprintf(stderr, "Process %d: Memory reallocation failed for cell indices.\n", rank);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            local_cells[local_cell_index].indices[local_cells[local_cell_index].count++] = i;
        }
    }
    //print_local_particles(rank, size, local_particles, local_n_part, inv_cell_side);

    ////////////////////////////////////////////////////////////
    /////////////////////Main Routine///////////////////////////
    ////////////////////////////////////////////////////////////
    for (long long t = 0; t < time_steps; t++) {

        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0) {
            printf("Time step %lld\n", t);
        }

        // --- 1. Build/Update Local Cells ---
        build_com(local_particles, local_n_part, ncside, cell_side, inv_cell_side, local_total_cells, local_cells);

        // --- 2. Exchange Ghost Cell Information ---
        MPI_Barrier(MPI_COMM_WORLD);  
        cell_t *ghost_upper = NULL;
        cell_t *ghost_lower = NULL;
        if(local_total_cells > 0) {
            ghost_upper = malloc(ncside * sizeof(cell_t));
            ghost_lower = malloc(ncside * sizeof(cell_t));
            exchange_ghost_cells(start_row, end_row, rank, size, local_rows, ncside, truesize, MPI_COMM_WORLD, local_cells, ghost_upper, ghost_lower);
            
        }
        

        // --- 3. Calculate Forces ---
        // Call a modified version of calculate_forces that uses local_cells and local_particles.
        calculate_forces(local_particles, local_cells, local_n_part, ncside, side, start_row, end_row, local_total_cells, ghost_lower, ghost_upper);
        free(ghost_upper);
        free(ghost_lower);
        
        // --- 4. Update Positions and Velocities ---
        particle_t *ghost_par = malloc(local_n_part * sizeof(particle_t));
        long ghost_par_count = 0;
        update_positions_and_velocities(local_particles, ghost_par, local_cells, start_row, end_row, ghost_par_count, local_n_part, ncside, side, inv_cell_side, local_total_cells, rank);
        free(ghost_par);
        //print_local_particles(rank, size, local_particles, local_n_part, inv_cell_side);

        // --- 5. Exchange Particles ---
        exchange_particles(rank, size, local_rows, ncside, truesize, MPI_COMM_WORLD, local_particles, ghost_par, ghost_par_count);
        /*
        // --- 5. Detect Collisions ---
        // Detect collisions locally (only within cells that the process owns).
        static long long local_collision_count = 0;
        detect_collisions(local_cells, local_particles, ncside, &local_n_part, &local_collision_count, local_total_cells, t);

        // --- 6. Migrate Particles Across Process Boundaries ---
        // After updating positions, some particles might have moved out of the local domain.
        // Pack such particles and send them to the appropriate process.
        // For instance, a particle with y_cell < start_row should be sent to process rank-1,
        // and one with y_cell > end_row should be sent to process rank+1.
        // You can use MPI_Sendrecv or MPI_Isend/MPI_Irecv to exchange these particles.
        // After migration, update local_particles and local_n_part accordingly.
        // (For brevity, the detailed migration code is omitted here.)
        
        // Synchronize at the end of the time step.
        */
        MPI_Barrier(MPI_COMM_WORLD);
        
    }

    for (long i = 0; i < local_total_cells; i++) {
        free(local_cells[i].indices);
    }
    free(local_cells);
    free(local_particles);

    MPI_Finalize();
}

///////////////////////////////////////
// Functions
///////////////////////////////////////

void build_com(particle_t *par, long long n_part, long ncside, double cell_size, double inv_cell_size, long total_cells, cell_t *cells)
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
            if(par[idx].removed) continue;
            cells[i].x += par[idx].x * par[idx].m;
            cells[i].y += par[idx].y * par[idx].m;
            cells[i].m += par[idx].m;
        }

        if (cells[i].m > 0)
        {
            cells[i].x /= cells[i].m;
            cells[i].y /= cells[i].m;
        }
    }
}

void calculate_forces(particle_t *par, cell_t *cells, long long n_part, long ncside, double side, int start_row, int end_row,  long total_cells, cell_t *ghost_lower, cell_t *ghost_upper)
{

    // Reset accelerations
    for (long long i = 0; i < n_part; i++)
    {
        par[i].ax = 0;
        par[i].ay = 0;
    }

    // 1. Compute same-cell interactions using each unique pair only once.
    for (long cell = 0; cell < total_cells; cell++)
    {
        cell_t current = cells[cell];
        for (int a = 0; a < current.count; a++)
        {
            int i = current.indices[a];
            for (int b = a + 1; b < current.count; b++)
            {      
                if(par[i].removed || par[current.indices[b]].removed) continue; // Skip removed particles
                int j = current.indices[b];

                double dx = par[j].x - par[i].x;
                double dy = par[j].y - par[i].y;

                double dist2 = dx * dx + dy * dy;

                double inv_r = 1.0 / sqrt(dist2);
                double force = G * ((par[i].m * par[j].m) / dist2);
                double fx = force * dx * inv_r; // Actual force component in x direction
                double fy = force * dy * inv_r; // Actual force component in y direction

                par[i].ax += fx / par[i].m;
                par[i].ay += fy / par[i].m;
                par[j].ax -= fx / par[j].m;
                par[j].ay -= fy / par[j].m;

                // Debug print for same-cell interactions
                /* if(i == 0) {
                    printf("P%d/P%d mag: %.6f fx: %.6f fy: %.6f\n", i, j, force, fx, fy);
                } */
            }
        }
    }

    // 2. Compute forces from neighboring cell COMs
    for (long long i = 0; i < n_part; i++)
    {
        if(par[i].removed) continue; // Skip removed particles
        int x_cell = par[i].x_cell;
        int y_cell = par[i].y_cell;
        int offsets[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
        for (int k = 0; k < 8; k++)
        {
            int dxc = offsets[k][0];
            int dyc = offsets[k][1];

            // Get neighbor cell with toroidal wrapping
            int nx = (x_cell + dxc + ncside) % ncside;
            int ny = ((y_cell - start_row) + dyc + ncside) % ncside;
            int neighbor_index = ny * ncside + nx;
            if (cells[neighbor_index].m == 0)
                continue; // Skip empty cell

            double dx_cm = 0;
            double dy_cm = 0;

            // Compute raw displacements.
            if(ny == start_row) {
                dx_cm = ghost_upper[nx].x - par[i].x;
                dy_cm = ghost_upper[nx].y - par[i].y;
            } else if(ny == end_row) {
                dx_cm = ghost_lower[nx].x - par[i].x;
                dy_cm = ghost_lower[nx].y - par[i].y;
            } else {
                dx_cm = cells[neighbor_index].x - par[i].x;
                dy_cm = cells[neighbor_index].y - par[i].y;
            }

            // Compute cell offset differences (normalized to -1, 0, or 1)
            int diff_x = nx - x_cell;
            if (diff_x > ncside / 2)
                diff_x -= ncside;
            if (diff_x < -ncside / 2)
                diff_x += ncside;

            int diff_y = ny - y_cell;
            if (diff_y > ncside / 2)
                diff_y -= ncside;
            if (diff_y < -ncside / 2)
                diff_y += ncside;

            // Adjust displacements based on the cell offset.
            if (diff_x > 0 && dx_cm < 0)
            {
                dx_cm += side;
            }
            else if (diff_x < 0 && dx_cm > 0)
            {
                dx_cm -= side;
            }

            if (diff_y > 0 && dy_cm < 0)
            {
                dy_cm += side;
            }
            else if (diff_y < 0 && dy_cm > 0)
            {
                dy_cm -= side;
            }

            // Compute squared distance, force, etc.
            double dist2_cm = dx_cm * dx_cm + dy_cm * dy_cm;
            double inv_r_cm = 1.0 / sqrt(dist2_cm);
            double force_cm = G * ((par[i].m * cells[neighbor_index].m) / dist2_cm);
            double fx_cm = force_cm * dx_cm * inv_r_cm;
            double fy_cm = force_cm * dy_cm * inv_r_cm;

            par[i].ax += fx_cm / par[i].m;
            par[i].ay += fy_cm / par[i].m;

            /* if(i == 0) {
                printf("P%d/C%d mag: %.6f fx: %.6f fy: %.6f\n", i, neighbor_index, force_cm, fx_cm, fy_cm);
            }    */
        }
    }
}

void update_positions_and_velocities(particle_t *par, particle_t *to_remove, cell_t *cells, int start_row, int end_row, int ghost_par_count, long long n_part, long ncside, double side, double inv_cell_size, long total_cells, int rank)
{

    for (long long i = 0; i < n_part; i++) 
    {
        if(par[i].removed) continue; // Skip removed particles
        // Store previous cell index
        int prev_x_cell = par[i].x_cell;
        int prev_y_cell = par[i].y_cell;
        int prev_cell_index = prev_y_cell * ncside + prev_x_cell;
        int local_prev_cell_index = (prev_y_cell - start_row) * ncside + prev_x_cell;
        
        // Update positions
        par[i].x += par[i].vx * DELTAT + 0.5 * par[i].ax * DELTAT * DELTAT;
        par[i].y += par[i].vy * DELTAT + 0.5 * par[i].ay * DELTAT * DELTAT;
        
        // Apply toroidal wrapping
        par[i].x = fmod(par[i].x, side);
        if (par[i].x < 0)
            par[i].x += side;

        par[i].y = fmod(par[i].y, side);
        if (par[i].y < 0)
            par[i].y += side;

        // Compute new cell
        int new_x_cell = (int)floor(par[i].x * inv_cell_size);
        int new_y_cell = (int)floor(par[i].y * inv_cell_size);

        // Clamp to ensure particles don't go out of bounds
        if (new_x_cell >= ncside)
            new_x_cell = ncside - 1;
        if (new_y_cell >= ncside)
            new_y_cell = ncside - 1;

        // Compute new cell index
        int new_cell_index = new_y_cell * ncside + new_x_cell;
        int local_new_cell_index = (new_y_cell - start_row) * ncside + new_x_cell;

        // Remove particle from old cell if it changed and add to new one
        if (new_cell_index != prev_cell_index)
        {
            
            if(new_y_cell < start_row || new_y_cell > end_row) {
                to_remove[ghost_par_count] = par[i]; // Store particle for removal
                ghost_par_count++;
                par[i].removed = 1; // Mark particle for removal
            }
            
            
            int *indices = cells[local_prev_cell_index].indices;
            int count = cells[local_prev_cell_index].count;
            
            // Find and remove the particle from the old cell
            for (int j = 0; j < count; j++)
            {
                if (indices[j] == i)
                {
                    indices[j] = indices[count - 1]; // Swap with last element
                    cells[local_prev_cell_index].count--;  // Reduce count
                    break;
                }
            }
            
            if(!(par[i].removed == 1)) {
                // Update particle cell assignment
                par[i].x_cell = new_x_cell;
                par[i].y_cell = new_y_cell;

                cells[local_new_cell_index].indices[cells[local_new_cell_index].count++] = i;
            }
            
            
        }

        // Update velocities
        if(!(par[i].removed == 1)) {
            par[i].vx += par[i].ax * DELTAT;
            par[i].vy += par[i].ay * DELTAT;
        }
            
    }
}

///////////////////////////////////////
// MPI Helper Functions
///////////////////////////////////////

void exchange_ghost_cells(int start_row, int end_row, int rank, int size, int local_rows, int ncside, int truesize, MPI_Comm comm, cell_t *local_cells, cell_t *ghost_upper, cell_t *ghost_lower) {

    cell_t *send_upper = &local_cells[0];
    cell_t *send_lower = &local_cells[(local_rows -1) * ncside];
    cell_t *recv_upper = malloc(ncside * sizeof(cell_t));
    cell_t *recv_lower = malloc(ncside * sizeof(cell_t));

    int rank_above = (rank - 1 + truesize) % truesize;
    int rank_below = (rank + 1) % truesize; 
    
    MPI_Sendrecv(send_upper, ncside * sizeof(cell_t), MPI_BYTE, rank_above, 0,
                recv_lower, ncside * sizeof(cell_t), MPI_BYTE, rank_below, 0,
                 comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_lower, ncside * sizeof(cell_t), MPI_BYTE, rank_below, 1,
                 recv_upper, ncside * sizeof(cell_t), MPI_BYTE, rank_above, 1,
                 comm, MPI_STATUS_IGNORE);    
    
    // Store ghost rows
    memcpy(ghost_upper, recv_upper, ncside * sizeof(cell_t));
    memcpy(ghost_lower, recv_lower, ncside * sizeof(cell_t));

    free(recv_upper);
    free(recv_lower);

}

void exchange_particles(int rank, int size, int local_rows, int ncside, int truesize, MPI_Comm comm, particle_t *local_particles, particle_t *ghost_particles, long ghost_par_count) {
    // Send and receive particles that have moved out of the local domain.
    // This is a simplified version; you may need to adjust it based on your specific requirements.

    // Send particles to the next process (rank + 1)
    MPI_Send(local_particles, local_rows * ncside * sizeof(particle_t), MPI_BYTE, (rank + 1) % size, 0, comm);

    // Receive particles from the previous process (rank - 1)
    MPI_Recv(ghost_particles, local_rows * ncside * sizeof(particle_t), MPI_BYTE, (rank - 1 + size) % size, 0, comm, MPI_STATUS_IGNORE);
}

// New function: determine local domain bounds for a process.
void get_local_domain(int rank, int size, int ncside, int *start_row, int *end_row) {
    // Divide ncside rows as evenly as possible among all processes.
    int rows_per_proc = ncside / size;
    int extra = ncside % size;
    *start_row = rank * rows_per_proc + (rank < extra ? rank : extra);
    *end_row = *start_row + rows_per_proc - 1;
    if (rank < extra) {
        (*end_row)++;
    }
}


///////////////////////////////////////
// Debug Functions
///////////////////////////////////////

//Cell row assignment debug print
void print_process_cell_assignment(int rank, int size, long ncside, int side, int start_row, int end_row) {

    if (rank == 0) printf("\n=== Cell Row Assignment ===\n");
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d/%d handles rows: %d to %d\n", 
           rank, size, start_row, end_row);
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_local_particles(int rank, int size, particle_t *par, long long n_part, int inv_cell_side) {

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\n=== Particle Distribution Across Processes ===\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Each process prints its own particles
    printf("\nProcess %d/%d has %lld particles:\n", rank, size, n_part);
    printf(" Local ID | Process | x_cell | y_cell | vx | vy | x | y\n");
    printf("---------------------------------------\n");
    
    for (long long i = 0; i < n_part; i++) {
        int x_cell = (int)(par[i].x * inv_cell_side);
        int y_cell = (int)(par[i].y * inv_cell_side);
        double vx = par[i].vx;
        double vy = par[i].vy;
        double x = par[i].x;
        double y = par[i].y;

        printf("%8lld | %6d | %6d | %6d | %f | %f | %f | %F\n", 
              i, rank, x_cell, y_cell, vx, vy, x, y);
    }
    
    // Add separator after last process
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == size-1) {
        printf("\n=== End of Distribution Report ===\n\n");
    }
}

void print_cells(cell_t *cells, long ncside, int rank) {
    printf("Process %d Cell Data:\n", rank);
    printf("Cell Index | x | y | m\n");
    printf("-------------------------\n");
    for (int i = 0; i < ncside; i++)
    {
        printf("Cell %d x: %.6f y: %.6f m: %.6f\n", i, cells[i].x, cells[i].y, cells[i].m);
    }
}