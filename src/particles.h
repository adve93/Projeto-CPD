#ifndef PARTICLES
#define PARTICLES

#include <math.h>

typedef struct  {

    double x, y;
    double vx, vy;
    double m;
} particle_t;

void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);

#endif