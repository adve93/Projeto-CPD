# Detect OS and set compiler accordingly
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    CC = /opt/homebrew/opt/llvm/bin/clang  # Use Homebrew's Clang
    FLAGS += -I/opt/homebrew/opt/llvm/include -L/opt/homebrew/opt/llvm/lib -O2 -fopenmp
else
    CC = gcc
    FLAGS += -O2 -fopenmp
endif

CC_MPI = mpicc
CC_FLAGS_MPI = -O2 -g

# Targets
TARGET_SEQ = parsim
TARGET_OMP = parsim-omp
TARGET_MPI = parsim-mpi

all: $(TARGET_SEQ) $(TARGET_OMP) $(TARGET_MPI)

# Sequential version (no OpenMP)
$(TARGET_SEQ): parsim.o
	$(CC) $(FLAGS) -o $(TARGET_SEQ) parsim.o -lm

parsim.o: parsim.c
	$(CC) $(FLAGS) -c parsim.c

# OpenMP version (linked with OpenMP)
$(TARGET_OMP): parsim-omp.o
	$(CC) $(FLAGS) -o $(TARGET_OMP) parsim-omp.o -lm

parsim-omp.o: parsim-omp.c
	$(CC) $(FLAGS) -c parsim-omp.c

# OpenMPI version
$(TARGET_MPI): parsim-mpi.c
	$(CC_MPI) $(CC_FLAGS_MPI) -o $(TARGET_MPI) parsim-mpi.c

parsim-mpi.o: parsim-mpi.c
	$(CC_MPI) $(CC_FLAGS_MPI) -c parsim-mpi.c

# Build only the sequential version
seq: $(TARGET_SEQ)

# Build only the OpenMP parallel version
omp: $(TARGET_OMP)

# Build only the OpenMPI parallel version
mpi: $(TARGET_MPI)

clean:
	rm -f $(TARGET_SEQ) $(TARGET_OMP) $(TARGET_MPI) *.o
	rm -f -r $(TARGET_MPI) *.dSYM
