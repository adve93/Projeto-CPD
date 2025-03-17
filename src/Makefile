# Detect OS and set compiler accordingly
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    CC = /opt/homebrew/opt/llvm/bin/clang  # Use Homebrew's Clang
    FLAGS += -I/opt/homebrew/opt/llvm/include -L/opt/homebrew/opt/llvm/lib -fopenmp
else
    CC = gcc
    FLAGS += -fopenmp
endif

# Targets
TARGET_SEQ = parsim
TARGET_OMP = parsim-omp

all: $(TARGET_SEQ) $(TARGET_OMP)

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

# Build only the sequential version
seq: $(TARGET_SEQ)

# Build only the OpenMP parallel version
omp: $(TARGET_OMP)

clean:
	rm -f $(TARGET_SEQ) $(TARGET_OMP) *.o
