CC = gcc
CFLAGS = -O2 -fopenmp

# Targets
TARGET_SEQ = parsim
TARGET_OMP = parsim-omp

all: $(TARGET_SEQ) $(TARGET_OMP)

# Sequential version
$(TARGET_SEQ): parsim.o
	$(CC) $(CFLAGS) -o $(TARGET_SEQ) parsim.o -lm

parsim.o: parsim.c
	$(CC) $(CFLAGS) -c parsim.c

# OpenMP version
$(TARGET_OMP): parsim-omp.o
	$(CC) $(CFLAGS) -o $(TARGET_OMP) parsim-omp.o -lm

parsim-omp.o: parsim-omp.c
	$(CC) $(CFLAGS) -c parsim-omp.c

clean:
	rm -f $(TARGET_SEQ) $(TARGET_OMP) *.o
