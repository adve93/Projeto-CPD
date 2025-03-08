CC = gcc
CFLAGS = -O2 -fopenmp
TARGET = parsim

all: $(TARGET)

$(TARGET): parsim.o
	$(CC) $(CFLAGS) -o $(TARGET) parsim.o -lm

parsim.o: parsim.c
	$(CC) $(CFLAGS) -c parsim.c

clean:
	rm -f $(TARGET) *.o