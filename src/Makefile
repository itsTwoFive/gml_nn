CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -lm

SRC = matrix.c gml_nn.c data_handler.c xor.c
OBJ = $(SRC:.c=.o)
TARGET = xor

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)