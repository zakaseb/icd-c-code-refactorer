CC      = gcc
CFLAGS  = -Wall -Wextra -Werror -std=c99 -Isrc
SRCS    = src/comms.c src/main.c
OBJS    = $(SRCS:.c=.o)
TARGET  = comms_app

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
