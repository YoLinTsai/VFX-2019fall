CSRCS = $(wildcard *.cpp)
CHDRS = $(wildcard *.h)
COBJS = $(addsuffix .o, $(basename $(CSRCS)))

CXX  = g++
ECHO = /bin/echo

CFLAGS = -g -Wall -std=c++11
CFLAGS = -O3 -Wall -std=c++11

OPENCV_FLAG = `pkg-config --cflags --libs opencv`

EXEC = detector


all: $(COBJS)
	@echo "> building -> $(EXEC)..."
	@$(CXX) $(CFLAGS) $(COBJS) -o $(EXEC) $(OPENCV_FLAG)

%.o: %.cpp
	@echo "> compiling: $<"
	@$(CXX) $(CFLAGS) -c -o $@ $< $(OPENCV_FLAG)

clean:
	@rm -f $(COBJS) $(EXEC)
