CSRCS = $(wildcard *.cpp)
CHDRS = $(wildcard *.h)
COBJS = $(addsuffix .o, $(basename $(CSRCS)))

CXX  = g++
ECHO = /bin/echo

CFLAGS = -g -Wall -std=c++11
CFLAGS = -O3 -Wall -std=c++11
LAFLAGS = -lm -lblas -llapack -llapacke

OPENCV_FLAG = `pkg-config --cflags --libs opencv`

EXEC = descriptor


all: $(COBJS)
	@echo "> building -> $(EXEC)..."
	@$(CXX) $(CFLAGS) $(COBJS) -o $(EXEC) $(OPENCV_FLAG) $(LAFLAGS)

%.o: %.cpp
	@echo "> compiling: $<"
	@$(CXX) $(CFLAGS) -c -o $@ $< $(OPENCV_FLAG) $(LAFLAGS)

clean:
	@rm -f $(COBJS) $(EXEC)
