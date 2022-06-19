CXX = g++ 
#CXXFLAGS = -g -std=c++11 -Wall -Wextra -Wno-sign-compare -O3
CXXFLAGS = -g -fopenmp -std=c++11 -Wall -Wno-sign-compare -O3

NVCXX = nvcc 
NVCXXFLAGS = -arch=sm_75 -g -std=c++11  -O3


CUDALIB = /usr/local/cuda/lib64
CUDAINC = /usr/local/cuda/include
SRCDIR = src
OBJDIR = obj
CUOBJDIR = cuobj
BINDIR = bin

INCS := $(wildcard $(SRCDIR)/*.h)
SRCS := $(wildcard $(SRCDIR)/*.cc)
OBJS := $(SRCS:$(SRCDIR)/%.cc=$(OBJDIR)/%.o)
CUSRCS := $(wildcard $(SRCDIR)/*.cu)
CUOBJS := $(CUSRCS:$(SRCDIR)/%.cu=$(CUOBJDIR)/%.o)

all:  bin/pr



bin/pr: $(OBJS) $(CUOBJS) 
	mkdir -p bin
	@echo "OBJ: "$(OBJS)
	@echo "CUOBJ: "$(CUOBJS)
	$(CXX) $^ -o $@ $(CXXFLAGS) -L$(CUDALIB) -lcudart -Iinclude -I$(CUDAINC) 
#$(NVCXX) $(NVCXXFLAGS) -L$(CUDALIB) -lcudart -lcuda -Iinclude -o $@  $^ 
			    @echo "Compiled "$<" successfully!"


.PHONY:	test clean

$(CUOBJS): $(CUOBJDIR)/%.o : $(SRCDIR)/%.cu
	    mkdir -p cuobj 
	    @echo $(NVCXX) $(NVCXXFLAGS) "-Iinclude -c" $< "-o" $@
	    @$(NVCXX) $(NVCXXFLAGS) -Iinclude -c $< -o $@
			    @echo "CUDA Compiled "$<" successfully!"

$(OBJS): $(OBJDIR)/%.o : $(SRCDIR)/%.cc
	    mkdir -p obj 
	    @echo $(CXX)  -c $< -o $@ $(CXXFLAGS) -L$(CUDALIB) -lcudart -Iinclude -I$(CUDAINC) 
	    $(CXX)  -c $< -o $@ $(CXXFLAGS) -L$(CUDALIB) -lcudart -Iinclude -I$(CUDAINC) 
			@echo "Compiled "$<" successfully!"

remote_facebook:
	mkdir -p result
	condor_submit pr_facebook.cmd

remote_livejournal:
	mkdir -p result
	condor_submit pr_livejournal.cmd

clean:
	rm -f $(OBJS) $(OBJS:%.o=%.d) 
	rm -f $(CUOBJS) $(CUOBJS:%.o=%.d) 
	rm -rf bin/*
	rm -rf result/*

