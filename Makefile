OPENDIRS = -I/usr/include/openslide/ -I/usr/local/include/opencv4

CC = g++
CXXFLAGS = -fopenmp -Wextra -std=c++17 -lm $(OPENDIRS) 
CXXLIBS = -lopenslide -ltiff `pkg-config opencv --libs`

NVCC = nvcc
NXXFLAGS = $(OPENDIRS) 
NXXLIBS = 


CUDADIR = /usr/local/cuda
CUDALIBDIR = -L/usr/local/cuda/lib64
CUDAINCDIR = -I/usr/local/cuda/include
CUDALINKLIBS = -lcudart 

SRCDIR = src
OBJDIR = obj
INCDIR = inc

#Target Executable Name
EXE = exe

#Object Files
OBJS = $(OBJDIR)/gaussian.o $(OBJDIR)/morphology.o $(OBJDIR)/main.o $(OBJDIR)/cuda_kernel.o $(OBJDIR)/cuda_wrapper.o

## [ Compile ] ##

# Link C++ and CUDA compiled object files to target exe
$(EXE) : $(OBJS)
	$(CC) $(CXXFLAGS) $(OBJS) -o $@ $(CXXLIBS) $(CUDAINCDIR) $(CUDALIBDIR) $(CUDALINKLIBS)

# Compile main .cpp file to object files:
$(OBJDIR)/%.o : %.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@ 

# Compile C++ source files to object files

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCDIR)/%.hpp
	$(CC) $(CXXFLAGS) -c $< -o $@ 

# Compile CUDA soruce files to object files
$(OBJDIR)/%.o : $(SRCDIR)/%.cu $(INCDIR)/%.cuh
	$(NVCC) $(NXXFLAGS) -c $< -o $@ $(NXXLIBS)

# Compile CUDA wrapper with C-style header prototype
$(OBJDIR)/%.o : $(SRCDIR)/%.cu $(INCDIR)/%.h
	$(NVCC) $(NXXFLAGS) -c $< -o $@ $(NXXLIBS)
	
# Clean objects in object directory
clean:
	rm -rf bin/* *.o $(EXE)