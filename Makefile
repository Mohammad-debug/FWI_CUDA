CXX      := g++
#CUXX 	:= nvcc 
CXXFLAGS := -fopenmp #-std=c++17 -pedantic-errors -Wall -Wextra -Werror 
CUXXFLAGS :=  #-std=c++17 -pedantic-errors -Wall -Wextra -Werror
#LDFLAGS  := -L/usr/lib -L/opt/cuda/include -lstdc++ -lm -lcudart 
LDFLAGS  := -L/usr/lib -L/usr/lib/cuda -lstdc++ -lm -lcudart 
OBJ_DIR  := obj
APP_DIR  := bin
TARGET   := seis_fwi
INCLUDE  := -Iinclude -Iinclude/cpu 
INCLUDE_CUDA := -Iinclude/cuda

#-I/usr/include/hdf5
SRC  :=  $(wildcard src/*.cpp) $(wildcard src/cpu/*.cpp) $(wildcard ext/*/*.cpp) 
#SRC_CUDA := $(wildcard src/cuda/*.cu)

LIB = #ext/inih/*.o

OBJECTS  := $(SRC:%.cpp=$(OBJ_DIR)/%.o) #$(SRC_CUDA:%.cu=$(OBJ_DIR)/%.o) $(LIB)


all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(INCLUDE_CUDA) -c $< -o $@ $(LDFLAGS)

#$(OBJ_DIR)/%.o: %.cu
#	@mkdir -p $(@D)
#	$(CUXX) $(CUXXFLAGS) $(INCLUDE_CUDA) -c $< -o $@ $(LDFLAGS)

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

.PHONY: all build clean debug release

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O2
release: all

run:
	python3 ./scripts/pre_proc_dinesh.py
	$(APP_DIR)/$(TARGET) 
	#python3 ./scripts/post_proc.py

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(APP_DIR)/*

