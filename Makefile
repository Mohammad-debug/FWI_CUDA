CXX      := g++-9
CXXFLAGS := # -std=c++17 -pedantic-errors -Wall -Wextra -Werror
LDFLAGS  := -L/usr/lib -lstdc++ -lm 
OBJ_DIR  := obj
APP_DIR  := bin
TARGET   := seis_fwi
INCLUDE  := -Iinclude -I/usr/include/hdf5
SRC      :=                      \
   $(wildcard src/*.cpp)         \
   $(wildcard ext/*/*.cpp)         \

LIB = #ext/inih/*.o

OBJECTS  := $(SRC:%.cpp=$(OBJ_DIR)/%.o) $(LIB)


all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@ $(LDFLAGS)

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
	python3 ./scripts/pre_proc.py
	$(APP_DIR)/$(TARGET) 

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(APP_DIR)/*

