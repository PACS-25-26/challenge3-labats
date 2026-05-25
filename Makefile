CXX = mpic++

CXXFLAGS = -std=c++17 -O3 -Wall -fopenmp -Iinclude

SRC_DIR = src

TARGET = jacobi

SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/solver.cpp $(SRC_DIR)/vtk_export.cpp

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

.PHONY: all clean