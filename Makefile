CXX = g++
CXXFLAGS = -O3 -std=c++17 -I/usr/include/openvino
LDFLAGS = -lopenvino -lportaudio -lfftw3

TARGET = amd_noise_suppression
SRC = amd_noise_suppression.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
