NVCC ?= nvcc
ARCH ?= 103a
TARGET := thin_gemm_bench

NVCCFLAGS := -O3 -std=c++17 -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
LDLIBS := -lcublas

.PHONY: all run clean

all: $(TARGET)

$(TARGET): thin_gemm_bench.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

run: $(TARGET)
	./$(TARGET) 2250 8000 results_b300.csv

clean:
	rm -f $(TARGET)
