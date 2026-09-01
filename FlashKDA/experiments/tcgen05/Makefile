NVCC ?= nvcc
ARCH ?= 103a
NVCC_FLAGS ?= -O3 -std=c++17 -lineinfo

GENCODE := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
TARGET := build/kda_mma_microbench

.PHONY: all run clean

all: $(TARGET)

$(TARGET): kda_mma_microbench.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(GENCODE) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
