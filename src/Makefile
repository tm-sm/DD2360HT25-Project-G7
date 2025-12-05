CUDA_PATH     ?= /usr/lib/cuda
HOST_COMPILER  = g++
NVCC           = nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

# Activated optimizations
DEFINES ?= 
ARGS = $(foreach a,$(DEFINES),-D$(a))
ifeq ($(findstring FAST_MATH, $(DEFINES)), FAST_MATH)
    ARGS += -use_fast_math
endif

NVCCFLAGS      = $(NVCC_DBG) -m64 $(ARGS)
GENCODE_FLAGS  = -gencode arch=compute_86,code=sm_86

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h bvh.h bvh_n.h aabb.h

cudart: cudart.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart cudart.o

cudart.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart.o -c main.cu

.PHONY: out.ppm
.PHONY: out.jpg
.PHONY: profile_basic
.PHONY: profile_metrics
.PHONY: clean

out.ppm: cudart
	rm -f out.ppm
	./cudart > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

profile_basic: cudart
	nvprof ./cudart > out.ppm

# use nvprof --query-metrics
profile_metrics: cudart
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cudart > out.ppm

clean:
	rm -f cudart cudart.o out.ppm out.jpg
