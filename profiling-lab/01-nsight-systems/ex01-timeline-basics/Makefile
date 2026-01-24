# Makefile for timeline-basics exercise

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

all: baseline improved solution

baseline: baseline.cu
	$(NVCC) $(NVCC_FLAGS) -o baseline baseline.cu

improved: improved.cu
	$(NVCC) $(NVCC_FLAGS) -o improved improved.cu

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) -o solution solution.cu

profile-baseline: baseline
	nsys profile -o baseline_report ./baseline
	nsys stats baseline_report.nsys-rep

profile-improved: improved
	nsys profile -o improved_report ./improved
	nsys stats improved_report.nsys-rep

profile-solution: solution
	nsys profile -o solution_report ./solution
	nsys stats solution_report.nsys-rep

clean:
	rm -f baseline improved solution *.nsys-rep *.sqlite

.PHONY: all clean profile-baseline profile-improved profile-solution
