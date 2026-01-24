NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

all: sequential overlapped solution

sequential: sequential.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

overlapped: overlapped.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

solution: solution.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

profile-sequential: sequential
	nsys profile --stats=true -o sequential_profile ./sequential

profile-overlapped: overlapped
	nsys profile --stats=true -o overlapped_profile ./overlapped

profile-solution: solution
	nsys profile --stats=true -o solution_profile ./solution

compare: profile-sequential profile-solution
	@echo "Compare profiles in Nsight Systems GUI:"
	@echo "  nsys-ui sequential_profile.nsys-rep solution_profile.nsys-rep"

clean:
	rm -f sequential overlapped solution *.nsys-rep *.sqlite

.PHONY: all clean profile-sequential profile-overlapped profile-solution compare
