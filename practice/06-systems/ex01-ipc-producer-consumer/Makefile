NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80

all: producer consumer solution_producer solution_consumer

producer: producer.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

consumer: consumer.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution_producer: solution_producer.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

solution_consumer: solution_consumer.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f producer consumer solution_producer solution_consumer
	rm -f ipc_handle.bin consumer_done

test: solution_producer solution_consumer
	@echo "Starting producer in background..."
	./solution_producer &
	@sleep 1
	@echo "Running consumer..."
	./solution_consumer
	@echo "Test complete!"

.PHONY: all clean test
