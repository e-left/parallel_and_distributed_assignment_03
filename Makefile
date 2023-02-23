gpu: $(wildcard src/*.cu)
	nvcc $(wildcard src/*.cu) -o out/gpu