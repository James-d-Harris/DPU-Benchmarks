cc = gcc
cflags = -O3 -Wall -fopenmp `dpu-pkg-config --cflags dpu`
ldflags = `dpu-pkg-config --libs dpu`
tasklets = 16
stack_size = 4096

all: bench_host bench_dpu

bench_dpu: bench_dpu.c
	dpu-upmem-dpurte-clang -O2 \
		-DNR_TASKLETS=$(tasklets) \
		-DSTACK_SIZE_DEFAULT=$(stack_size) \
		bench_dpu.c -o bench_dpu

bench_host: bench_host.c
	$(cc) $(cflags) bench_host.c $(ldflags) -o bench_host

clean:
	rm -f bench_host bench_dpu bench_results.csv
