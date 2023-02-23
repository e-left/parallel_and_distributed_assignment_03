#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <limits>
#include <sstream>
#include <string.h>

#include "fglt.cuh"
#include "lib.cuh"

struct timeval tic() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv;
}

static double toc(struct timeval begin) {
    struct timeval end;
    gettimeofday(&end, NULL);
    double stime = ((double)(end.tv_sec - begin.tv_sec) * 1000) +
                   ((double)(end.tv_usec - begin.tv_usec) / 1000);
    stime = stime / 1000;
    return (stime);
}

int main(int argc, char** argv) {
    int BLOCKS;
    int THREADS;

    std::string filename = "graph.mtx";

    // ----- retrieve the (non-option) arguments:
    if ((argc <= 3) || (argv[argc - 3] == NULL) || (argv[argc - 1][0] == '-')) {
        // there is NO input...
        std::cout << "Improper usage" << std::endl;
        std::cout << "Usage: ./executable <BLOCKS> <THREADS> <filename>" << std::endl;
        return 1;
    } else {
        filename = argv[argc - 1];
        THREADS = atoi(argv[argc - 2]);
        BLOCKS = atoi(argv[argc - 3]);
        std::cout << "Using graph stored in '" << filename << "'." << std::endl;
        std::cout << "Using " << BLOCKS << " and " << THREADS << " per block." << std::endl;
        
    }
    mwIndex *row, *col;
    mwSize n, m;

    readMTX(&row, &col, &n, &m, filename.c_str());

    // GPU
    double *f; 
    cudaMalloc((void**)&f, NGRAPHLET * n * sizeof(double *));
    double *fn;
    cudaMalloc((void**)&fn, NGRAPHLET * n * sizeof(double *));

    // move row, col to gpu
    // copy to device memory
    mwIndex *rowGPU, *colGPU;
    cudaMalloc((void**)&rowGPU, m * sizeof(mwIndex));
    cudaMalloc((void**)&colGPU, (n + 1) * sizeof(mwIndex));
    cudaMemcpy(rowGPU, row, m * sizeof(mwIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(colGPU, col, (n + 1) * sizeof(mwIndex), cudaMemcpyHostToDevice);


    // initialize memory needed for the kernel
    double *t00;// = (double *)malloc(n * sizeof(double));
    cudaMalloc((void**)&t00, n * sizeof(double));
    double *fl;// = (double *)malloc(n * sizeof(double));
    cudaMalloc((void**)&fl, BLOCKS * THREADS * n * sizeof(double));
    int *pos;// = (int *)malloc(n * sizeof(int));
    cudaMalloc((void**)&pos, BLOCKS * THREADS * n * sizeof(int));
    mwIndex *isUsed;// = (mwIndex *)malloc(n * sizeof(mwIndex));
    cudaMalloc((void**)&isUsed, BLOCKS * THREADS * n * sizeof(mwIndex));
    double *c3;// = (double *)malloc(m * sizeof(double));
    cudaMalloc((void**)&c3, m * sizeof(double));
    int *isNgbh;// = (int *)malloc(n * sizeof(int));
    cudaMalloc((void**)&isNgbh, BLOCKS * THREADS * n * sizeof(int));

    std::cout << "Initiating fast graphlet transform for '" << filename << std::endl;

    // time it
    struct timeval timer_all = tic();
    // convert to kernel
    compute<<<BLOCKS, THREADS>>>(f, fn, rowGPU, colGPU, n, m, t00, fl, pos, isUsed, c3, isNgbh);
    printf("Total elapsed time: %.6f sec\n", toc(timer_all));

    std::stringstream intfilename;
    intfilename << "_B_" << BLOCKS << "_T_" << THREADS << "_";
    std::string output_filename = filename.substr(0, filename.length() - 4) + intfilename.str() + "freq_net.csv";

    std::cout << "Computation complete, outputting frequency counts to " << output_filename << std::endl;

    // output
    std::fstream ofnet(output_filename, std::ios::out);
    // move fn back to host
    double *fnhost = (double *)malloc(NGRAPHLET * n * sizeof(double));
    cudaMemcpy(fnhost, fn, NGRAPHLET * n * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Moved results back to host" << std::endl;

    if (ofnet.is_open()) {
        output(ofnet, fnhost, n, NGRAPHLET);
    }

    std::cout << "Finished, cleaning up..." << std::endl;

    free(fnhost);
    free(row);
    free(col);

    cudaFree(fl);
    cudaFree(pos);
    cudaFree(isUsed);
    cudaFree(t00);
    cudaFree(c3);
    cudaFree(isNgbh);

    cudaFree(f);
    cudaFree(fn);
    cudaFree(rowGPU);
    cudaFree(colGPU);

    return 0;
}
