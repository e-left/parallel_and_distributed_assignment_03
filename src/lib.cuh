#include "fglt.cuh"

void readMTX(
    mwIndex **row,
    mwIndex **col,
    mwSize *const n,
    mwSize *const m,
    char const *const filename);
    
std::ostream &output(std::ostream &outfile, double *arr, int rows, int cols);