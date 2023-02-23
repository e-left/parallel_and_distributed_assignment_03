#include "fglt.cuh"
#include <algorithm>

__device__ static void remove_neighbors(
    int *isNgbh,
    mwIndex i,
    mwIndex *ii,
    mwIndex *jStart) {
    // --- remove neighbors
    for (mwIndex id_i = jStart[i]; id_i < jStart[i + 1]; id_i++) {
        // get the column (k)
        mwIndex k = ii[id_i];
        isNgbh[k] = 0;
    }
}

__device__ static void raw2net(
    double *const f,
    double const *const d,
    mwIndex const i) {
    f[0 + NGRAPHLET * i] = d[0 + NGRAPHLET * i];
    f[1 + NGRAPHLET * i] = d[1 + NGRAPHLET * i] - 2 * d[3 + NGRAPHLET * i];
    f[2 + NGRAPHLET * i] = d[2 + NGRAPHLET * i] - d[3 + NGRAPHLET * i];
    f[3 + NGRAPHLET * i] = d[3 + NGRAPHLET * i];
}

__device__ static void compute_all_available(
    double *f,
    mwIndex i) {
    f[2 + NGRAPHLET * i] = f[0 + NGRAPHLET * i] * (f[0 + NGRAPHLET * i] - 1) * 0.5;
}

__device__ static void spmv_first_pass(
    double *f2_i,
    double *f1,
    mwIndex i,
    mwIndex *jStart,
    mwIndex *ii,
    mwSize n) {

    // --- loop through every nonzero element A(i,k)
    for (mwIndex id_i = jStart[i]; id_i < jStart[i + 1]; id_i++) {

        // get the column (k)
        mwIndex k = ii[id_i];

        // --- matrix-vector products
        f2_i[0] += f1[NGRAPHLET * k];
    }

    f2_i[0] -= f1[NGRAPHLET * i];
}

__device__ static void p2(
    double *f4_i,
    double *c3,
    double *t00,
    mwIndex i,
    mwIndex *jStart,
    mwIndex *ii,
    double *fl,
    int *pos,
    int *isNgbh,
    mwIndex *isUsed) {

    // setup the count of nonzero columns (j) visited for this row (i)
    mwIndex cnt = 0;

    // --- loop through every nonzero element A(i,k)
    for (mwIndex id_i = jStart[i]; id_i < jStart[i + 1]; id_i++) {

        // get the column (k)
        mwIndex k = ii[id_i];

        isNgbh[k] = id_i + 1;

        // --- loop through all nonzero elemnts A(k,j)
        for (mwIndex id_k = jStart[k]; id_k < jStart[k + 1]; id_k++) {

            // get the column (j)
            mwIndex j = ii[id_k];

            if (i == j)
                continue;

            // if this column is not visited yet for this row (i), then set it
            if (!isUsed[j]) {
                fl[j] = 0;      // initialize corresponding element
                isUsed[j] = 1;  // set column as visited
                pos[cnt++] = j; // add column position to list of visited
            }

            // increase count of A(i,j)
            fl[j]++;
        }
    }

    // --- perform reduction on [cnt] non-empty columns (j)
    for (mwIndex l = 0; l < cnt; l++) {

        // get next column number (j)
        mwIndex j = pos[l];

        if (isNgbh[j]) {
            c3[isNgbh[j] - 1] = fl[j];

            f4_i[0] += fl[j];
        }

        // declare it non-used
        isUsed[j] = 0;
    }

    f4_i[0] /= 2;
}

__global__ void compute(
    double *const f,
    double *const fn,
    mwIndex *ii,
    mwIndex *jStart,
    mwSize n,
    mwSize m,
    double *t00,
    double *fl,
    int* pos,
    mwIndex* isUsed,
    double* c3,
    int* isNgbh ) {


    // setup loop variables for parallelisation
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // parallelised
    for (mwSize i = index; i < n; i += stride) {
        // get degree of vertex (i)
        f[0 + NGRAPHLET * i] = jStart[i + 1] - jStart[i];
        // t00[i] = f[1 + NGRAPHLET * i] - 2;
        t00[i] = -2;
    }

    // parallelised
    // --- first pass
    for (mwIndex i = index; i < n; i += stride) {
        // on cilk version its worker id (zero based)
        // here its current thread id
        int ip = index;

        // d_4
        p2(&f[3 + NGRAPHLET * i],
           c3, t00, i, jStart, ii,
           &fl[ip * n], &pos[ip * n], &isNgbh[ip * n], &isUsed[ip * n]);

        // d_2
        spmv_first_pass(&f[1 + NGRAPHLET * i], &f[0], i, jStart, ii, n);

        // d_3
        compute_all_available(f, i);
        remove_neighbors(&isNgbh[ip * n], i, ii, jStart);
    }

    // parallelised
    // transform to net
    for (mwIndex i = index; i < n; i += stride) {
        raw2net(fn, f, i);
    }
}
