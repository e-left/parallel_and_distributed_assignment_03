#ifndef FGLT_H_
#define FGLT_H_

#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <stdio.h>

#include <sys/time.h>

// type definitions
#ifdef MX_COMPAT_32
typedef int mwSize;
typedef int mwIndex;
#else
typedef size_t mwSize;  /* unsigned pointer-width integer */
typedef size_t mwIndex; /* unsigned pointer-width integer */
#endif

#define NGRAPHLET 4

/*!
 * \brief Perform the FGLT transform.
 *
 * \param f [out] An array-of-pointers of size (n, 16) where the raw frequencies should be stored.
 * \param fn [out] An array-of-pointers of size (n, 16) where the net frequencies should be stored.
 * \param ii [in] The column indices of the adjacency matrix.
 * \param jStart [in] The first non-zero row index of each column.
 * \param n [in] The number of columns of the adjacency matrix.
 * \param m [in] The number of nonzero elements in the adjacency matrix.
 * \param np [in] The number of parallel workers to use for the transform.
 */
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
    int* isNgbh);

#endif /* FGLT_H_ */
