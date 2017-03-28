#ifndef IRLS_LINEAR_REGRESSION_H
#define IRLS_LINEAR_REGRESSION_H

#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"
#include <utility>
#include <memory>
#include <functional>

#include <iostream>
#include <iomanip>
void printGSLMatrix(const gsl_matrix* mat)
{
    std::cout<<std::setprecision(6);
    for(size_t r=0; r<mat->size1; ++r)
    {
        for(size_t c=0; c<mat->size2; ++c)
        {
            std::cout<<gsl_matrix_get(mat, r, c)<<" ";
        }
        std::cout<<std::endl;
    }
}

void printGSLVector(const gsl_vector* vec)
{
    std::cout<<std::setprecision(6);
    for(size_t i=0; i<vec->size; ++i)
    {
        std::cout<<gsl_vector_get(vec, i)<<" ";
    }
    std::cout<<std::endl;
}

/*
 * b (coefficeints), cov (covariance matrix), and wchisq (weighted chi-square) are output
 * WegihtFunction signature: void (const gsl_vector* y, const gsl_matrix* x, const gsl_vector* b, gsl_matrix *w)
 * w is the new weight output.
 */
typedef std::unique_ptr<gsl_matrix, std::function<void(gsl_matrix*)> > GslMatUniPtr;
typedef std::unique_ptr<gsl_vector, std::function<void(gsl_vector*)> > GslVecUniPtr;
typedef std::unique_ptr<gsl_permutation, std::function<void(gsl_permutation*)> > GslPerUniPtr;

template <class WeightFunction>
int irls_linear_regression(const gsl_vector* y, const gsl_matrix* x, gsl_matrix* weight, WeightFunction &&wf,
                           gsl_vector* const b, gsl_matrix* const cov, double* const wchisq,
                           size_t maxIter=20, double tolerance=0.000001)
{
    const size_t n = y->size;
    const size_t p = x->size2;
    if(n!=x->size1 || n!=weight->size1 || n!=weight->size2)
        return -1;

    if(b==nullptr || b->size!=p)
        return -2;

    auto gslMatDeleter = [](gsl_matrix* mPtr){gsl_matrix_free(mPtr);};
    auto gslVecDeleter = [](gsl_vector* vPtr){gsl_vector_free(vPtr);};
    auto gslPerDeleter = [](gsl_permutation* pPtr){gsl_permutation_free(pPtr);};

    GslVecUniPtr b_new (gsl_vector_alloc(p), gslVecDeleter);
    GslVecUniPtr b_old (gsl_vector_alloc(p), gslVecDeleter);
    GslMatUniPtr tempA (gsl_matrix_alloc(p, n), gslMatDeleter);
    GslMatUniPtr leftMat (gsl_matrix_alloc(p, p), gslMatDeleter);
    GslVecUniPtr rightVec (gsl_vector_alloc(p), gslVecDeleter);
    GslMatUniPtr leftMatInv (gsl_matrix_alloc(p, p), gslMatDeleter);
    GslPerUniPtr permu (gsl_permutation_alloc(p), gslPerDeleter);
    GslVecUniPtr residual (gsl_vector_alloc(n), gslVecDeleter);
    int signum;
    if(!b_new || !b_old || !tempA || !leftMat || !rightVec || !leftMatInv || !permu || !residual)
        return -3;

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, x, weight, 0.0, tempA.get());
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tempA.get(), x, 0.0, leftMat.get());
    gsl_blas_dgemv(CblasNoTrans, 1.0, tempA.get(), y, 0.0, rightVec.get());
    gsl_linalg_LU_decomp(leftMat.get(), permu.get(), &signum);
    gsl_linalg_LU_invert(leftMat.get(), permu.get(), leftMatInv.get());
    gsl_blas_dgemv(CblasNoTrans, 1.0, leftMatInv.get(), rightVec.get(), 0.0, b_new.get());

    for(size_t i=0; i<maxIter; ++i)
    {
        b_new.swap(b_old);
        wf(y, x, b_old.get(), weight);

        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, x, weight, 0.0, tempA.get());
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tempA.get(), x, 0.0, leftMat.get());
        gsl_blas_dgemv(CblasNoTrans, 1.0, tempA.get(), y, 0.0, rightVec.get());
        gsl_linalg_LU_decomp(leftMat.get(), permu.get(), &signum);
        gsl_linalg_LU_invert(leftMat.get(), permu.get(), leftMatInv.get());
        gsl_blas_dgemv(CblasNoTrans, 1.0, leftMatInv.get(), rightVec.get(), 0.0, b_new.get());

        gsl_vector_sub(b_old.get(), b_new.get());
        if(gsl_blas_dasum(b_old.get())<tolerance)
            break;
    }

    gsl_vector_memcpy(b, b_new.get());
    if(cov!=nullptr && cov->size1==cov->size2 && cov->size1==p)
        gsl_matrix_memcpy(cov, leftMatInv.get());
    if(wchisq!=nullptr)
    {
        gsl_blas_dgemv(CblasNoTrans, 1.0, x, b_new.get(), 0.0, residual.get());
        gsl_vector_sub(residual.get(), y);
        *wchisq = 0.0;
        for(size_t i=0; i<residual->size; ++i)
            *wchisq = *wchisq+gsl_vector_get(residual.get(), i)*gsl_vector_get(residual.get(), i)
                *gsl_matrix_get(weight, i, i);
    }
    return 0;
}

#endif // IRLS_LINEAR_REGRESSION_H
