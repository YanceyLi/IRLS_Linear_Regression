#ifndef IRLS_LINEAR_REGRESSION_H
#define IRLS_LINEAR_REGRESSION_H

#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"
#include <utility>

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
 * WegihtFunction signature: void (const gsl_vector* y, const gsl_matrix* x, const gsl_vector* b, gsl_matrix *w)
 * w is the new weight output.
 */
template <class WeightFunction>
gsl_vector* irls_linear_regression(const gsl_vector* y, const gsl_matrix* x, gsl_matrix* weight,
                                   WeightFunction &&wf, size_t maxIter=20, double tolerance=0.000001)
{
    const size_t n = y->size;
    const size_t p = x->size2;
    if(n!=x->size1 || n!=weight->size1 || n!=weight->size2)
        return nullptr;

    gsl_vector* b = gsl_vector_alloc(p);
    if(b==nullptr)
        return nullptr;
    gsl_vector* b_old = gsl_vector_alloc(p);
    gsl_matrix* tempA = gsl_matrix_alloc(p, n);
    gsl_matrix* leftMat = gsl_matrix_alloc(p, p);
    gsl_vector* rightVec = gsl_vector_alloc(p);
    gsl_matrix* leftMatInv = gsl_matrix_alloc(p, p);
    gsl_permutation* permu = gsl_permutation_alloc(p);
    int signum;
    if(b_old==nullptr || tempA==nullptr || leftMat==nullptr || rightVec==nullptr || leftMatInv==nullptr
            || permu==nullptr)
    {
        gsl_vector_free(b);
        gsl_vector_free(b_old);
        gsl_matrix_free(tempA);
        gsl_matrix_free(leftMat);
        gsl_vector_free(rightVec);
        gsl_matrix_free(leftMatInv);
        gsl_permutation_free(permu);
        return nullptr;
    }

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, x, weight, 0.0, tempA);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tempA, x, 0.0, leftMat);
    gsl_blas_dgemv(CblasNoTrans, 1.0, tempA, y, 0.0, rightVec);
    gsl_linalg_LU_decomp(leftMat, permu, &signum);
    gsl_linalg_LU_invert(leftMat, permu, leftMatInv);
    gsl_blas_dgemv(CblasNoTrans, 1.0, leftMatInv, rightVec, 0.0, b);

    for(size_t i=0; i<maxIter; ++i)
    {
        std::swap(b_old, b);
        wf(y, x, b_old, weight);

        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, x, weight, 0.0, tempA);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tempA, x, 0.0, leftMat);
        gsl_blas_dgemv(CblasNoTrans, 1.0, tempA, y, 0.0, rightVec);
        gsl_linalg_LU_decomp(leftMat, permu, &signum);
        gsl_linalg_LU_invert(leftMat, permu, leftMatInv);
        gsl_blas_dgemv(CblasNoTrans, 1.0, leftMatInv, rightVec, 0.0, b);

        gsl_vector_sub(b_old, b);
        if(gsl_blas_dasum(b_old)<tolerance)
            break;
    }
    gsl_vector_free(b_old);
    gsl_matrix_free(tempA);
    gsl_matrix_free(leftMat);
    gsl_vector_free(rightVec);
    gsl_matrix_free(leftMatInv);
    gsl_permutation_free(permu);
    return b;
}

#endif // IRLS_LINEAR_REGRESSION_H
