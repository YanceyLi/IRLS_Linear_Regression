#include <iostream>
#include <cmath>
#include "irls_linear_regression.h"

using namespace std;

void initGSLMatrix(gsl_matrix* mat, double* data)
{
    if(mat==nullptr || data==nullptr)
        return;
    for(size_t r=0; r<mat->size1; ++r)
        for(size_t c=0; c<mat->size2; ++c)
        {
            gsl_matrix_set(mat, r, c, *data);
            ++data;
        }
}

void initGSLVector(gsl_vector* vec, double* data)
{
    if(vec==nullptr || data==nullptr)
        return;
    for(size_t i=0; i<vec->size; ++i)
    {
        gsl_vector_set(vec, i, *data);
        ++data;
    }
}

// use residual in each iteration as weight
void weightFunction1(const gsl_vector* y, const gsl_matrix* x, const gsl_vector* b, gsl_matrix* w)
{
    const double delta = 0.0001;
    gsl_vector* expected = gsl_vector_alloc(y->size);
    gsl_blas_dgemv(CblasNoTrans, 1.0, x, b, 0.0, expected);
    gsl_vector_sub(expected, y);
    gsl_matrix_set_zero(w);
    for(size_t i=0; i<w->size1; ++i)
    {
        double error = std::abs(gsl_vector_get(expected, i));
        if(error<delta)
            error = delta;
        gsl_matrix_set(w, i, i, 1.0/error);
    }
    gsl_vector_free(expected);
}

void wf1InitialWeight(gsl_matrix* initW)
{
    gsl_matrix_set_identity(initW);
}

/*
 * use variance-inverted in each iteration as weight. Method is used in
 * MERKLE, W., Statistical methods in regression and calibration analysis of chromosome aberration data,
 * Radiat. Environ. Biophys. 21 (1983) 217–233.
 */
class WeightFunction2
{
private:
    gsl_vector* sampleSize;

public:
    WeightFunction2(double* sampleSizeData, size_t n)
    {
        sampleSize = gsl_vector_alloc(n);
        for(size_t i=0; i<n; ++i)
        {
            gsl_vector_set(sampleSize, i, *sampleSizeData);
            ++sampleSizeData;
        }
    }

    ~WeightFunction2()
    {
        gsl_vector_free(sampleSize);
    }

    WeightFunction2(const WeightFunction2& other) = delete;
    WeightFunction2& operator= (const WeightFunction2& other) = delete;

    void initialWeight(const gsl_vector* y, gsl_matrix* initW)
    {
        gsl_matrix_set_zero(initW);
        for(size_t i=0; i<y->size; ++i)
            gsl_matrix_set(initW, i, i, gsl_vector_get(sampleSize, i)/gsl_vector_get(y, i));
    }

    void operator() (const gsl_vector* y, const gsl_matrix* x, const gsl_vector* b, gsl_matrix* w)
    {
        gsl_vector* expected = gsl_vector_alloc(y->size);
        gsl_blas_dgemv(CblasNoTrans, 1.0, x, b, 0.0, expected);
        gsl_matrix_set_zero(w);
        for(size_t i=0; i<expected->size; ++i)
            gsl_matrix_set(w, i, i, gsl_vector_get(sampleSize, i)/gsl_vector_get(expected, i));
        gsl_vector_free(expected);
    }
};

int main()
{
    cout << "Hello World!" << endl;

    // test set 1
    gsl_matrix* x1 = gsl_matrix_alloc(7, 3);
    gsl_vector* y1 = gsl_vector_alloc(7);
    double data1x[21] = {1.0, 0.18, 0.89,
                         1.0, 1.0, 0.26,
                         1.0, 0.92, 0.11,
                         1.0, 0.07, 0.37,
                         1.0, 0.85, 0.16,
                         1.0, 0.99, 0.41,
                         1.0, 0.87, 0.47};
    double data1y[7] = {109.85, 155.72, 137.66, 76.17, 139.75, 162.6, 151.77};
    initGSLMatrix(x1, data1x);
    initGSLVector(y1, data1y);
    gsl_matrix* w1 = gsl_matrix_alloc(7, 7);

    wf1InitialWeight(w1);
    gsl_vector* b1 = gsl_vector_alloc(3);
    irls_linear_regression(y1, x1, w1, weightFunction1, b1, nullptr, nullptr);
    cout << "Result of test1, irls" << endl;
    printGSLVector(b1);
    gsl_vector_free(b1);

    gsl_matrix_free(w1);
    gsl_vector_free(y1);
    gsl_matrix_free(x1);

    // test set 2
    gsl_matrix* x2 = gsl_matrix_alloc(10, 3);
    gsl_vector* y2 = gsl_vector_alloc(10);
    double data2x[30] = {1.0, 0.0, 0.0,
                        1.0, 0.1, 0.01,
                        1.0, 0.25, 0.0625,
                        1.0, 0.5, 0.25,
                        1.0, 0.75, 0.5625,
                        1.0, 1.0, 1.0,
                        1.0, 2.0, 4.0,
                        1.0, 3.0, 9.0,
                        1.0, 4.0, 16.0,
                        1.0, 5.0, 25.0};
    double data2y[10] = {0.0121757543673902,
                        0.0123684210526316,
                        0.0281578947368421,
                        0.0631578947368421,
                        0.108918918918919,
                        0.171290061944519,
                        0.507085916740478,
                        1.00870322019147,
                        1.66045845272206,
                        2.42319277108434};
    double data2Sample[10] = {3778.0, 3800.0, 3800.0, 3800.0, 3700.0,
                             3713.0, 2258.0, 1149.0, 698.0, 664.0};
    initGSLMatrix(x2, data2x);
    initGSLVector(y2, data2y);
    gsl_matrix* w2 = gsl_matrix_alloc(10, 10);
    gsl_matrix* cov2 = gsl_matrix_alloc(3, 3);
    gsl_vector* b2 = gsl_vector_alloc(3);
    double chisq;

    wf1InitialWeight(w2);
    irls_linear_regression(y2, x2, w2, weightFunction1, b2, cov2, &chisq);
    cout << "Result of test2, irls" << endl;
    printGSLVector(b2);
    cout << "Covariance, chisq "<<chisq<<endl;
    printGSLMatrix(cov2);

    WeightFunction2 wf2(data2Sample, 10);
    wf2.initialWeight(y2, w2);
    irls_linear_regression(y2, x2, w2, wf2, b2, cov2, &chisq);
    cout << "Result of test2, weight method 2" << endl;
    printGSLVector(b2);
    cout << "Covariance, chisq "<<chisq<<endl;
    printGSLMatrix(cov2);

    gsl_vector_free(b2);
    gsl_matrix_free(cov2);
    gsl_matrix_free(w2);
    gsl_vector_free(y2);
    gsl_matrix_free(x2);

    return 0;
}

