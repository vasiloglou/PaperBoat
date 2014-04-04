/*
Copyright © 2010, Ismion Inc
All rights reserved.
http://www.ismion.com/

Redistribution and use in source and binary forms, with or without
modification IS NOT permitted without specific prior written
permission. Further, neither the name of the company, Ismion
Inc, nor the names of its employees may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE Ismion Inc "AS IS" AND ANY
EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COMPANY BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef CPP_BLAS_H_
#define CPP_BLAS_H_
#include "blas.h"
/**
 * @file cppblas.h
 *
 * @brief A template version of BLAS
 */
namespace fl {
namespace dense {
template<typename Precision>
class CppBlas {
};

template<>
class CppBlas<float> {
  public:
    static void rot(f77_integer CONST_REF a1, float *a2, f77_integer CONST_REF a3, float *a4, f77_integer CONST_REF a5, const float *a6, const float *a7) {
      F77_FUNC(srot)(a1, a2, a3, a4, a5, a6, a7);
    }
    static void rotg(float *a1, float *a2, float *a3, float *a4) {
      F77_FUNC(srotg)(a1, a2, a3, a4);
    }
    static void rotm(f77_integer CONST_REF a1, float *a2, f77_integer CONST_REF a3, float *a4, f77_integer CONST_REF a5, const float *a6) {
      F77_FUNC(srotm)(a1, a2, a3, a4, a5, a6);
    }
    static void rotmg(float *a1, float *a2, float *a3, const float *a4, float *a5) {
      F77_FUNC(srotmg)(a1, a2, a3, a4, a5);
    }
    static void swap(f77_integer CONST_REF a1, float *a2, f77_integer CONST_REF a3, float *a4, f77_integer CONST_REF a5) {
      F77_FUNC(sswap)(a1, a2, a3, a4, a5);
    }
    static void copy(f77_integer CONST_REF a1, const float *a2, f77_integer CONST_REF a3, float *a4, f77_integer CONST_REF a5) {
      F77_FUNC(scopy)(a1, a2, a3, a4, a5);
    }
    static void axpy(f77_integer CONST_REF a1, float const a2, const float *a3, f77_integer CONST_REF a4, float *a5, f77_integer CONST_REF a6) {
      F77_FUNC(saxpy)(a1, a2, a3, a4, a5, a6);
    }
    static float dot(f77_integer CONST_REF a1, const float *a2, f77_integer CONST_REF a3, const float *a4, f77_integer CONST_REF a5) {
      return  F77_FUNC(sdot)(a1, a2, a3, a4, a5);
    }
    static double dsdot(f77_integer CONST_REF a1, const float *a2, const float *a3, f77_integer CONST_REF a4, const float *a5, f77_integer CONST_REF a6) {
      return F77_FUNC(sdsdot)(a1, a2, a3, a4, a5, a6);
    }
    static void scal(f77_integer CONST_REF a1, float const a2, float *a3, f77_integer CONST_REF a4) {
      F77_FUNC(sscal)(a1, a2, a3, a4);
    }
    static float nrm2(f77_integer CONST_REF a1, const float *a2, f77_integer CONST_REF a3) {
      return  F77_FUNC(snrm2)(a1, a2, a3);
    }
    static float asum(f77_integer CONST_REF a1, const float *a2, f77_integer CONST_REF a3) {
      return F77_FUNC(sasum)(a1, a2, a3);
    }
    static index_t iamax(f77_integer CONST_REF a1, const float *a2, f77_integer CONST_REF a3) {
      return F77_FUNC(isamax)(a1, a2, a3);
    }
    static void gemv(const char *a1, f77_integer CONST_REF a2, f77_integer CONST_REF a3, float const a4, const float *a5, f77_integer CONST_REF a6, const float *a7, f77_integer CONST_REF a8, float CONST_REF a9, float *a10, index_t CONST_REF a11) {
      F77_FUNC(sgemv)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
    }
    static void gbmv(const char *a1, f77_integer CONST_REF a2, f77_integer CONST_REF a3, f77_integer CONST_REF a4, f77_integer CONST_REF a5, f77_real CONST_REF a6, const float *a7, f77_integer CONST_REF a8, const float *a9, index_t CONST_REF a10, f77_real CONST_REF a11, float *a12, index_t CONST_REF a13) {
      F77_FUNC(sgbmv)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
    }
    static void symv(const char *a1, f77_integer CONST_REF a2, const float *a3, const float *a4, f77_integer CONST_REF a5, const float *a6, f77_integer CONST_REF a7, const float *a8, float *a9, f77_integer CONST_REF a10) {
      F77_FUNC(ssymv)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
    }
    static void sbmv(const char *a1, f77_integer CONST_REF a2, f77_integer CONST_REF a3, const float *a4, const float *a5, f77_integer CONST_REF a6, const float *a7, f77_integer CONST_REF a8, const float *a9, float *a10, f77_integer CONST_REF a11) {
      F77_FUNC(ssbmv)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
    }
    static void spmv(const char *a1, f77_integer CONST_REF a2, const float *a3, const float *a4, const float *a5, f77_integer CONST_REF a6, const float *a7, float *a8, f77_integer CONST_REF a9) {
      F77_FUNC(sspmv)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
    }
    static void trmv(const char *a1, const char *a2, const char *a3, f77_integer CONST_REF a4, const float *a5, f77_integer CONST_REF a6, float *a7, f77_integer CONST_REF a8) {
      F77_FUNC(strmv)(a1, a2, a3, a4, a5, a6, a7, a8);
    }
    static void tbmv(const char *a1, const char *a2, const char *a3, f77_integer CONST_REF a4, f77_integer CONST_REF a5, const float *a6, f77_integer CONST_REF a7, float *a8, f77_integer CONST_REF a9) {
      F77_FUNC(stbmv)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
    }
    static void trsv(const char *a1, const char *a2, const char *a3, f77_integer CONST_REF a4, const float *a5, f77_integer CONST_REF a6, float *a7, f77_integer CONST_REF a8) {
      F77_FUNC(strsv)(a1, a2, a3, a4, a5, a6, a7, a8);
    }
    static void tbsv(const char *a1, const char *a2, const char *a3, f77_integer CONST_REF a4, f77_integer CONST_REF a5, const float *a6, f77_integer CONST_REF a7, float *a8, f77_integer CONST_REF a9) {
      F77_FUNC(stbsv)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
    }
    static void tpmv(const char *a1, const char *a2, const char *a3, f77_integer CONST_REF a4, const float *a5, float *a6, f77_integer CONST_REF a7) {
      F77_FUNC(stpmv)(a1, a2, a3, a4, a5, a6, a7);
    }
    static void tpsv(const char *a1, const char *a2, const char *a3, f77_integer CONST_REF a4, const float *a5, float *a6, f77_integer CONST_REF a7) {
      F77_FUNC(stpsv)(a1, a2, a3, a4, a5, a6, a7);
    }
    static void ger(f77_integer CONST_REF a1, f77_integer CONST_REF a2, const float *a3, const float *a4, f77_integer CONST_REF a5, const float *a6, f77_integer CONST_REF a7, float *a8, f77_integer CONST_REF a9) {
      F77_FUNC(sger)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
    }
    static void syr(const char *a1, f77_integer CONST_REF a2, const float *a3, const float *a4, f77_integer CONST_REF a5, float *a6, f77_integer CONST_REF a7) {
      F77_FUNC(ssyr)(a1, a2, a3, a4, a5, a6, a7);
    }
    static void spr(const char *a1, f77_integer CONST_REF a2, const float *a3, const float *a4, f77_integer CONST_REF a5, float *a6) {
      F77_FUNC(sspr)(a1, a2, a3, a4, a5, a6);
    }
    static void spr2(const char *a1, f77_integer CONST_REF a2, const float *a3, const float *a4, f77_integer CONST_REF a5, const float *a6, f77_integer CONST_REF a7, float *a8) {
      F77_FUNC(sspr2)(a1, a2, a3, a4, a5, a6, a7, a8);
    }
    static void syr2(const char *a1, f77_integer CONST_REF a2, const float *a3, const float *a4, f77_integer CONST_REF a5, const float *a6, f77_integer CONST_REF a7, float *a8, f77_integer CONST_REF a9) {
      F77_FUNC(ssyr2)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
    }
    static void gemm(const char *a1, const char *a2, f77_integer CONST_REF a3, f77_integer CONST_REF a4, f77_integer CONST_REF a5, float const a6, const float *a7, f77_integer CONST_REF a8, const float *a9, index_t CONST_REF a10, float CONST_REF a11, float *a12, index_t CONST_REF a13) {
      F77_FUNC(sgemm)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,  a11, a12, a13);
    }
    static void symm(const char *a1, const char *a2, f77_integer CONST_REF a3, f77_integer CONST_REF a4, const float *a5, const float *a6, f77_integer CONST_REF a7, const float *a8, f77_integer CONST_REF a9, const float *a10, float *a11, f77_integer CONST_REF a12) {
      F77_FUNC(ssymm)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
    }
    static void syrk(const char *a1, const char *a2, f77_integer CONST_REF a3, f77_integer CONST_REF a4, const float *a5, const float *a6, f77_integer CONST_REF a7, const float *a8, float *a9, f77_integer CONST_REF a10) {
      F77_FUNC(ssyrk)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
    }
    static void syr2k(const char *a1, const char *a2, f77_integer CONST_REF a3, f77_integer CONST_REF a4, const float *a5, const float *a6, f77_integer CONST_REF a7, const float *a8, f77_integer CONST_REF a9, const float *a10, float *a11, f77_integer CONST_REF a12) {
      F77_FUNC(ssyr2k)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
    }
    static void trmm(const char *a1, const char *a2, const char *a3, const char *a4, f77_integer CONST_REF a5, f77_integer CONST_REF a6, const float *a7, const float *a8, f77_integer CONST_REF a9, float *a10, f77_integer CONST_REF a11) {
      F77_FUNC(strmm)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
    }
    static void trsm(const char *a1, const char *a2, const char *a3, const char *a4, f77_integer CONST_REF a5 , f77_integer CONST_REF a6, const float *a7, const float *a8, f77_integer CONST_REF a9, float *a10, f77_integer CONST_REF a11) {
      F77_FUNC(strsm)(a1, a2, a3, a4, a5 , a6, a7, a8, a9, a10, a11);
    }
};

template<>
class CppBlas<double> {
  public:
    static void rot(f77_integer CONST_REF a1, double *a2, f77_integer CONST_REF a3, double *a4, f77_integer CONST_REF a5, const double *a6, const double *a7) {
      F77_FUNC(drot)(a1, a2, a3, a4, a5, a6, a7);
    }
    static void rotg(double *a1, double *a2, double *a3, double *a4) {
      F77_FUNC(drotg)(a1, a2, a3, a4);
    }
    static void rotm(f77_integer CONST_REF a1, double *a2, f77_integer CONST_REF a3, double *a4, f77_integer CONST_REF a5, const double *a6) {
      F77_FUNC(drotm)(a1, a2, a3, a4, a5, a6);
    }
    static void rotmg(double *a1, double *a2, double *a3, const double *a4, double *a5) {
      F77_FUNC(drotmg)(a1, a2, a3, a4, a5);
    }
    static void swap(f77_integer CONST_REF a1, double *a2, f77_integer CONST_REF a3, double *a4, f77_integer CONST_REF a5) {
      F77_FUNC(dswap)(a1, a2, a3, a4, a5);
    }
    static void copy(f77_integer CONST_REF a1, const double *a2, f77_integer CONST_REF a3, double *a4, f77_integer CONST_REF a5) {
      F77_FUNC(dcopy)(a1, a2, a3, a4, a5);
    }
    static void axpy(f77_integer CONST_REF a1, double const a2, const double *a3, f77_integer CONST_REF a4, double *a5, f77_integer CONST_REF a6) {
      F77_FUNC(daxpy)(a1, a2, a3, a4, a5, a6);
    }
    static double dot(f77_integer CONST_REF a1, const double *a2, f77_integer CONST_REF a3, const double *a4, f77_integer CONST_REF a5) {
      return F77_FUNC(ddot)(a1, a2, a3, a4, a5);
    }
    static void scal(f77_integer CONST_REF a1, double const a2, double *a3, f77_integer CONST_REF a4) {
      F77_FUNC(dscal)(a1, a2, a3, a4);
    }
    static double nrm2(f77_integer CONST_REF a1, const double *a2, f77_integer CONST_REF a3) {
      return  F77_FUNC(dnrm2)(a1, a2, a3);
    }
    static double asum(f77_integer CONST_REF a1, const double *a2, f77_integer CONST_REF a3) {
      return F77_FUNC(dasum)(a1, a2, a3);
    }
    static index_t iamax(f77_integer CONST_REF a1, const double *a2, f77_integer CONST_REF a3) {
      return F77_FUNC(idamax)(a1, a2, a3);
    }
    static void gemv(const char *a1, f77_integer CONST_REF a2, f77_integer CONST_REF a3, double const a4, const double *a5, f77_integer CONST_REF a6, const double *a7, f77_integer CONST_REF a8, double CONST_REF a9, double *a10, index_t CONST_REF a11) {
      F77_FUNC(dgemv)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
    }
    static void gbmv(const char *a1, f77_integer CONST_REF a2, f77_integer CONST_REF a3, f77_integer CONST_REF a4, f77_integer CONST_REF a5, f77_double CONST_REF a6, const double *a7, f77_integer CONST_REF a8, const double *a9, index_t CONST_REF a10, f77_double CONST_REF a11, double *a12, index_t CONST_REF a13) {
      F77_FUNC(dgbmv)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
    }
    static void symv(const char *a1, f77_integer CONST_REF a2, const double *a3, const double *a4, f77_integer CONST_REF a5, const double *a6, f77_integer CONST_REF a7, const double *a8, double *a9, f77_integer CONST_REF a10) {
      F77_FUNC(dsymv)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
    }
    static void sbmv(const char *a1, f77_integer CONST_REF a2, f77_integer CONST_REF a3, const double *a4, const double *a5, f77_integer CONST_REF a6, const double *a7, f77_integer CONST_REF a8, const double *a9, double *a10, f77_integer CONST_REF a11) {
      F77_FUNC(dsbmv)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
    }
    static void spmv(const char *a1, f77_integer CONST_REF a2, const double *a3, const double *a4, const double *a5, f77_integer CONST_REF a6, const double *a7, double *a8, f77_integer CONST_REF a9) {
      F77_FUNC(dspmv)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
    }
    static void trmv(const char *a1, const char *a2, const char *a3, f77_integer CONST_REF a4, const double *a5, f77_integer CONST_REF a6, double *a7, f77_integer CONST_REF a8) {
      F77_FUNC(dtrmv)(a1, a2, a3, a4, a5, a6, a7, a8);
    }
    static void tbmv(const char *a1, const char *a2, const char *a3, f77_integer CONST_REF a4, f77_integer CONST_REF a5, const double *a6, f77_integer CONST_REF a7, double *a8, f77_integer CONST_REF a9) {
      F77_FUNC(dtbmv)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
    }
    static void trsv(const char *a1, const char *a2, const char *a3, f77_integer CONST_REF a4, const double *a5, f77_integer CONST_REF a6, double *a7, f77_integer CONST_REF a8) {
      F77_FUNC(dtrsv)(a1, a2, a3, a4, a5, a6, a7, a8);
    }
    static void tbsv(const char *a1, const char *a2, const char *a3, f77_integer CONST_REF a4, f77_integer CONST_REF a5, const double *a6, f77_integer CONST_REF a7, double *a8, f77_integer CONST_REF a9) {
      F77_FUNC(dtbsv)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
    }
    static void tpmv(const char *a1, const char *a2, const char *a3, f77_integer CONST_REF a4, const double *a5, double *a6, f77_integer CONST_REF a7) {
      F77_FUNC(dtpmv)(a1, a2, a3, a4, a5, a6, a7);
    }
    static void tpsv(const char *a1, const char *a2, const char *a3, f77_integer CONST_REF a4, const double *a5, double *a6, f77_integer CONST_REF a7) {
      F77_FUNC(dtpsv)(a1, a2, a3, a4, a5, a6, a7);
    }
    static void ger(f77_integer CONST_REF a1, f77_integer CONST_REF a2, const double *a3, const double *a4, f77_integer CONST_REF a5, const double *a6, f77_integer CONST_REF a7, double *a8, f77_integer CONST_REF a9) {
      F77_FUNC(dger)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
    }
    static void syr(const char *a1, f77_integer CONST_REF a2, const double *a3, const double *a4, f77_integer CONST_REF a5, double *a6, f77_integer CONST_REF a7) {
      F77_FUNC(dsyr)(a1, a2, a3, a4, a5, a6, a7);
    }
    static void spr(const char *a1, f77_integer CONST_REF a2, const double *a3, const double *a4, f77_integer CONST_REF a5, double *a6) {
      F77_FUNC(dspr)(a1, a2, a3, a4, a5, a6);
    }
    static void spr2(const char *a1, f77_integer CONST_REF a2, const double *a3, const double *a4, f77_integer CONST_REF a5, const double *a6, f77_integer CONST_REF a7, double *a8) {
      F77_FUNC(dspr2)(a1, a2, a3, a4, a5, a6, a7, a8);
    }
    static void syr2(const char *a1, f77_integer CONST_REF a2, const double *a3, const double *a4, f77_integer CONST_REF a5, const double *a6, f77_integer CONST_REF a7, double *a8, f77_integer CONST_REF a9) {
      F77_FUNC(dsyr2)(a1, a2, a3, a4, a5, a6, a7, a8, a9);
    }
    static void gemm(const char *a1, const char *a2, f77_integer CONST_REF a3, f77_integer CONST_REF a4, f77_integer CONST_REF a5, double const a6, const double *a7, f77_integer CONST_REF a8, const double *a9, index_t CONST_REF a10, double CONST_REF a11, double *a12, index_t CONST_REF a13) {
      F77_FUNC(dgemm)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,  a11, a12, a13);
    }
    static void symm(const char *a1, const char *a2, f77_integer CONST_REF a3, f77_integer CONST_REF a4, const double *a5, const double *a6, f77_integer CONST_REF a7, const double *a8, f77_integer CONST_REF a9, const double *a10, double *a11, f77_integer CONST_REF a12) {
      F77_FUNC(dsymm)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
    }
    static void syrk(const char *a1, const char *a2, f77_integer CONST_REF a3, f77_integer CONST_REF a4, const double *a5, const double *a6, f77_integer CONST_REF a7, const double *a8, double *a9, f77_integer CONST_REF a10) {
      F77_FUNC(dsyrk)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
    }
    static void syr2k(const char *a1, const char *a2, f77_integer CONST_REF a3, f77_integer CONST_REF a4, const double *a5, const double *a6, f77_integer CONST_REF a7, const double *a8, f77_integer CONST_REF a9, const double *a10, double *a11, f77_integer CONST_REF a12) {
      F77_FUNC(dsyr2k)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
    }
    static void trmm(const char *a1, const char *a2, const char *a3, const char *a4, f77_integer CONST_REF a5, f77_integer CONST_REF a6, const double *a7, const double *a8, f77_integer CONST_REF a9, double *a10, f77_integer CONST_REF a11) {
      F77_FUNC(dtrmm)(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
    }
    static void trsm(const char *a1, const char *a2, const char *a3, const char *a4, f77_integer CONST_REF a5 , f77_integer CONST_REF a6, const double *a7, const double *a8, f77_integer CONST_REF a9, double *a10, f77_integer CONST_REF a11) {
      F77_FUNC(dtrsm)(a1, a2, a3, a4, a5 , a6, a7, a8, a9, a10, a11);
    }
};  //CppBlas

}; // namespace la
}; // namespace fl
#endif
