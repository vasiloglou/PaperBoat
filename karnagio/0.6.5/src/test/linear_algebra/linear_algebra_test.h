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
/**
 * @file uselapack_test.h
 *
 * Tests for LAPACK integration.
 */

#include "boost/test/unit_test.hpp"

#include "fastlib/la/linear_algebra.h"
#include "fastlib/dense/matrix.h"
#include "fastlib/dense/small_matrix.h"

/**
 * Creates a matrix locally.
 * The matrix cotents are column-major.
 */
#define MAKE_MATRIX_TRANS(Precision, name, n_rows, n_cols, contents ...) \
  Precision name ## _values [] = { contents }; \
  DEBUG_ASSERT(sizeof(name ## _values) / sizeof(Precision) == n_rows * n_cols); \
  fl::dense::Matrix<Precision, false> name; \
  name.Alias(name ## _values, (n_rows), (n_cols));

/**
 * Creates a vector locally.
 * The matrix cotents are column-major.
 */
#define MAKE_VECTOR(Precision, name, length, contents ...) \
  Precision name ## _values [] = { contents }; \
  DEBUG_ASSERT(sizeof(name ## _values) / sizeof(Precision) == (length)); \
  fl::dense::Matrix<Precision, true> name; \
  name.Alias(name ## _values, (length));
template<typename Precision>
bool VectorApproxEqual(const fl::dense::Matrix<Precision, true>& a,
                       const fl::dense::Matrix<Precision, true>& b,
                       Precision eps) {
  if (a.length() != b.length()) {
    fprintf(stderr, "XXX Size mismatch.\n");
    return false;
  }

  int wrong = 0;
  Precision max_diff = 0;

  for (index_t i = 0; i < a.length(); i++) {
    Precision diff = fabs(a.get(i) - b.get(i));
    max_diff = std::max(max_diff, diff);
    if (!(diff <= eps)) {
      wrong++;
      if (wrong <= 3) {
        fprintf(stderr, "XXX Mismatch (index %d) zero-based (%e)\n",
                i, diff);
      }
    }
  }

  if (wrong) {
    fprintf(stderr, "XXX Total %d mismatches, max diff %e.\n",
            wrong, max_diff);
  }

  return wrong == 0;
}
template<typename Precision>
void AssertApproxVector(const fl::dense::Matrix<Precision, true>& a,
                        const fl::dense::Matrix<Precision, true>& b, Precision eps) {
  if (!VectorApproxEqual(a, b, eps)) {
    a.PrintDebug("a");
    b.PrintDebug("b");
    abort();
  }
  //fprintf(stderr, "... Correct vector!\n");
}

template<typename Precision>
bool MatrixApproxEqual(const fl::dense::Matrix<Precision, false>& a,
                       const fl::dense::Matrix<Precision, false>& b,
                       Precision eps) {
  if (a.n_rows() != b.n_rows() || a.n_cols() != b.n_cols()) {
    fprintf(stderr, "XXX Size mismatch.\n");
    return false;
  }

  int wrong = 0;
  Precision max_diff = 0;

  for (index_t c = 0; c < a.n_cols(); c++) {
    for (index_t r = 0; r < a.n_rows(); r++) {
      Precision diff = fabs(a.get(r, c) - b.get(r, c));
      max_diff = std::max(max_diff, diff);
      if (!(diff <= eps)) {
        wrong++;
        if (wrong <= 3) {
          fprintf(stderr, "XXX Mismatch (%d, %d) zero-based (%e)\n",
                  r, c, diff);
        }
      }
    }
  }

  if (wrong) {
    fprintf(stderr, "XXX Total %d mismatches, max diff %e.\n",
            wrong, max_diff);
  }

  return wrong == 0;
}

template<typename Precision>
void AssertApproxMatrix(const fl::dense::Matrix<Precision, false>& a,
                        const fl::dense::Matrix<Precision, false>& b,
                        Precision eps) {
  if (!MatrixApproxEqual(a, b, eps)) {
    a.PrintDebug("a");
    b.PrintDebug("b");
    abort();
  }
  //fprintf(stderr, "... Correct matrix!\n");
}

template<typename Precision>
void AssertExactMatrix(const fl::dense::Matrix<Precision, false>& a,
                       const fl::dense::Matrix<Precision, false>& b) {
  AssertApproxMatrix<Precision>(a, b, 0);
}

template<typename Precision>
void AssertApproxTransMatrix(const fl::dense::Matrix<Precision, false>& a,
                             const fl::dense::Matrix<Precision, false>& b,
                             Precision eps) {
  fl::dense::Matrix<Precision, false> a_trans;
  fl::la::Transpose<fl::la::Init>(a, &a_trans);
  if (!MatrixApproxEqual(a_trans, b, eps)) {
    a_trans.PrintDebug("a_trans");
    b.PrintDebug("b");
    abort();
  }
}

template<typename Precision>
void TestVectorDot() {

  NOTIFY("TestVectorDot: Starting");
  MAKE_VECTOR(Precision, a, 4,    2, 1, 4, 5);
  MAKE_VECTOR(Precision, b, 4,    3, 0, 2, -1);

  //BOOST_CHECK_EQUAL(F77_FUNC(ddot)(4, a.ptr(), 1, a.ptr(), 1),  4+1+16+25);
  BOOST_CHECK_EQUAL(fl::dense::ops::Dot(a, a), 4 + 1 + 16 + 25);
  BOOST_CHECK_EQUAL(fl::dense::ops::Dot(a, b), 6 + 0 + 8 - 5);
  BOOST_CHECK_CLOSE(fl::dense::ops::LengthEuclidean(b), (Precision)sqrt(9 + 0 + 4 + 1), (Precision)1.0e-5);
  BOOST_CHECK_CLOSE(fl::dense::ops::LengthEuclidean(a), (Precision)sqrt(4 + 1 + 16 + 25), (Precision)1.0e-5);

  NOTIFY("TestVectorDot: Finished");
}

// ---- INCLUDED FROM ORIGINAL LA TEST -----
// ----
// ----
// ---- (Except distance tests are omitted)
template<typename Precision>
void MakeCountMatrix(index_t n_rows, index_t n_cols,
                     fl::dense::Matrix<Precision, false> *m) {

  m->Init(n_rows, n_cols);

  for (index_t c = 0; c < n_cols; c++) {
    for (index_t r = 0; r < n_rows; r++) {
      m->set(r, c, r + c);
    }
  }
}

template<typename Precision>
void MakeConstantMatrix(index_t n_rows, index_t n_cols, Precision v, fl::dense::Matrix<Precision, false> *m) {
  m->Init(n_rows, n_cols);

  for (index_t c = 0; c < n_cols; c++) {
    for (index_t r = 0; r < n_rows; r++) {
      m->set(r, c, v);
    }
  }
}

/** Tests level 1 BLAS-ish stuff. */
template<typename Precision>
void TestMatrixSimpleMath() {

  NOTIFY("TestMatrixSimpleMath: Starting");

  fl::dense::Matrix<Precision, false> m1;
  fl::dense::Matrix<Precision, false> m2;
  fl::dense::Matrix<Precision, false> m3;
  fl::dense::Matrix<Precision, false> m4;
  fl::dense::Matrix<Precision, false> m5;

  MakeCountMatrix<Precision>(3, 4, &m1);
  MakeCountMatrix<Precision>(3, 4, &m2);
  fl::dense::ops::AddTo<Precision>(m2, &m1);
  fl::dense::ops::Add<fl::la::Init>(m1, m2, &m3);
  BOOST_CHECK(m3.get(0, 0) == 0);
  BOOST_CHECK(m3.get(2, 3) == (2 + 3)*3);
  MakeConstantMatrix<Precision>(3, 4, 7.0, &m4);
  fl::dense::ops::Add<fl::la::Init>(m1, m4, &m5);
  BOOST_CHECK(m5.get(2, 3) == (2 + 3)*2 + 7.0);
  BOOST_CHECK(m5.get(1, 3) == (1 + 3)*2 + 7.0);
  BOOST_CHECK(m5.get(1, 0) == (1 + 0)*2 + 7.0);

  NOTIFY("TestMatrixSimpleMath: Finished");
}

template<typename Precision>
void MakeCountVector(index_t n, fl::dense::Matrix<Precision, true> *v) {
  v->Init(n);

  for (index_t c = 0; c < n; c++) {
    (*v)[c] = c;
  }
}

template<typename Precision>
void MakeConstantVector(index_t n, Precision d, fl::dense::Matrix<Precision, true> *v) {
  v->Init(n);

  for (index_t c = 0; c < n; c++) {
    (*v)[c] = d;
  }
}

/** Tests level 1 BLAS-ish stuff. */
template<typename Precision>
void TestVectorSimpleMath() {

  NOTIFY("TestVectorSimpleMath: Starting");

  fl::dense::Matrix<Precision, true> v1;
  fl::dense::Matrix<Precision, true> v2;
  fl::dense::Matrix<Precision, true> v3;
  fl::dense::Matrix<Precision, true> v4;
  fl::dense::Matrix<Precision, true> v5;

  MakeCountVector<Precision>(6, &v1);
  MakeCountVector<Precision>(6, &v2);
  fl::dense::ops::AddTo(v2, &v1);
  fl::dense::ops::Add<fl::la::Init>(v1, v2, &v3);
  BOOST_CHECK(v3[0] == 0);
  BOOST_CHECK(v3[5] == (5)*3);
  MakeConstantVector<Precision>(6, 7.0, &v4);
  fl::dense::ops::Add<fl::la::Init>(v1, v4, &v5);
  BOOST_CHECK(v5[5] == (5)*2 + 7.0);
  BOOST_CHECK(v5[4] == (4)*2 + 7.0);
  BOOST_CHECK(v5[1] == (1)*2 + 7.0);

  NOTIFY("TestVectorSimpleMath: Finished");
}



// ---- -------- ----
// ---- -------- ----
// ---- -------- ----
// ---- NEW TESTS ----
// ---- -------- ----
// ---- -------- ----
// ---- -------- ----
template<typename Precision>
void TestMultiply() {

  NOTIFY("TestMultiply: Starting");

  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
                    3, 1, 4,
                    1, 5, 9,
                    2, 6, 5);

  MAKE_MATRIX_TRANS(Precision, b, 3, 4,
                    3, 5, 8,
                    9, 7, 9,
                    3, 2, 3,
                    8, 4, 6);

  MAKE_MATRIX_TRANS(Precision, product_expect, 3, 4,
                    30, 76, 97,
                    52, 98, 144,
                    17, 31, 45,
                    40, 64, 98);

  fl::dense::Matrix<Precision, false> product_actual;

  // product_actual is uninitialized
  fl::dense::ops::Mul<fl::la::Init>(a, b, &product_actual);
  AssertExactMatrix<Precision>(product_expect, product_actual);

  product_actual.SetZero();
  fl::dense::ops::Mul<fl::la::Overwrite, fl::la::NoTrans, fl::la::NoTrans>(a, b, &product_actual);
  AssertExactMatrix<Precision>(product_expect, product_actual);

  MAKE_MATRIX_TRANS(Precision, product_expect_transa, 3, 4,
                    46, 100, 76,
                    70, 125, 105,
                    23, 40, 33,
                    52, 82, 70);

  fl::dense::Matrix<Precision, false> product_actual_transa;
  fl::dense::ops::Mul<fl::la::Init, fl::la::Trans, fl::la::NoTrans>(a, b, &product_actual_transa);
  AssertExactMatrix<Precision>(product_expect_transa, product_actual_transa);

  fl::dense::Matrix<Precision, false> a_t, b_t;
  fl::dense::ops::Transpose<fl::la::Init>(a, &a_t);
  fl::dense::ops::Transpose<fl::la::Init>(b, &b_t);

  product_actual_transa.Destruct();
  fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::Trans>(
    a_t, b_t, &product_actual_transa);
  AssertExactMatrix<Precision>(product_expect_transa, product_actual_transa);

  product_actual.SetZero();
  fl::dense::ops::Mul<fl::la::Overwrite, fl::la::Trans, fl::la::NoTrans>(a, b, &product_actual_transa);
  AssertExactMatrix<Precision>(product_expect_transa, product_actual_transa);

  // test matrix-vector multiplication
  MAKE_VECTOR(Precision, v1, 3,     9, 1, 2);
  MAKE_VECTOR(Precision, a_v1, 3,   32, 26, 55);
  MAKE_VECTOR(Precision, v1_a, 3,   36, 32, 34);
  MAKE_VECTOR(Precision, v2, 3,     2, 3, 4);
  MAKE_VECTOR(Precision, a_v2, 3,   17, 41, 55);
  MAKE_VECTOR(Precision, v2_a, 3,   25, 53, 42);

  fl::dense::Matrix<Precision, true> a_v1_actual;
  fl::dense::ops::Mul<fl::la::Init>(a, v1, &a_v1_actual);
  AssertApproxVector<Precision>(a_v1, a_v1_actual, 0);

  fl::dense::Matrix<Precision, true> a_v2_actual;
  a_v2_actual.Init(3);
  fl::dense::ops::Mul<fl::la::Overwrite, fl::la::NoTrans>(a, v2, &a_v2_actual);
  AssertApproxVector<Precision>(a_v2, a_v2_actual, 0);

  fl::dense::Matrix<Precision, true> v1_a_actual;
  fl::dense::ops::Mul<fl::la::Init, fl::la::Trans>(a, v1,  &v1_a_actual);
  AssertApproxVector<Precision>(v1_a, v1_a_actual, 0);

  fl::dense::Matrix<Precision, true> v2_a_actual;
  v2_a_actual.Init(3);
  fl::dense::ops::Mul<fl::la::Overwrite, fl::la::Trans>(a, v2, &v2_a_actual);
  AssertApproxVector<Precision>(v2_a, v2_a_actual, 0);

  // Test non-square matrices (we had some bad debug checks)
  MAKE_VECTOR(Precision, v3, 4,     1, 2, 3, 4);
  MAKE_VECTOR(Precision, b_v3, 3,   62, 41, 59);
  MAKE_VECTOR(Precision, v1_b, 4,   48, 106, 35, 88);

  fl::dense::SmallVector<Precision, 3> b_v3_actual;

  fl::dense::ops::Mul<fl::la::Overwrite, fl::la::NoTrans>(b, v3, &b_v3_actual);
  AssertApproxVector<Precision>(b_v3, b_v3_actual, 0);

  fl::dense::Matrix<Precision, true> v1_b_actual;
  fl::dense::ops::Mul<fl::la::Init, fl::la::Trans>(b, v1, &v1_b_actual);
  AssertApproxVector<Precision>(v1_b, v1_b_actual, 0);

  NOTIFY("TestMultiplfy: Finished");
}

template<typename Precision>
void TestApplyFunc(bool print_debug_info) {

  NOTIFY("TestApplyFunc: Starting");

  Precision precision = 1.0e-3;
  for (index_t i = 0; i < 10; i++) {

    // Random matrix.
    fl::dense::Matrix<Precision, false> x;
    fl::dense::Matrix<Precision, false> y, sine_answer, cosine_answer, tangent_answer,
    exp_answer, log_answer, pow_answer;

    index_t num_rows = fl::math::Random(10) + 1;
    index_t num_cols = fl::math::Random(10) + 1;
    index_t length = num_rows * num_cols;
    x.Init(num_rows, num_cols);
    y.Init(num_rows, num_cols);
    sine_answer.Init(num_rows, num_cols);
    cosine_answer.Init(num_rows, num_cols);
    tangent_answer.Init(num_rows, num_cols);
    exp_answer.Init(num_rows, num_cols);
    log_answer.Init(num_rows, num_cols);
    pow_answer.Init(num_rows, num_cols);

    if (print_debug_info) {
      NOTIFY("Generating two matrices with %d rows %d cols", num_rows,
             num_cols);
    }

    for (index_t j = 0; j < num_cols; j++) {
      for (index_t i = 0; i < num_rows; i++) {
        x.set(i, j, fl::math::Random<Precision>(-5, 5));
      }
    }

    // Test Sin.
    fl::dense::ops::Sin<Precision> sine_function;
    Precision trig_frequency = fl::math::Random<Precision>(0, 5);
    sine_function.Init(trig_frequency);
    fl::dense::ops::DotFunExpert(length, sine_function, x.ptr(), y.ptr());
    for (index_t j = 0; j < num_cols; j++) {
      for (index_t i = 0; i < num_rows; i++) {
        sine_answer.set(i, j, sin(trig_frequency * x.get(i, j)));
      }
    }
    AssertApproxMatrix<Precision>(sine_answer, y, precision);

    // Test Cos.
    fl::dense::ops::Cos<Precision> cosine_function;
    cosine_function.Init(trig_frequency);
    fl::dense::ops::DotFunExpert(length, cosine_function, x.ptr(), y.ptr());
    for (index_t j = 0; j < num_cols; j++) {
      for (index_t i = 0; i < num_rows; i++) {
        cosine_answer.set(i, j, cos(trig_frequency * x.get(i, j)));
      }
    }
    AssertApproxMatrix<Precision>(cosine_answer, y, precision);

    // Test Tan.
    fl::dense::ops::Tan<Precision> tangent_function;
    tangent_function.Init(trig_frequency);
    fl::dense::ops::DotFunExpert(length, tangent_function, x.ptr(), y.ptr());
    for (index_t j = 0; j < num_cols; j++) {
      for (index_t i = 0; i < num_rows; i++) {
        tangent_answer.set(i, j, tan(trig_frequency * x.get(i, j)));
      }
    }
    AssertApproxMatrix<Precision>(tangent_answer, y, precision);

    // Test Exp.
    fl::dense::ops::Exp<Precision> exp_function;
    Precision exp_frequency = fl::math::Random<Precision>(0, 10) + 1;
    exp_function.Init(exp_frequency);
    fl::dense::ops::DotFunExpert(length, exp_function, x.ptr(), y.ptr());
    for (index_t j = 0; j < num_cols; j++) {
      for (index_t i = 0; i < num_rows; i++) {
        exp_answer.set(i, j, exp(exp_frequency * x.get(i, j)));
      }
    }
    AssertApproxMatrix<Precision>(exp_answer, y, precision);

    // We are not testing Log because of Log(0) results in NaN...

    // Test Pow.
    fl::dense::ops::Pow<Precision> pow_function;
    pow_function.Init(exp_frequency);
    fl::dense::ops::DotFunExpert(length, pow_function, x.ptr(), y.ptr());
    for (index_t j = 0; j < num_cols; j++) {
      for (index_t i = 0; i < num_rows; i++) {
        pow_answer.set(i, j, pow(exp_frequency, x.get(i, j)));
      }
    }
    AssertApproxMatrix<Precision>(pow_answer, y, precision);

  }
  NOTIFY("TestApplyFunc: Finished");
}

template<typename Precision>
void TestEntrywiseOperations(bool print_debug_info) {

  NOTIFY("TestEntrywiseOperations: Starting");

  Precision precision = 1.0e-3;
  for (index_t i = 0; i < 10; i++) {

    // x and y hold the fixed values, a holds the answer for the
    // Hadamard product, and z is always the temporary variable. a_pow
    // holds the answer for the entrywise integer power, a_pow = x^pow
    fl::dense::Matrix<Precision, false> x;
    fl::dense::Matrix<Precision, false> y;
    fl::dense::Matrix<Precision, false> z;
    fl::dense::Matrix<Precision, false> a;
    fl::dense::Matrix<Precision, false> a_pow;

    index_t num_rows = fl::math::Random(10) + 1;
    const int power = 5;
    index_t num_cols = fl::math::Random(10) + 1;
    index_t length = num_rows * num_cols;
    x.Init(num_rows, num_cols);
    y.Init(num_rows, num_cols);
    z.Init(num_rows, num_cols);
    a.Init(num_rows, num_cols);
    a_pow.Init(num_rows, num_cols);

    if (print_debug_info) {
      NOTIFY("Generating two matrices with %d rows %d cols", num_rows,
             num_cols);
    }

    for (index_t j = 0; j < num_cols; j++) {
      for (index_t i = 0; i < num_rows; i++) {
        x.set(i, j, fl::math::Random<Precision>(-5, 5));
        y.set(i, j, fl::math::Random<Precision>(-5, 5));
        a.set(i, j, x.get(i, j) * y.get(i, j));
        a_pow.set(i, j, fl::math::Pow<Precision, power, 1>(x.get(i, j)));
      }
    }

    if (print_debug_info) {
      x.PrintDebug();
      y.PrintDebug();
      NOTIFY("Hadamard product should be: ");
      a.PrintDebug();
      NOTIFY("x raised to the power of %d should be: ", power);
      a_pow.PrintDebug();
    }

    // Test the expert Hadamard product.
    fl::dense::ops::DotMulExpert(length, x.ptr(), y.ptr(), z.ptr());
    AssertApproxMatrix<Precision>(a, z, precision);

    // Destroy and test the DotMul: first fl::la::Init and fl::la::Overwrite.
    z.Destruct();
    fl::dense::ops::DotMul<fl::la::Init>(x, y, &z);
    AssertApproxMatrix<Precision>(a, z, precision);
    fl::dense::ops::DotMul<fl::la::Overwrite>(x, y, &z);
    AssertApproxMatrix<Precision>(a, z, precision);

    // Test the DotMulTo.
    z.CopyValues(x);
    fl::dense::ops::DotMulTo(length, y.ptr(), z.ptr());
    AssertApproxMatrix<Precision>(a, z, precision);
    z.CopyValues(x);
    fl::dense::ops::DotMulTo(y, &z);
    AssertApproxMatrix<Precision>(a, z, precision);

    // Test the DotIntPowExpert.
    fl::dense::ops::DotIntPowExpert(length, power, x.ptr(), z.ptr());
    AssertApproxMatrix<Precision>(a_pow, z, precision);

    // Test the DotIntPow.
    z.Destruct();
    fl::dense::ops::DotIntPow<fl::la::Init>(power, x, &z);
    AssertApproxMatrix<Precision>(a_pow, z, precision);
    fl::dense::ops::DotIntPow<fl::la::Overwrite>(power, x, &z);
    AssertApproxMatrix<Precision>(a_pow, z, precision);

    // Test the DotIntPowTo.
    z.CopyValues(x);
    fl::dense::ops::DotIntPowTo(length, power, z.ptr());
    AssertApproxMatrix<Precision>(a_pow, z, precision);
    z.CopyValues(x);
    fl::dense::ops::DotIntPowTo(power, &z);
    AssertApproxMatrix<Precision>(a_pow, z, precision);

  } // end of repetition.

  NOTIFY("TestEntrywiseOperations: Finished");
}
