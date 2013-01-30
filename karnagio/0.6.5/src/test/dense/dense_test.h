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
#include "fastlib/dense/matrix.h"
#include "fastlib/dense/small_matrix.h"
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"

using namespace fl::dense;

/**
 * Creates a matrix locally.
 * The matrix cotents are column-major.
 */
#define MAKE_MATRIX_TRANS(Precision, name, n_rows, n_cols, contents ...) \
  Precision name ## _values [] = { contents }; \
  DEBUG_ASSERT(sizeof(name ## _values) / sizeof(Precision) == n_rows * n_cols); \
  Matrix<Precision, false> name; \
  name.Alias(name ## _values, (n_rows), (n_cols));

/**
 * Creates a vector locally.
 * The matrix cotents are column-major.
 */
#define MAKE_VECTOR(Precision, name, length, contents ...) \
  Precision name ## _values [] = { contents }; \
  DEBUG_ASSERT(sizeof(name ## _values) / sizeof(Precision) == (length)); \
  Matrix<Precision, true> name; \
  name.Alias(name ## _values, (length));

template<typename Precision>
bool VectorApproxEqual(const Matrix<Precision, true>& a,
                       const Matrix<Precision, true>& b,
                       Precision eps) {
  BOOST_REQUIRE_EQUAL(a.length(), b.length());

  Precision max_diff = 0.0;
  for (index_t i = 0; i < a.length(); i++) {
    Precision diff = fabs(a.get(i) - b.get(i));
    max_diff = std::max(max_diff, diff);
  }

  BOOST_CHECK_LE(max_diff, eps);

  return true;
}

template<typename Precision>
void AssertApproxVector(const Matrix<Precision, true>& a,
                        const Matrix<Precision, true>& b, Precision eps) {
  VectorApproxEqual(a, b, eps);
}

template<typename Precision>
bool MatrixApproxEqual(const Matrix<Precision, false>& a,
                       const Matrix<Precision, false>& b,
                       Precision eps) {
  BOOST_REQUIRE_EQUAL(a.n_rows(), b.n_rows());
  BOOST_REQUIRE_EQUAL(a.n_cols(), b.n_cols());

  Precision max_diff = 0;
  for (index_t c = 0; c < a.n_cols(); c++) {
    for (index_t r = 0; r < a.n_rows(); r++) {
      Precision diff = fabs(a.get(r, c) - b.get(r, c));
      max_diff = std::max(max_diff, diff);
    }
  }

  BOOST_CHECK_LE(max_diff, eps);

  return true;
}

template<typename Precision>
void AssertApproxMatrix(const Matrix<Precision, false>& a,
                        const Matrix<Precision, false>& b,
                        Precision eps) {
  MatrixApproxEqual(a, b, eps);
}

template<typename Precision>
void AssertExactMatrix(const Matrix<Precision, false>& a,
                       const Matrix<Precision, false>& b) {
  AssertApproxMatrix<Precision>(a, b, 0);
}

template<typename Precision>
void AssertApproxTransMatrix(const Matrix<Precision, false>& a,
                             const Matrix<Precision, false>& b,
                             Precision eps) {
  Matrix<Precision, false> a_trans;
  fl::dense::ops::Transpose<fl::la::Init>(a, &a_trans);
  MatrixApproxEqual(a_trans, b, eps);
}

template<typename Precision>
void TestVectorDot() {

  NOTIFY("TestVectorDot: Starting");
  MAKE_VECTOR(Precision, a, 4,    2, 1, 4, 5);
  MAKE_VECTOR(Precision, b, 4,    3, 0, 2, -1);

  //BOOST_CHECK_EQUAL(F77_FUNC(ddot)(4, a.ptr(), 1, a.ptr(), 1),  4+1+16+25);
  BOOST_CHECK_EQUAL(fl::dense::ops::Dot<Precision>(a, a), 4 + 1 + 16 + 25);
  BOOST_CHECK_EQUAL(fl::dense::ops::Dot<Precision>(a, b), 6 + 0 + 8 - 5);
  BOOST_CHECK_CLOSE(fl::dense::ops::LengthEuclidean<Precision>(b), (Precision)sqrt(9 + 0 + 4 + 1), (Precision)1.0e-5);
  BOOST_CHECK_CLOSE(fl::dense::ops::LengthEuclidean<Precision>(a), (Precision)sqrt(4 + 1 + 16 + 25), (Precision)1.0e-5);

  NOTIFY("TestVectorDot: Finished");
}

// ---- INCLUDED FROM ORIGINAL LA TEST -----
// ----
// ----
// ---- (Except distance tests are omitted)
template<typename Precision>
void MakeCountMatrix(index_t n_rows, index_t n_cols,
                     Matrix<Precision, false> *m) {

  m->Init(n_rows, n_cols);

  for (index_t c = 0; c < n_cols; c++) {
    for (index_t r = 0; r < n_rows; r++) {
      m->set(r, c, r + c);
    }
  }
}

template<typename Precision>
void MakeConstantMatrix(index_t n_rows, index_t n_cols, Precision v, Matrix<Precision, false> *m) {
  m->Init(n_rows, n_cols);

  for (index_t c = 0; c < n_cols; c++) {
    for (index_t r = 0; r < n_rows; r++) {
      m->set(r, c, v);
    }
  }
}

template<typename Precision>
void TestMatrixSerialization() {
  NOTIFY("TestMatrixSerialization: Starting");
  Matrix<Precision, false> m1;
  MakeCountMatrix<Precision>(3, 4, &m1);
  {
    std::ofstream ofs("temp");
    boost::archive::text_oarchive oa(ofs);
    oa << m1;
  }
  Matrix<Precision> m2;
  {
    std::ifstream ifs("temp");
    boost::archive::text_iarchive ia(ifs);
    ia >> m2;
  }
}

/** Tests level 1 BLAS-ish stuff. */
template<typename Precision>
void TestMatrixSimpleMath() {

  NOTIFY("TestMatrixSimpleMath: Starting");

  Matrix<Precision, false> m1;
  Matrix<Precision, false> m2;
  Matrix<Precision, false> m3;
  Matrix<Precision, false> m4;
  Matrix<Precision, false> m5;

  MakeCountMatrix<Precision>(3, 4, &m1);
  MakeCountMatrix<Precision>(3, 4, &m2);
  fl::dense::ops::AddTo<Precision>(m2, &m1);
  fl::dense::ops::Add<fl::la::Init>(m1, m2, &m3);
  BOOST_CHECK(m3.get(0, 0) == 0);
  BOOST_CHECK(m3.get(2, 3) == (2 + 3)*3);
  fl::dense::ops::AddExpert<Precision>(-1.0, m2, &m1);
  BOOST_CHECK(m1.get(0, 0) == 0);
  BOOST_CHECK(m1.get(2, 3) == (2 + 3));
  fl::dense::ops::ScaleExpert(Precision(4.0), &m1);
  BOOST_CHECK(m1.get(2, 3) == (2 + 3)*4);
  MakeConstantMatrix<Precision>(3, 4, 7.0, &m4);
  fl::dense::ops::Add<fl::la::Init>(m1, m4, &m5);
  BOOST_CHECK(m5.get(2, 3) == (2 + 3)*4 + 7.0);
  BOOST_CHECK(m5.get(1, 3) == (1 + 3)*4 + 7.0);
  BOOST_CHECK(m5.get(1, 0) == (1 + 0)*4 + 7.0);

  NOTIFY("TestMatrixSimpleMath: Finished");
}

template<typename Precision>
void MakeCountVector(index_t n, Matrix<Precision, true> *v) {
  v->Init(n);

  for (index_t c = 0; c < n; c++) {
    (*v)[c] = c;
  }
}

template<typename Precision>
void MakeConstantVector(index_t n, Precision d, Matrix<Precision, true> *v) {
  v->Init(n);

  for (index_t c = 0; c < n; c++) {
    (*v)[c] = d;
  }
}

/** Tests level 1 BLAS-ish stuff. */
template<typename Precision>
void TestVectorSimpleMath() {

  NOTIFY("TestVectorSimpleMath: Starting");

  Matrix<Precision, true> v1;
  Matrix<Precision, true> v2;
  Matrix<Precision, true> v3;
  Matrix<Precision, true> v4;
  Matrix<Precision, true> v5;

  MakeCountVector<Precision>(6, &v1);
  MakeCountVector<Precision>(6, &v2);
  fl::dense::ops::AddTo<Precision>(v2, &v1);
  fl::dense::ops::Add<fl::la::Init>(v1, v2, &v3);
  BOOST_CHECK(v3[0] == 0);
  BOOST_CHECK(v3[5] == (5)*3);
  fl::dense::ops::AddExpert<Precision>(-1.0, v2, &v1);
  BOOST_CHECK(v1[0] == 0);
  BOOST_CHECK(v1[5] == (5));
  fl::dense::ops::ScaleExpert(Precision(4.0), &v1);
  BOOST_CHECK(v1[5] == (5)*4);
  MakeConstantVector<Precision>(6, 7.0, &v4);
  fl::dense::ops::Add<fl::la::Init>(v1, v4, &v5);
  BOOST_CHECK(v5[5] == (5)*4 + 7.0);
  BOOST_CHECK(v5[4] == (4)*4 + 7.0);
  BOOST_CHECK(v5[1] == (1)*4 + 7.0);

  NOTIFY("TestVectorSimpleMath: Finished");
}

/** Tests aliases and copies */
template<typename Precision>
void TestVector() {

  NOTIFY("TestVector: Starting");

  Matrix<Precision, true> v1;
  const Matrix<Precision, true> *v_const;
  Matrix<Precision, true> v2;
  Matrix<Precision, true> v3;
  Matrix<Precision, true> v4;
  Matrix<Precision, true> v6;
  Matrix<Precision, true> v7;
  Matrix<Precision, true> v8;
  Matrix<Precision, true> v9;

  MakeCountVector<Precision>(10, &v1);
  BOOST_CHECK(v1.length() == 10);
  BOOST_CHECK(v1.ptr()[3] == v1[3]);
  v_const = &v1;
  BOOST_CHECK(v_const->ptr()[3] == (*v_const)[3]);


  v2.Alias(v1);
  BOOST_CHECK(v2[9] == 9);
  BOOST_CHECK(v1.ptr() == v2.ptr());
  v2.MakeColumnSubvector(0, 2, 5, &v3);
  BOOST_CHECK(v3.length() == 5);
  BOOST_CHECK(v3[4] == 6);
  BOOST_CHECK(v3.ptr() != v2.ptr());
  v4.Copy(v3);
  BOOST_CHECK(v4.length() == 5);
  BOOST_CHECK(v4[4] == 6);
  SmallVector<Precision, 21> v5;
  v5.SetZero();
  BOOST_CHECK(v5[20] == 0.0);
  v6.Alias(v1.ptr(), v1.length());
  BOOST_CHECK(v6[9] == 9);
  BOOST_CHECK(v6[3] == 3);
  v7.Own(&v1);
  BOOST_CHECK(v7[9] == 9);
  BOOST_CHECK(v7[3] == 3);
  v8.WeakCopy(v1);
  BOOST_CHECK(v8[9] == 9);
  BOOST_CHECK(v8[3] == 3);
  MakeConstantVector<Precision>(10, 3.5, &v9);
  BOOST_CHECK(v9[0] == 3.5);
  BOOST_CHECK(v1[0] == 0.0);
  v9.SwapValues(&v1);
  BOOST_CHECK_EQUAL(v1[0], 3.5);
  BOOST_CHECK(v9[0] == 0.0);
  BOOST_CHECK(v2[0] == 3.5);
  BOOST_CHECK(v3[0] == 3.5);
  BOOST_CHECK(v4[0] != 3.5);
  BOOST_CHECK(v6[0] == 3.5);
  BOOST_CHECK(v7[0] == 3.5);
  BOOST_CHECK(v8[0] == 3.5);
  v8.SetZero();
  BOOST_CHECK(v1[0] == 0.0);

  NOTIFY("TestVector: Finished");
}

template<typename Precision>
void TestMatrix() {

  NOTIFY("TestMatrix: Starting");

  Matrix<Precision, false> m1;
  const Matrix<Precision, false> *m_const;
  Matrix<Precision, false> m2;
  Matrix<Precision, false> m3;
  Matrix<Precision, false> m4;
  Matrix<Precision, false> m6;
  Matrix<Precision, false> m7;
  Matrix<Precision, false> m8;
  Matrix<Precision, false> m9;

  MakeCountMatrix<Precision>(13, 10, &m1);
  BOOST_CHECK(m1.n_cols() == 10);
  BOOST_CHECK(m1.n_rows() == 13);
  BOOST_CHECK(m1.ptr()[3] == m1.get(3, 0));
  m_const = &m1;
  BOOST_CHECK(m_const->ptr()[3] == (*m_const).get(3, 0));

  Matrix<Precision, true> v1, v2;
  m1.MakeColumnVector(0, &v1);
  m1.MakeColumnVector(1, &v2);
  BOOST_CHECK(v1[12] == 12);
  BOOST_CHECK(v2[12] == 13);

  m2.Alias(m1);
  BOOST_CHECK(m2.get(9, 0) == 9);
  BOOST_CHECK(m1.ptr() == m2.ptr());
  m2.MakeColumnSlice(2, 5, &m3);
  BOOST_CHECK(m3.n_cols() == 5);
  BOOST_CHECK(m3.get(4, 0) == 6);
  BOOST_CHECK(m3.ptr() != m2.ptr());
  m4.Copy(m3);
  BOOST_CHECK(m4.n_cols() == 5);
  BOOST_CHECK(m4.get(4, 0) == 6);
  SmallMatrix<Precision, 21, 21> m5;
  m5.SetZero();
  BOOST_CHECK(m5.get(20, 0) == 0.0);
  m6.Alias(m1.ptr(), m1.n_rows(), m1.n_cols());
  BOOST_CHECK(m6.get(9, 0) == 9);
  BOOST_CHECK(m6.get(3, 0) == 3);
  m7.Own(&m1);
  BOOST_CHECK(m7.get(9, 0) == 9);
  BOOST_CHECK(m7.get(3, 0) == 3);
  m8.WeakCopy(m1);
  BOOST_CHECK(m8.get(9, 0) == 9);
  BOOST_CHECK(m8.get(3, 0) == 3);
  MakeConstantMatrix<Precision>(13, 10, 3.5, &m9);
  BOOST_CHECK(m9.get(0, 0) == 3.5);
  m9.SwapValues(&m1);
  BOOST_CHECK(m9.get(0, 0) == 0.0);
  BOOST_CHECK(m1.get(0, 0) == 3.5);
  BOOST_CHECK(m2.get(0, 0) == 3.5);
  BOOST_CHECK(m3.get(0, 0) == 3.5);
  BOOST_CHECK(m4.get(0, 0) != 3.5);
  BOOST_CHECK(m6.get(0, 0) == 3.5);
  BOOST_CHECK(m7.get(0, 0) == 3.5);
  BOOST_CHECK(m8.get(0, 0) == 3.5);
  m8.SetZero();
  BOOST_CHECK(m1.get(0, 0) == 0.0);

  m8.ref(3, 4) = 21.75;
  BOOST_CHECK(m8.get(3, 4) == 21.75);

  NOTIFY("TestMatrix: Finished");
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

  Matrix<Precision, false> product_actual;

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

  Matrix<Precision, false> product_actual_transa;
  fl::dense::ops::Mul<fl::la::Init, fl::la::Trans, fl::la::NoTrans>(a, b, &product_actual_transa);
  AssertExactMatrix<Precision>(product_expect_transa, product_actual_transa);

  Matrix<Precision, false> a_t, b_t;
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

  Matrix<Precision, true> a_v1_actual;
  fl::dense::ops::Mul<fl::la::Init>(a, v1, &a_v1_actual);
  AssertApproxVector<Precision>(a_v1, a_v1_actual, 0);

  Matrix<Precision, true> a_v2_actual;
  a_v2_actual.Init(3);
  fl::dense::ops::Mul<fl::la::Overwrite, fl::la::NoTrans>(a, v2, &a_v2_actual);
  AssertApproxVector<Precision>(a_v2, a_v2_actual, 0);

  Matrix<Precision, true> v1_a_actual;
  fl::dense::ops::Mul<fl::la::Init, fl::la::Trans>(a, v1,  &v1_a_actual);
  AssertApproxVector<Precision>(v1_a, v1_a_actual, 0);

  Matrix<Precision, true> v2_a_actual;
  v2_a_actual.Init(3);
  fl::dense::ops::Mul<fl::la::Overwrite, fl::la::Trans>(a, v2, &v2_a_actual);
  AssertApproxVector<Precision>(v2_a, v2_a_actual, 0);

  // Test non-square matrices (we had some bad debug checks)
  MAKE_VECTOR(Precision, v3, 4,     1, 2, 3, 4);
  MAKE_VECTOR(Precision, b_v3, 3,   62, 41, 59);
  MAKE_VECTOR(Precision, v1_b, 4,   48, 106, 35, 88);

  SmallVector<Precision, 3> b_v3_actual;

  fl::dense::ops::Mul<fl::la::Overwrite, fl::la::NoTrans>(b, v3, &b_v3_actual);
  AssertApproxVector<Precision>(b_v3, b_v3_actual, 0);

  Matrix<Precision, true> v1_b_actual;
  fl::dense::ops::Mul<fl::la::Init, fl::la::Trans>(b, v1, &v1_b_actual);
  AssertApproxVector<Precision>(v1_b, v1_b_actual, 0);

  NOTIFY("TestMultiplfy: Finished");
}

template<typename Precision>
void TestInverse() {

  NOTIFY("TestInverse: Starting");

  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
                    .5, 0, 0,
                    0, 1, 0,
                    0, 0, 2);
  MAKE_MATRIX_TRANS(Precision, a_inv_expect, 3, 3,
                    2, 0, 0,
                    0, 1, 0,
                    0, 0, .5);

  MAKE_MATRIX_TRANS(Precision, b, 3, 3,
                    3, 1, 4,
                    1, 5, 9,
                    2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, b_inv_expect, 3, 3,
                    0.3222222, -0.2111111, 0.1222222,
                    -0.1444444, -0.0777778, 0.2555556,
                    0.0444444, 0.1777778, -0.1555556);
  MAKE_MATRIX_TRANS(Precision, c, 3, 3,
                    1, 0, 0,
                    0, 1, 0,
                    0, 0, 0);

  Matrix<Precision, false> a_inv_actual;
  success_t temp;
  fl::dense::ops::Inverse<fl::la::Init>(a, &a_inv_actual, &temp);
  BOOST_CHECK(PASSED(temp));
  AssertExactMatrix<Precision>(a_inv_expect, a_inv_actual);

  Matrix<Precision, false> b_inv_actual;

  b_inv_actual.Init(3, 3);
  fl::dense::ops::Inverse<fl::la::Overwrite>(b, &b_inv_actual, &temp);
  BOOST_CHECK(PASSED(temp));
  AssertApproxMatrix<Precision>(b_inv_expect, b_inv_actual, 1.0e-5);

  Matrix<Precision, false> c_inv_actual;
  // Try inverting a 3x3 rank-3 matrix
  fl::dense::ops::Inverse<fl::la::Init>(c, &c_inv_actual, &temp);
  BOOST_CHECK(!PASSED(temp));

  // Try inverting a 3x3 rank-3 matrix
  fl::dense::ops::InverseExpert(&b, &temp);
  BOOST_CHECK(PASSED(temp));
  AssertApproxMatrix<Precision>(b, b_inv_actual, 1.0e-5);

  NOTIFY("TestInverse: Finished");
}

template<typename Precision>
void TestDeterminant() {

  NOTIFY("TestDeterminant: Starting");

  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
                    3, 1, 4,
                    1, 5, 9,
                    2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, b, 3, 3,
                    -3, 5, -8,
                    9, -7, 9,
                    2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, c, 3, 3,
                    -3, -5, -8,
                    9, -7, 9,
                    -2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, d, 3, 3,
                    31, 41, 59,
                    26, 53, 58,
                    97, 93, 23);

  int sign;

  BOOST_CHECK_CLOSE((Precision) - 90.0, (Precision)fl::dense::ops::Determinant<Precision>(a), (Precision)1.0e-4);
  BOOST_CHECK_CLOSE((Precision)log(90.0), (Precision)fl::dense::ops::DeterminantLog<Precision>(a, &sign), (Precision)1.0e-4);
  DEBUG_ASSERT_MSG(sign == -1, "%d", sign);
  BOOST_CHECK_CLOSE((Precision) - 412.0, (Precision)fl::dense::ops::Determinant<Precision>(b), (Precision)1.0e-4);
  BOOST_CHECK_CLOSE((Precision)262.0, (Precision)fl::dense::ops::Determinant<Precision>(c), (Precision)1.0e-4);
  BOOST_CHECK_CLOSE((Precision)log(262.0), (Precision)fl::dense::ops::DeterminantLog<Precision>(c, &sign), (Precision)1.0e-4);
  DEBUG_ASSERT_MSG(sign == 1, "%d", sign);
  BOOST_CHECK_CLOSE((Precision) - 8.3934e4, (Precision)fl::dense::ops::Determinant<Precision>(d), (Precision)2.5e-3);

  NOTIFY("TestDeterminant: Finished");
}

template<typename Precision>
void TestQR() {

  NOTIFY("TestQR: Starting");

  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
                    3, 1, 4,
                    1, 5, 9,
                    2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, a_q_expect, 3, 3,
                    -0.58835, -0.19612, -0.78446,
                    0.71472, -0.57986, -0.39107,
                    -0.37819, -0.79076, 0.48133);
  MAKE_MATRIX_TRANS(Precision, a_r_expect, 3, 3,
                    -5.09902, 0.00000, 0.00000,
                    -8.62911, -5.70425, 0.00000,
                    -6.27572, -4.00511, -3.09426);

  MAKE_MATRIX_TRANS(Precision, b, 3, 4,
                    3, 5, 8,
                    9, 7, 9,
                    3, 2, 3,
                    8, 4, 6);
  MAKE_MATRIX_TRANS(Precision, b_q_expect, 3, 3,
                    -0.303046, -0.505076, -0.808122,
                    0.929360, 0.030979, -0.367872,
                    0.210838, -0.862519, 0.460010);
  MAKE_MATRIX_TRANS(Precision, b_r_expect, 3, 4,
                    -9.89949, 0.00000, 0.00000,
                    -13.53604, 5.27025, 0.00000,
                    -4.34366, 1.74642, 0.28751,
                    -9.29340, 5.35157, 0.99669);

  MAKE_MATRIX_TRANS(Precision, c, 4, 3,
                    3, 9, 3, 8,
                    5, 7, 2, 4,
                    8, 9, 3, 6);
  MAKE_MATRIX_TRANS(Precision, c_q_expect, 4, 3,
                    -0.234978, -0.704934, -0.234978, -0.626608,
                    0.846774, 0.175882, -0.039891, -0.500449,
                    -0.464365, 0.686138, -0.180138, -0.530217);
  //0.110115, 0.036705, -0.954329, 0.275287
  MAKE_MATRIX_TRANS(Precision, c_r_expect, 3, 3,
                    -12.76715, 0.00000, 0.00000,
                    -9.08582, 3.38347, 0.00000,
                    -12.68882, 5.23476, -1.26139);

  Matrix<Precision, false> a_q_actual;
  Matrix<Precision, false> a_r_actual;
  Matrix<Precision, false> a_q_r_actual;
  success_t temp;
  fl::dense::ops::QR<fl::la::Init>(a, &a_q_actual, &a_r_actual, &temp);
  BOOST_CHECK(PASSED(temp));
  fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(a_q_actual, a_r_actual, &a_q_r_actual);
  AssertApproxMatrix<Precision>(a, a_q_r_actual, 1.0e-5);

  Matrix<Precision, false> b_q_actual;
  Matrix<Precision, false> b_r_actual;
  Matrix<Precision, false> b_q_r_actual;
  fl::dense::ops::QR<fl::la::Init>(b, &b_q_actual, &b_r_actual, &temp);
  BOOST_CHECK(PASSED(temp));
  fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(b_q_actual, b_r_actual, &b_q_r_actual);
  AssertApproxMatrix<Precision>(b, b_q_r_actual, 1.0e-5);

  Matrix<Precision, false> c_q_actual;
  Matrix<Precision, false> c_r_actual;
  Matrix<Precision, false> c_q_r_actual;

  fl::dense::ops::QR< fl::la::Init>(c, &c_q_actual, &c_r_actual, &temp);
  BOOST_CHECK(PASSED(temp));
  fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(c_q_actual, c_r_actual, &c_q_r_actual);
  AssertApproxMatrix<Precision>(c, c_q_r_actual, 1.0e-5);

  NOTIFY("TestQR: Finished");
}

template<typename Precision>
void TestEigen() {

  NOTIFY("TestEigen: Starting");

  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
                    3, 1, 4,
                    1, 5, 9,
                    2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, a_eigenvectors_expect, 3, 3,
                    0.212480, 0.912445, 0.172947,
                    0.599107, -0.408996, 0.592976,
                    0.771960, 0.012887, -0.786428);
  MAKE_VECTOR(Precision, a_eigenvalues_real_expect, 3,
              13.08576, 2.58001, -2.66577);
  MAKE_VECTOR(Precision, a_eigenvalues_imag_expect, 3,
              0, 0, 0);

  MAKE_MATRIX_TRANS(Precision, b, 2, 2,
                    3, 4,
                    -2, -1);
  MAKE_MATRIX_TRANS(Precision, b_eigenvectors_real_expect, 2, 2,
                    0.40825, 0.81650,
                    0.40825, 0.81650);
  MAKE_MATRIX_TRANS(Precision, b_eigenvectors_imag_expect, 2, 2,
                    0.40825, 0.0,
                    -0.40825, 0.0);
  MAKE_VECTOR(Precision, b_eigenvalues_real_expect, 2,
              1.0, 1.0);
  MAKE_VECTOR(Precision, b_eigenvalues_imag_expect, 2,
              2.0, -2.0);

  Matrix<Precision, false> a_eigenvectors_actual;
  Matrix<Precision, true> a_eigenvalues_actual;

  success_t temp;
  fl::dense::ops::Eigenvectors< fl::la::Init>(
    a, &a_eigenvalues_actual, &a_eigenvectors_actual, &temp);
  BOOST_CHECK(PASSED(temp));
  AssertApproxVector<Precision>(a_eigenvalues_real_expect, a_eigenvalues_actual, 1.0e-3);
  for (index_t i = 0; i < a_eigenvectors_actual.n_cols(); i++) {
    double  sign = 0;
    if (a_eigenvectors_actual.get(0, i) > 0) {
      sign = 1;
    }
    else {
      sign = -1;
    }
    for (index_t j = 0; j < a_eigenvectors_actual.n_rows(); j++) {
      a_eigenvectors_actual.set(j, i,
                                a_eigenvectors_actual.get(j, i)*sign);
    }
  }
  AssertApproxTransMatrix<Precision>(a_eigenvectors_expect, a_eigenvectors_actual, 1.0e-3);

  Matrix<Precision, true> a_eigenvalues_real_actual;
  Matrix<Precision, true> a_eigenvalues_imag_actual;
  fl::dense::ops::Eigenvalues<fl::la::Init>(
    a, &a_eigenvalues_real_actual, &a_eigenvalues_imag_actual, &temp);
  BOOST_CHECK(PASSED(temp));
  AssertApproxVector<Precision>(a_eigenvalues_real_expect, a_eigenvalues_real_actual, 1.0e-3);
  AssertApproxVector<Precision>(a_eigenvalues_imag_expect, a_eigenvalues_imag_actual, 0.0);

  Matrix<Precision, true> a_eigenvalues_actual_2;
  fl::dense::ops::Eigenvalues<fl::la::Init>(
    a, &a_eigenvalues_actual_2, &temp);
  BOOST_CHECK(PASSED(temp));
  AssertApproxVector<Precision>(a_eigenvalues_real_expect, a_eigenvalues_actual_2, 1.0e-3);

  // complex eigenvalues

  /*
   * This function no longer fails on imaginary, but sets them to NaN
   */
  //Matrix b_eigenvectors_actual;
  //Vector b_eigenvalues_actual;
  //BOOST_CHECK(!PASSED(fl::la::EigenvectorsInit(
  //    b, &b_eigenvalues_actual, &b_eigenvectors_actual)));

  Matrix<Precision, false> b_eigenvectors_real_actual;
  Matrix<Precision, false> b_eigenvectors_imag_actual;
  Matrix<Precision, true> b_eigenvalues_real_actual;
  Matrix<Precision, true> b_eigenvalues_imag_actual;
  fl::dense::ops::Eigenvectors<fl::la::Init>(
    b, &b_eigenvalues_real_actual, &b_eigenvalues_imag_actual,
    &b_eigenvectors_real_actual, &b_eigenvectors_imag_actual, &temp);
  BOOST_CHECK(PASSED(temp));
  AssertApproxVector<Precision>(b_eigenvalues_real_expect, b_eigenvalues_real_actual, 1.0e-3);
  AssertApproxMatrix<Precision>(b_eigenvectors_real_expect, b_eigenvectors_real_actual, 1.0e-3);
  AssertApproxVector<Precision>(b_eigenvalues_imag_expect, b_eigenvalues_imag_actual, 1.0e-3);
  AssertApproxMatrix<Precision>(b_eigenvectors_imag_expect, b_eigenvectors_imag_actual, 1.0e-3);

  NOTIFY("TestEigen: Finished");
}

template<typename Precision>
void TrySchur(const Matrix<Precision, false> &orig) {
  Matrix<Precision, false> z;
  Matrix<Precision, false> t;
  Matrix<Precision, true> eigen_real;
  Matrix<Precision, true> eigen_imag;
  success_t success;
  fl::dense::ops::Schur<fl::la::Init>(orig, &eigen_real, &eigen_imag, &t, &z, &success);

  Matrix<Precision, false> z_trans;
  fl::dense::ops::Transpose<fl::la::Init>(z, &z_trans);
  Matrix<Precision, false> tmp;
  fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(t, z_trans, &tmp);
  Matrix<Precision, false> result;
  fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(z, tmp, &result);

  AssertApproxMatrix<Precision>(orig, result, 1.0e-4);

  /*
   * This test now fails because Schur finds real components while
   * Eigenvectors on 3 args only finds true real eigenvalues
   */
  //Vector eigen_real_2;
  //Matrix eigenvectors_2;
  //fl::dense::ops::EigenvectorsInit(orig, &eigen_real_2, &eigenvectors_2);
  //AssertApproxVector(eigen_real_2, eigen_real, 1.0e-8);
}

template<typename Precision>
void TestSchur() {

  NOTIFY("TestSchur: Starting");

  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
                    3, 1, 4,
                    1, 5, 9,
                    2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, b, 5, 5,
                    3, 1, 4, 1, 5,
                    9, 2, 6, 5, 3,
                    5, 8, 9, 7, 9,
                    3, 2, 3, 8, 4,
                    6, 2, 6, 4, 3);

  TrySchur<Precision>(a);
  TrySchur<Precision>(b);

  NOTIFY("TestSchur: Finished");
}

template<typename Precision>
void AssertProperSVD(const Matrix<Precision, false>& orig,
                     const Matrix<Precision, true> &s, const Matrix<Precision, false>& u,
                     const Matrix<Precision, false>& vt) {
  Matrix<Precision, false> s_matrix;
  s_matrix.Init(s.length(), s.length());
  s_matrix.SetDiagonal(s);
  Matrix<Precision, false> tmp;
  fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(u, s_matrix, &tmp);
  Matrix<Precision, false> result;
  fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(tmp, vt, &result);
  AssertApproxMatrix<Precision>(result, orig, 1.0e-4);
}

template<typename Precision>
void TrySVD(const Matrix<Precision, false>& orig) {
  Matrix<Precision, true> s;
  Matrix<Precision, false> u;
  Matrix<Precision, false> vt;
  success_t success;
  fl::dense::ops::SVD<fl::la::Init>(orig, &s, &u, &vt, &success);
  AssertProperSVD<Precision>(orig, s, u, vt);
}

template<typename Precision>
void TestSVD() {

  NOTIFY("TestSVD: Starting");

  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
                    3, 1, 4,
                    1, 5, 9,
                    2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, a_u_expect, 3, 3,
                    -0.21141,  -0.55393,  -0.80528,
                    0.46332,  -0.78225,   0.41645,
                    -0.86060,  -0.28506,   0.42202);
  MAKE_VECTOR(Precision, a_s_expect, 3,
              13.58236, 2.84548, 2.32869);
  MAKE_MATRIX_TRANS(Precision, a_vt_expect, 3, 3,
                    -0.32463,   0.79898,  -0.50620,
                    -0.75307,   0.10547,   0.64943,
                    -0.57227,  -0.59203,  -0.56746);
  MAKE_MATRIX_TRANS(Precision, b, 3, 10,
                    3, 1, 4,
                    1, 5, 9,
                    2, 6, 5,
                    3, 5, 8,
                    9, 7, 9,
                    3, 2, 3,
                    8, 4, 6,
                    2, 6, 4,
                    3, 3, 8,
                    3, 2, 7);
  MAKE_MATRIX_TRANS(Precision, c, 9, 3,
                    3, 1, 4, 1, 5, 9, 2, 6, 5,
                    3, 5, 8, 9, 7, 9, 3, 2, 3,
                    8, 4, 6, 2, 6, 4, 3, 3, 8);
  MAKE_MATRIX_TRANS(Precision, d, 3, 3,
                    0, 1, 0,
                    -1, 0, 0,
                    0, 0, 1);

  Matrix<Precision, false> a_u_actual;
  Matrix<Precision, true> a_s_actual;
  Matrix<Precision, false> a_vt_actual;

  success_t temp;
  fl::dense::ops::SVD<fl::la::Init>(a, &a_s_actual, &a_u_actual, &a_vt_actual, &temp);
  AssertProperSVD<Precision>(a, a_s_actual, a_u_actual, a_vt_actual);
  AssertApproxVector<Precision>(a_s_expect, a_s_actual, 1.0e-3);
  AssertApproxMatrix<Precision>(a_u_expect, a_u_actual, 1.0e-3);
  AssertApproxMatrix<Precision>(a_vt_expect, a_vt_actual, 1.0e-3);

  Matrix<Precision, true> a_s_actual_2;
  success_t success;
  fl::dense::ops::SVD<fl::la::Init>(a, &a_s_actual_2, &success);
  AssertApproxVector<Precision>(a_s_expect, a_s_actual_2, 1.0e-3);

  TrySVD<Precision>(b);
  TrySVD<Precision>(c);
  TrySVD<Precision>(d);

  // let's try a big, but asymmetric, one
  Matrix<Precision, false> e;
  e.Init(3000, 10);
  for (index_t j = 0; j < e.n_cols(); j++) {
    for (index_t i = 0; i < e.n_rows(); i++) {
      e.set(i, j, rand() * 1.0 / RAND_MAX);
    }
  }

  TrySVD<Precision>(e);

  NOTIFY("TestSVD: Finished");
}

template<typename Precision>
void TryCholesky(const Matrix<Precision, false> &orig) {
  Matrix<Precision, false> u;
  success_t temp;
  fl::dense::ops::Cholesky<fl::la::Init>(orig, &u, &temp);
  BOOST_CHECK(PASSED(temp));
  Matrix<Precision, false> result;
  fl::dense::ops::Mul<fl::la::Init, fl::la::Trans, fl::la::NoTrans>(u, u, &result);
  AssertApproxMatrix<Precision>(orig, result, 1.0e-3);
}

template<typename Precision>
void TestCholesky() {

  NOTIFY("TestCholesky: Starting");

  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
                    1, 0, 0,
                    0, 2, 0,
                    0, 0, 3);
  MAKE_MATRIX_TRANS(Precision, b, 4, 4,
                    9.00,   0.60,  -0.30,   1.50,
                    0.60,  16.04,   1.18,  -1.50,
                    -0.30,   1.18,   4.10,  -0.57,
                    1.50,  -1.50,  -0.57,  25.45);
  TryCholesky<Precision>(a);
  TryCholesky<Precision>(b);

  NOTIFY("TestCholesky: Finished");
}

template<typename Precision>
void TrySolveMatrix(const Matrix<Precision, false>& a, const Matrix<Precision, false>& b) {
  Matrix<Precision, false> x;
  success_t temp;
  fl::dense::ops::Solve<fl::la::Init>(a, b, &x, &temp);
  BOOST_CHECK(PASSED(temp));
  Matrix<Precision, false> result;
  fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(a, x, &result);
  AssertApproxMatrix<Precision>(b, result, 1.0e-3);
}

template<typename Precision>
void TrySolveVector(const Matrix<Precision, false>& a,
                    const Matrix<Precision, true>& b) {
  Matrix<Precision, true> x;
  success_t success;
  fl::dense::ops::Solve<fl::la::Init>(a, b, &x, &success);
  Matrix<Precision, true> result;
  fl::dense::ops::Mul<fl::la::Init, fl::la::NoTrans>(a, x, &result);
  AssertApproxVector<Precision>(b, result, 1.0e-3);
}

template<typename Precision>
void TestSolve() {

  NOTIFY("TestSolve: Starting");

  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
                    3, 1, 4,
                    1, 5, 9,
                    2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, a_vectors, 3, 5,
                    1, 2, 3,
                    4, 5, 2,
                    1, 6, 3,
                    2, 1, 8,
                    4, 2, 6);
  MAKE_VECTOR(Precision, a_vector_1, 3,   3, 1, 2);
  MAKE_VECTOR(Precision, a_vector_2, 3,   2, 4, 6);
  MAKE_VECTOR(Precision, a_vector_3, 3,   2, 4, 6);
  MAKE_VECTOR(Precision, a_vector_4, 3,   5, 7, 8);
  MAKE_MATRIX_TRANS(Precision, b, 5, 5,
                    3, 1, 4, 1, 5,
                    9, 2, 6, 5, 3,
                    5, 8, 9, 7, 9,
                    3, 2, 3, 8, 4,
                    6, 2, 6, 4, 3);

  TrySolveMatrix<Precision>(a, a_vectors);
  TrySolveVector<Precision>(a, a_vector_1);
  TrySolveVector<Precision>(a, a_vector_2);
  TrySolveVector<Precision>(a, a_vector_3);
  TrySolveVector<Precision>(a, a_vector_4);

  NOTIFY("TestSolve: Finished");
}

/**
 * Writen by Nick to Test LeastSquareFit
 */
template<typename Precision>
void TestLeastSquareFit() {

  NOTIFY("TestLeastSquareFit: Starting");

  Matrix<Precision, false> x;
  Matrix<Precision, false> y;
  Matrix<Precision, false> a;
  x.Init(3, 2);
  x.set(0, 0, 1.0);
  x.set(0, 1, -1.0);
  x.set(1, 0, 0.33);
  x.set(1, 1, 0.44);
  x.set(2, 0, 1.5);
  x.set(2, 1, -0.2);
  y.Init(3, 2);
  y.set(0, 0, 1.5);
  y.set(0, 1, -2.0);
  y.set(1, 0, -0.3);
  y.set(1, 1, 4.0);
  y.set(2, 0, 0.2);
  y.set(2, 1, -0.4);
  success_t success;
  fl::dense::ops::LeastSquareFit<fl::la::Init>(y, x, &a, &success);
  Matrix<Precision, false> true_a;
  true_a.Init(2, 2);
  true_a.set(0, 0, 0.0596);
  true_a.set(0, 1, 1.0162);
  true_a.set(1, 0, -1.299);
  true_a.set(1, 1, 4.064);
  for (index_t i = 0; i < 2; i++) {
    for (index_t j = 0; j < 2; j++) {
      BOOST_CHECK_CLOSE(true_a.get(i, j), a.get(i, j), (Precision)0.1);
    }
  }

  NOTIFY("TestLeastSquareFit: Finished");
}

template<typename Precision>
void TestApplyFunc(bool print_debug_info) {

  NOTIFY("TestApplyFunc: Starting");

  Precision precision = 1.0e-3;
  for (index_t i = 0; i < 10; i++) {

    // Random matrix.
    Matrix<Precision, false> x;
    Matrix<Precision, false> y, sine_answer, cosine_answer, tangent_answer,
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
    Matrix<Precision, false> x;
    Matrix<Precision, false> y;
    Matrix<Precision, false> z;
    Matrix<Precision, false> a;
    Matrix<Precision, false> a_pow;

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
