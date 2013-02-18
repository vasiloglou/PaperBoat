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
 * @file matrix.h
 *
 * Basic double-precision vector and matrix classes.
 */

#ifndef FASTLIB_DENSE_MATRIX_H
#define FASTLIB_DENSE_MATRIX_H

#include <new>
#include <fstream>
#include "boost/scoped_ptr.hpp"
#include "boost/scoped_array.hpp"
#include "boost/serialization/split_member.hpp"
#include "boost/serialization/nvp.hpp"
#include "boost/serialization/base_object.hpp"
#include "boost/mpl/void.hpp"
#include "fastlib/base/base.h"
#include <vector>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fastlib/math/fl_math.h"
#include "fastlib/traits/fl_traits.h"
#include "linear_algebra_aux.h"
#include "linear_algebra.h"
#include "boost/archive/archive_exception.hpp"

namespace fl {
namespace  dense {
template<typename Precision, bool IsVector>
class Matrix;

}
}

/**
 * @brief forward declaration of MonolithicPoint
 */
namespace fl {
namespace data {
/* Forward Declarations */
template<typename CalcPrecisionType>
class MonolithicPoint;
}
}

namespace fl {
namespace dense {
/**
 * @brief Static assertion for the matrix class
 *        we use them to make sure people
 *        do not accidently give wrong  template arguments
 *
 */

/**
 * @brief Because we want to be backwards compliant the Matrix
 *        can also be used as an old vector. We use the following
 *        assertion so that people don't accidently call matrix
 *        member functions that cannot be used for vectors.
 *  @code
 *   template<bool> struct You_are_trying_to_use_a_vector_as_a_matrix;
     template<> struct You_are_trying_to_use_a_vector_as_a_matrix<false> {};
 *  @endcode
 */
template<bool> struct You_are_trying_to_use_a_vector_as_a_matrix;
template<> struct You_are_trying_to_use_a_vector_as_a_matrix<false> {};



/**
 * @brief  We use this static assertion so that we don't accidently
 *         try to copy a matrix on a vector
 *
 */
template<bool, bool> struct You_are_assigning_a_matrix_on_a_vector {};
template<> struct You_are_assigning_a_matrix_on_a_vector<true, false>;


/**
   *  @brief this is a helper trait to make sure that the PrintDebug
   *         function works correctly
   */
template<typename Precision, bool IsVector>
class PrintTrait {
  public:
    static void Print(const Matrix<Precision, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr);
};

template<bool IsVector>
class PrintTrait<float, IsVector> {
  public:
    static void Print(const Matrix<float, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr)  {
      fprintf(stream, "----- MATRIX %s ------\n", name);
      for (index_t r = 0; r < mat.n_rows(); r++) {
        for (index_t c = 0; c < mat.n_cols(); c++) {
          fprintf(stream, "%+3.3f ", mat.get(r, c));
        }
        fprintf(stream, "\n");
      }
    }
};

template<bool IsVector>
class PrintTrait<double, IsVector> {
  public:
    static void Print(const Matrix<double, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr)  {
      fprintf(stream, "----- MATRIX %s ------\n", name);
      for (index_t r = 0; r < mat.n_rows(); r++) {
        for (index_t c = 0; c < mat.n_cols(); c++) {
          fprintf(stream, "%+3.3f ", mat.get(r, c));
        }
        fprintf(stream, "\n");
      }
    }
};

template<bool IsVector>
class PrintTrait<long double, IsVector> {
  public:
    static void Print(const Matrix<long double, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr)  {
      fprintf(stream, "----- MATRIX %s ------\n", name);
      for (index_t r = 0; r < mat.n_rows(); r++) {
        for (index_t c = 0; c < mat.n_cols(); c++) {
          fprintf(stream, "%+3.3f ", mat.get(r, c));
        }
        fprintf(stream, "\n");
      }
    }
};

template<bool IsVector>
class PrintTrait<int, IsVector> {
  public:
    static void Print(const Matrix<int, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr)  {
      fprintf(stream, "----- MATRIX %s ------\n", name);
      for (index_t r = 0; r < mat.n_rows(); r++) {
        for (index_t c = 0; c < mat.n_cols(); c++) {
          fprintf(stream, "%+3.3i ", mat.get(r, c));
        }
        fprintf(stream, "\n");
      }
    }
};

template<bool IsVector>
class PrintTrait<long int, IsVector> {
  public:
    static void Print(const Matrix<long int, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr) {
      fprintf(stream, "----- MATRIX %s ------\n", name);
      for (index_t r = 0; r < mat.n_rows(); r++) {
        for (index_t c = 0; c < mat.n_cols(); c++) {
          fprintf(stream, "%+3.3i ", mat.get(r, c));
        }
        fprintf(stream, "\n");
      }
    }
};

template<bool IsVector>
class PrintTrait<long long int, IsVector> {
  public:
    static void Print(const Matrix<long long int, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr) {
      fprintf(stream, "----- MATRIX %s ------\n", name);
      for (index_t r = 0; r < mat.n_rows(); r++) {
        for (index_t c = 0; c < mat.n_cols(); c++) {
          fprintf(stream, "%+3.3i ", mat.get(r, c));
        }
        fprintf(stream, "\n");
      }
    }
};

/**
 * @brief General-Precision column-major matrix for use with LAPACK.
 * This class is backwords compatible, so that it can represent
 * the old fastlib GenVector.
 *
 * @code
 *  template<typename Precision, bool IsVector=false>
 *   class Matrix;
 * @endcode
 *
 * @param Precision In general you can choose any precision you want, but
 *        our current LAPACK implementation suppors floats and doubles.
 *        It is not difficult thought to support complex numbers. You can
 *        use integers too,  but LAPACK will not work
 *
 * Your code can have huge performance hits if you fail to realize this
 * is column major.  For datasets, your columns should be individual points
 * and your rows should be features.
 *
 * TODO: If it's not entirely obvious or well documented how to use this
 * class please let the FASTlib people know.
 */
template < typename Precision, bool IsVector = false >
class Matrix : virtual public fl::dense::ops {
    friend class ops;
    friend class boost::serialization::access;
  protected:
    /** Linearized matrix (column-major). */
    Precision *ptr_;
    /** Number of rows. */
    index_t n_rows_;
    /** Number of columns. */
    index_t n_cols_;
    /** Number of elements for faster access. */
    index_t n_elements_;
    /**
     * Number of elements that fit in allocated memory.
     * Set to zero if aliasing.
     */
    index_t capacity_;

  public:
    class iterator {
      public:
        iterator() {
          point_.reset(new fl::data::MonolithicPoint<Precision>());
          origin_ = NULL;
          column_ = 0;
        }

        iterator(const Matrix<Precision, false> *origin, index_t column) {
          point_.reset(new fl::data::MonolithicPoint<Precision>());
          origin_ = origin;
          column_ = column;
        }

        iterator(const iterator &other) {
          point_.reset(new fl::data::MonolithicPoint<Precision>());
          origin_ = other.origin_;
          column_ = other.column_;
        }
        iterator &operator=(const iterator &other) {
          origin_ = other.origin_;
          column_ = other.column_;
          return *this;
        }

        iterator &operator++() {
          column_++;
          return *this;
        }
        iterator operator++(int) {
          iterator temp(*this);
          column_++;
          return temp;
        }

        iterator &operator+=(ptrdiff_t offset) {
          column_ += offset;
          return *this;
        }
        iterator operator+(ptrdiff_t offset) const {
          iterator temp(*this);
          temp += offset;
          return temp;
        }

        iterator &operator--() {
          column_--;
          return *this;
        }
        iterator operator--(int) {
          iterator temp(*this);
          column_--;
          return temp;
        }

        iterator &operator-=(ptrdiff_t offset) {
          column_ -= offset;
          return *this;
        }
        iterator operator-(ptrdiff_t offset) const {
          iterator temp(*this);
          temp -= offset;
          return temp;
        }

        fl::data::MonolithicPoint<Precision> *operator->() {
          origin_->MakeColumnVector(column_, point_.get());
          return point_.get();
        }
        fl::data::MonolithicPoint<Precision> &operator*() {
          origin_->MakeColumnVector(column_, point_.get());
          return *point_;
        }

        bool operator!=(const iterator &other) const {
          return !(*this == other);
        }
        bool operator==(const iterator &other) const {
          return column_ == other.column_ && origin_ == other.origin_;
        }

      private:
        boost::scoped_ptr<fl::data::MonolithicPoint<Precision> > point_;
        const Matrix<Precision, false> *origin_;
        index_t column_;
    };
    typedef Precision Precision_t;
    typedef Precision CalcPrecision_t;
    static const bool IsVector_t = IsVector;
    /**
     * Creates a Matrix with uninitialized elements of the specified size.
     */
    Matrix(index_t in_rows) {
      Uninitialize_();
      Init(in_rows);
    }
    Matrix(index_t in_rows, index_t in_cols) {
      if (IsVector == true) {
        if (in_cols != 1) {
          fl::logger->Die()<<"You are trying to initialize a vector with more than one column";
        }
      }
      Uninitialize_();
      Init(in_rows, in_cols);
    }

    /**
     * Copy constructor -- for use in collections.
     */
    Matrix(const Matrix<Precision, IsVector>& other) {
      Uninitialize_();
      Copy(other);
    }

    /**
     * Creates a matrix that can be initialized.
     */
    Matrix() {
      Uninitialize_();
    }

    /**
     * Empty destructor.
     */
    virtual ~Matrix() {
      Destruct();
    }

    /**
     * Destructs this, so that it is suitable for you to call an initializer
     * on this again.
     */
    void Destruct() {
      // DEBUG_ASSERT_MSG(ptr_ != BIG_BAD_POINTER(Precision), "You forgot to initialize a Matrix before it got automatically freed.");
      if (unlikely(should_free())) {
        delete[] ptr_;
      }
      Uninitialize_();
    }
    iterator begin() const  {
      return iterator(this, 0);
    };
    iterator end() const {
      return iterator(this, n_cols_);
    }

    /**
     * Creates a Matrix with uninitialized elements of the specified size.
     * NOTICE! this should be used only when you are in the matrix mode
     * if you set IsVector template parameter to true, which means you
     * are using it as a vector then you will get a compile time error.
     */
    void Init(index_t in_rows, index_t in_cols) {
      if (IsVector == true && in_cols > 1) {
        fl::logger->Die()<<"You are trying to use a vector as a matrix";
      }
      // DEBUG_ONLY(AssertUninitialized_());
      Destruct();
      n_rows_ = in_rows;
      n_cols_ = in_cols;
      n_elements_ = n_rows_ * n_cols_;
      capacity_ = n_elements_ + (n_elements_ == 0);
      ptr_ = new (std::nothrow) Precision[capacity_];
      if (ptr_==NULL) {
        fl::logger->Die() << "Failed to allocate memory of :" <<
          capacity_ *sizeof(Precision) << " bytes";
      }
    }

    /**
     * You can use it to initialize vectors and matrices too. For matrices
     * it will assume that it is an one column
     */
    void Init(index_t in_rows) {
      // DEBUG_ONLY(AssertUninitialized_());
      Destruct();
      n_rows_ = in_rows;
      n_cols_ = 1;
      n_elements_ = n_rows_;
      capacity_ = n_elements_ + (n_elements_ == 0);
      ptr_ = new (std::nothrow) Precision[capacity_];
      if (ptr_==NULL) {
        fl::logger->Die() << "Failed to allocate memory of :" <<
          capacity_ *sizeof(Precision) << " bytes";
      }
    }

    /**
     * This is for STL vector compliance
     */
    void resize(index_t in_rows, index_t in_cols) {
      You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
      DEBUG_ASSERT(should_free());
      DEBUG_SAME_SIZE(in_rows, n_rows_);

      n_cols_ = in_cols;
      Resize_(n_rows_ * n_cols_);
    }

    /**
     * This is for STL vector compliance
     */
    void resize(index_t in_rows) {
      DEBUG_ASSERT(should_free());
      DEBUG_SAME_SIZE(n_cols_, 1);

      n_rows_ = in_rows;
      Resize_(n_rows_);
    }

    template<typename OtherPrecision>
    void push_back(const Matrix<OtherPrecision, true> &in_vector) {
      You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
      fl::You_have_a_precision_conflict<Precision, OtherPrecision>();
      DEBUG_ASSERT(should_free());
      DEBUG_SAME_SIZE(n_rows_, in_vector.size());

      ++n_cols_;
      Resize_(n_elements_ + n_rows_);

      CopyValues_(ptr_ + n_elements_ - n_rows_, in_vector.ptr(), n_rows_);
    }

    void push_back(Precision in_value) {
      DEBUG_ASSERT(should_free());
      DEBUG_SAME_SIZE(n_cols_, 1);

      ++n_rows_;
      Resize_(n_elements_ + 1);

      ptr_[n_elements_ - 1] = in_value;
    }

    /**
     * Creates a diagonal matrix.
     * NOTICE! this should be used only when you are in the matrix mode
     * if you set IsVector template parameter to true, which means you
     * are using it as a vector then you will get a compile time error.
     */
    template<template<typename> class SerialContainer>
    void InitDiagonal(const SerialContainer<Precision>& v) {
      You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
      Init(v.size(), v.size());
      SetZero();
      SetDiagonal(v);
    }

    /**
     * Creates a diagonal matrix.
     * NOTICE! this should be used only when you are in the matrix mode
     * if you set IsVector template parameter to true, which means you
     * are using it as a vector then you will get a compile time error.
     */
    void InitDiagonal(const index_t dimension, const Precision value) {
      You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
      Init(dimension, dimension);
      SetZero();
      for (index_t i = 0; i < dimension; i++) {
        this->set(i, i, value);
      }
    }

    /**
      * Sets the entire matrix to zero.
      */
    void SetAll(Precision d) {
      for (int i = 0; i < n_elements_; ++i) {
        ptr_[i] = d;
      }
    }

    /**
     * Makes this matrix all zeroes.
     */
    void SetZero() {
      // TODO: If IEEE floating point is used, this can just be a memset to
      // zero
      SetAll(0);
    }

    /**
     * Makes this a diagonal matrix whose diagonals are the values in v.
     * NOTICE! this should be used only when you are in the matrix mode
     * if you set IsVector template parameter to true, which means you
     * are using it as a vector then you will get a compile time error.
     */
    template<typename SerialContainer>
    void SetDiagonal(const SerialContainer& v) {
      You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
      DEBUG_SAME_SIZE(std::min(n_rows_, n_cols_), v.length());
      SetZero();
      index_t n = v.length();
      for (index_t i = 0; i < n; i++) {
        set(i, i, v[i]);
      }
    }

    /**
     * Makes this uninitialized matrix a copy of the other Matrix.
     *
     * @param other the vector to explicitly copy
     */
    template<typename OtherPrecision, bool OtherIsVector>
    void Copy(const Matrix<OtherPrecision, OtherIsVector>& other) {
      You_are_assigning_a_matrix_on_a_vector<IsVector, OtherIsVector>();
      fl::You_have_a_precision_conflict<Precision, OtherPrecision>();
      Copy(other.ptr(), other.n_rows(), other.n_cols());
    }

    /**
     * Makes this uninitialized matrix a copy of the other vector.
     *
     * @param ptr_in the pointer to a block of column-major doubles
     * @param n_rows_in the number of rows
     * @param n_cols_in the number of columns
     * NOTICE! this should be used only when you are in the matrix mode
     * if you set IsVector template parameter to true, which means you
     * are using it as a vector then you will get a compile time error.
     */
    template<typename OtherPrecision>
    void Copy(const OtherPrecision *ptr_in, index_t n_rows_in, index_t n_cols_in) {
      fl::You_have_a_precision_conflict<Precision, OtherPrecision>();
      Init(n_rows_in, n_cols_in);

      CopyValues_(ptr_, ptr_in, n_elements_);
    }

    /**
     * Makes this uninitialized Vector a copy of the other vector.
     * If you use this for a matrix then it assumes single column
     * @param ptr_in the pointer to a block of column-major doubles
     * @param n_rows_in the number of rows
    */
    template<typename OtherPrecision>
    void Copy(const OtherPrecision *ptr_in, index_t n_rows_in) {
      fl::You_have_a_precision_conflict<Precision, OtherPrecision>();
      Init(n_rows_in);

      CopyValues_(ptr_, ptr_in, n_elements_);
    }

    /**
     * Makes this uninitialized matrix an alias of another matrix.
     *
     * Changes to one matrix are visible in the other (and vice-versa).
     *
     * @param other the other vector
     */
    template<typename OtherPrecision, bool OtherIsVector>
    void Alias(const Matrix<OtherPrecision, OtherIsVector> & other) {
      fl::You_have_a_precision_conflict<Precision, OtherPrecision>();
      You_are_assigning_a_matrix_on_a_vector<IsVector, OtherIsVector>();
      // we trust in good faith that const-ness won't be abused
      Alias(other.ptr_, other.n_rows(), other.n_cols());
    }

    /**
     * Makes this uninitialized matrix an alias of an existing block of doubles.
     *
     * @param ptr_in the pointer to a block of column-major doubles
     * @param n_rows_in the number of rows
     * @param n_cols_in the number of columns
     * NOTICE! this should be used only when you are in the matrix mode
     * if you set IsVector template parameter to true, which means you
     * are using it as a vector then you will get a compile time error.
     */
    void Alias(Precision *ptr_in, index_t n_rows_in, index_t n_cols_in) {
      if (IsVector == true && n_cols_in > 1) {
        fl::logger->Die()<<"You are trying to use a vector as a matrix";
      }
      // DEBUG_ONLY(AssertUninitialized_());
      Destruct();
      ptr_ = ptr_in;
      n_rows_ = n_rows_in;
      n_cols_ = n_cols_in;
      n_elements_ = n_rows_ * n_cols_;
      capacity_ = 0;
    }

    void Alias(Precision *ptr_in, index_t n_rows_in) {
      // DEBUG_ONLY(AssertUninitialized_());
      Destruct();
      ptr_ = ptr_in;
      n_rows_ = n_rows_in;
      n_cols_ = 1;
      n_elements_ = n_rows_ * n_cols_;
      capacity_ = 0;
    }

    /**
     * Makes this a 1 row by N column alias of a vector of length N.
     *
     * @param row_vector the vector to alias
     */

    /*  void AliasRowVector(const GenVector<T>& row_vector) {
        Alias(const_cast<T*>(row_vector.ptr()), 1, row_vector.length());
      }
    */

    /**
     * Makes this an N row by 1 column alias of a vector of length N.
     *
     * @param col_vector the vector to alias
     */
    /*  void AliasColVector(const GenVector<T>& col_vector) {
        Alias(const_cast<T*>(col_vector.ptr()), col_vector.length(), 1);
      }
    */

    /**
     * Makes this a weak copy or alias of the other.
     *
     * This is identical to Alias.
     */
    template<bool OtherIsVector>
    void WeakCopy(const Matrix<Precision, OtherIsVector>& other) {
      Alias(other);
    }

    /**
     * Makes this uninitialized matrix the "owning copy" of the other
     * matrix; the other vector becomes an alias and this becomes the
     * standard.
     *
     * The other matrix must be the "owning copy" of its memory.
     *
     * @param other a pointer to the other matrix
     */
    template<bool OtherIsVector>
    void Own(Matrix<Precision, OtherIsVector>* other) {
      You_are_assigning_a_matrix_on_a_vector<IsVector, OtherIsVector>();
      DEBUG_ASSERT(other->should_free());
      Own(other->ptr(), other->n_rows(), other->n_cols());
      other->capacity_ = 0;
    }

    /**
     * Initializes this uninitialized matrix as the "owning copy" of
     * some linearized chunk of RAM allocated with new.
     *
     * @param ptr_in the pointer to a block of column-major doubles
     *        allocated via new.
     * @param n_rows_in the number of rows
     * @param n_cols_in the number of columns
     */
    void Own(Precision *ptr_in, index_t n_rows_in, index_t n_cols_in) {
      if (IsVector == true && n_cols_in > 1) {
        fl::logger->Die()<<"You are trying to use a vector as a matrix";
      }
      // DEBUG_ONLY(AssertUninitialized_());
      Destruct();
      ptr_ = ptr_in;
      n_rows_ = n_rows_in;
      n_cols_ = n_cols_in;
      n_elements_ = n_rows_ * n_cols_;
      capacity_ = n_elements_;
      DEBUG_ASSERT(should_free());
    }

    void Own(Precision *ptr_in, index_t n_rows_in) {
      // DEBUG_ONLY(AssertUninitialized_());
      Destruct();
      ptr_ = ptr_in;
      n_rows_ = n_rows_in;
      n_cols_ = 1;
      n_elements_ = n_rows_;
      capacity_ = n_elements_;
      DEBUG_ASSERT(should_free());
    }

    /**
      * Make a matrix that is an alias of a particular slice of my columns.
      *
      * @param start_col the first column
      * @param n_cols_new the number of columns in the new matrix
      * @param dest an UNINITIALIZED matrix
      */
    void MakeColumnSlice(index_t start_col, index_t n_cols_new,
                         Matrix<Precision, IsVector> *dest) const {
      You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
      DEBUG_BOUNDS(start_col, n_cols_);
      DEBUG_BOUNDS(start_col + n_cols_new, n_cols_ + 1);
      dest->Alias(ptr_ + start_col * n_rows_, n_rows_, n_cols_new);
    }

    /**
     * Make an alias of a reshaped version of this matrix (column-major format).
     *
     * For instance, a matrix with 2 rows and 6 columns can be reshaped
     * into a matrix with 12 rows 1 column, 1 row and 12 columns, or a variety
     * of other shapes.  The layout of the new elements correspond exactly
     * to just pretending that the current column-major matrix is laid out
     * as a different column-major matrix.
     *
     * It is required that n_rows_new * n_cols_new is the same as
     * n_rows * n_cols of the original matrix.
     *
     * TODO: Considering using const Matrix& for third-party classes that want
     * to implicitly convert to Matrix.
     *
     * @param n_rows_in new number of rows
     * @param n_cols_in new number of columns
     * @param dest a pointer to an unitialized matrix
     * @return a reshaped matrix backed by the original
     */
    void MakeReshaped(index_t n_rows_in, index_t n_cols_in,
                      Matrix<Precision, IsVector> *dest) const {
      You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
      DEBUG_SAME_SIZE(n_elements_, n_rows_in * n_cols_in);
      dest->Alias(ptr_, n_rows_in, n_cols_in);
    }

    /**
     * Makes an alias of a particular column.
     *
     * @param col the column to alias
     * @param dest a pointer to an uninitialized vector, which will be
     *        initialized as an alias to the particular column
     */
    template<typename OtherPrecision, bool OtherIsVector>
    void MakeColumnVector(index_t col,
                          Matrix<OtherPrecision, OtherIsVector> *dest) const {
      DEBUG_BOUNDS(col, n_cols_);
      dest->Alias(n_rows_ * col + ptr_, n_rows_);
    }

    /**
     * Makes an alias of a subvector of particular column.
     *
     * @param col the column to alias
     * @param start_row the first row to put in the subvector
     * @param n_rows_new the number of rows of the subvector
     * @param dest a pointer to an uninitialized vector, which will be
     *        initialized as an alias to the particular column's subvector
     */
    template<bool OtherIsVector>
    void MakeColumnSubvector(index_t col, index_t start_row, index_t n_rows_new,
                             Matrix<Precision, OtherIsVector> *dest) const {
      DEBUG_BOUNDS(col, n_cols_);
      DEBUG_BOUNDS(start_row, n_rows_);
      DEBUG_BOUNDS(start_row + n_rows_new, n_rows_ + 1);
      dest->Alias(n_rows_ * col + start_row + ptr_, n_rows_new);
    }

    template<bool OtherIsVector>
    void MakeSubvector(index_t start_row, index_t n_rows_new,
                       Matrix<Precision, OtherIsVector> *dest) const {
      DEBUG_BOUNDS(0, n_cols_);
      DEBUG_BOUNDS(start_row, n_rows_);
      DEBUG_BOUNDS(start_row + n_rows_new, n_rows_ + 1);
      dest->Alias(0 + start_row + ptr_, n_rows_new);
    }

    /**
     * Retrieves a pointer to a contiguous array corresponding to a particular
     * column.
     *
     * @param col the column number
     * @return an array where the i'th element is the i'th row of that
     *         particular column
     */

    Precision *GetColumnPtr(index_t col) {
      DEBUG_BOUNDS(col, n_cols_);
      return n_rows_ * col + ptr_;
    }

    /**
     * Retrieves a pointer to a contiguous array corresponding to a particular
     * column.
     *
     * @param col the column number
     * @return an array where the i'th element is the i'th row of that
     *         particular column
     */
    const Precision *GetColumnPtr(index_t col) const {
      DEBUG_BOUNDS(col, n_cols_);
      return n_rows_ * col + ptr_;
    }

    /**
     * Copies a vector to a matrix column.
     * @param col1 the column number of this matrix
     * @param col2 the column number of the other matrix
     * @param mat the other matrix
     * @return nothing
     */
    template<typename OtherPrecision, bool OtherIsVector>
    void CopyColumnFromMat(index_t col1, index_t col2,
                           Matrix<OtherPrecision, OtherIsVector> &mat) {
      fl::You_have_a_precision_conflict<Precision, OtherPrecision>();
      DEBUG_BOUNDS(col1, n_cols_);
      DEBUG_BOUNDS(col2, mat.n_cols());
      DEBUG_SAME_SIZE(n_rows_, mat.n_rows());

      CopyValues_(GetColumnPtr(col1), mat.GetColumnPtr(col2), n_rows_);
    }
    /**
     * Copies a block of columns to a matrix column.
     * @param col1 the column number of this matrix
     * @param col2 the column number of the other matrixa
     * @param ncols the number of columns
     * @param mat the other matrix
     * @return nothing
     */
    template<typename OtherPrecision, bool OtherIsVector>
    void CopyColumnFromMat(index_t col1, index_t col2, index_t ncols,
                           Matrix<OtherPrecision, OtherIsVector> &mat) {
      fl::You_have_a_precision_conflict<Precision, OtherPrecision>();
      DEBUG_BOUNDS(col1, n_cols_);
      DEBUG_BOUNDS(col2, mat.n_cols());
      DEBUG_BOUNDS(col1 + ncols - 1, n_cols_);
      DEBUG_BOUNDS(col2 + ncols - 1, mat.n_cols());
      DEBUG_SAME_SIZE(n_rows_, mat.n_rows());

      CopyValues_(GetColumnPtr(col1), mat.GetColumnPtr(col2), ncols * n_rows_);
    }

    /**
    * Copies a column of matrix 1  to a column of matrix 2.
    * @param col1 the column number
    * @return nothing
    */
    template<typename OtherPrecision, bool OtherIsVector>
    void CopyVectorToColumn(index_t col,
                            Matrix<OtherPrecision, OtherIsVector> &vec) {
      fl::You_have_a_precision_conflict<Precision, OtherPrecision>();
      DEBUG_ASSERT(vec.n_cols() == 1 || vec.n_rows() == 1);
      DEBUG_BOUNDS(col, n_cols_);

      CopyValues_(GetColumnPtr(col), vec.ptr(), n_rows_);
    }

    /**
     * Changes the number of columns, but REQUIRES that there are no aliases
     * to this matrix anywhere else.
     *
     * If the size is increased, the remaining space is not initialized.
     *
     * @param new_n_cols the new number of columns
     */
    void ResizeNoalias(index_t new_n_cols) {
      resize(n_rows_, new_n_cols);
    }

    /**
     * Swaps all values in this matrix with values in the other.
     *
     * This is different from Swap, because Swap will only change what these
     * point to.
     *
     * @param other an identically sized vector to swap values with
     */
    template<typename OtherPrecision, bool OtherIsVector>
    void SwapValues(Matrix<OtherPrecision, OtherIsVector>* other) {
      DEBUG_SAME_SIZE(n_elements_, other->n_elements());

      SwapValues_(ptr_, other->ptr(), n_elements_);
    }

    /**
     * Copies the values from another matrix to this matrix.
     *
     * @param other the vector to copy from
     */
    template<typename OtherPrecision, bool OtherIsVector>
    void CopyValues(const Matrix<OtherPrecision, OtherIsVector>& other) {
      fl::You_have_a_precision_conflict<Precision, OtherPrecision>();
      DEBUG_SAME_SIZE(n_elements_, other.n_elements());

      CopyValues_(ptr_, other.ptr_, n_elements_);
    }

    template<typename OtherPrecision>
    void CopyValues(const OtherPrecision *other) {
      fl::You_have_a_precision_conflict<Precision, OtherPrecision>();

      CopyValues_(ptr_, other, n_elements_);
    }

    /**
     * Prints to a stream as a debug message.
     *
     * @param name a name that will be printed with the matrix
     * @param stream the stream to print to, defaults to @c stderr
     */
    //We need to templatize this !!!!
    void PrintDebug(const char *name = "", FILE *stream = stderr) const {
      PrintTrait<Precision, IsVector>::Print(*this, name, stream);
    }

  public:
    /**
     * Returns a pointer to the very beginning of the matrix, stored
     * in a column-major format.
     *
     * This is suitable for BLAS and LAPACK calls.
     */
    const Precision *ptr() const {
      return ptr_;
    }

    /**
     * Returns a pointer to the very beginning of the matrix, stored
     * in a column-major format.
     *
     * This is suitable for BLAS and LAPACK calls.
     */
    virtual Precision *ptr() {
      return ptr_;
    }

    /**
     * Gets a particular double at the specified row and column.
     *
     * @param r the row number
     * @param c the column number
     */
    Precision get(index_t r, index_t c) const {
      DEBUG_BOUNDS(r, n_rows_);
      DEBUG_BOUNDS(c, n_cols_);
      return ptr_[c * n_rows_ + r];
    }

    Precision get(index_t r) const {
      DEBUG_BOUNDS(r, n_elements());
      return ptr_[r];
    }

    /**
     * Sets the value at the row and column.
     *
     * @param r the row number
     * @param c the column number
     * @param v the value to set
     */
    void set(index_t r, index_t c, Precision v) {
      DEBUG_BOUNDS(r, n_rows_);
      DEBUG_BOUNDS(c, n_cols_);
      ptr_[c * n_rows_ + r] = v;
    }

    void set(index_t r, Precision v) {
      DEBUG_BOUNDS(r, n_elements());
      ptr_[r] = v;
    }

    /**
     * Gets a reference to a particular row and column.
     *
     * It is highly recommended you treat this as a single value rather than
     * part of an array; use ColumnSlice or Reshaped instead to make
     * subsections.
     */
    Precision &ref(index_t r, index_t c) {
      DEBUG_BOUNDS(r, n_rows_);
      DEBUG_BOUNDS(c, n_cols_);
      return ptr_[c * n_rows_ + r];
    }

    Precision &ref(index_t r) {
      DEBUG_BOUNDS(r, n_elements());
      return ptr_[r];
    }

    Precision &operator[](index_t r) {
      DEBUG_BOUNDS(size_t(r), n_elements());
      return ptr_[r];
    }

    const Precision &operator[](index_t r) const {
      DEBUG_BOUNDS(size_t(r), n_elements());
      return ptr_[r];
    }

    /** Returns the number of columns. */
    index_t n_cols() const {
      return n_cols_;
    }

    /** Returns the number of rows. */
    index_t n_rows() const {
      return n_rows_;
    }

    /**
     * Returns the total number of elements (power user).
     *
     * This is useful for iterating over all elements of the matrix when the
     * row/column structure is not important.
     */
    size_t n_elements() const {
      // TODO: putting the size_t on the outside may be faster (32-bit
      // versus 64-bit multiplication in cases) but is more likely to result
      // in bugs
      return n_elements_;
    }

    /**
     * It is exaclty the same like n_elements()
     * This definition provided a uniform interface for
     * blas/lapck type operations
     */
    index_t length() const {
      return n_elements();
    }

    size_t size() const {
      return n_elements();
    }

    bool should_free() const {
      return capacity_ != 0;
    }

  private:
    void AssertUninitialized_() const {
      DEBUG_ASSERT_MSG(n_rows_ == std::numeric_limits<index_t>::max(), "Cannot re-init matrices.");
    }

    void Uninitialize_() {
      ptr_ = NULL;
      n_rows_ = 0;
      n_cols_ = 0;
      n_elements_ = 0;
      capacity_ = 0;
    }

    void Resize_(index_t size_in) {
      if (unlikely(size_in > capacity_)) {
        capacity_ += size_in;
        Precision *old_ptr = ptr_;
        ptr_ = new (std::nothrow) Precision[capacity_];
        if (ptr_==NULL) {
          fl::logger->Die() << "Failed to allocate memory of :" <<
          capacity_ *sizeof(Precision) << " bytes";
        }
        ::memcpy(ptr_, old_ptr, n_elements_ * sizeof(Precision));
        delete[] old_ptr;
      }

      n_elements_ = size_in;
    }

    template<typename OtherPrecision>
    void CopyValues_(
      Precision *dest,
      const OtherPrecision *src,
      index_t count
    ) {
      while (count--) {
        *dest++ = *src++;
      }
    }

    void CopyValues_(
      Precision *dest,
      const Precision *src,
      index_t count
    ) {
      ::memcpy(dest, src, count * sizeof(Precision));
    }

    template<typename OtherPrecision>
    void SwapValues_(
      Precision *ptr,
      OtherPrecision *other_ptr,
      index_t count
    ) {
      while (count--) {
        Precision temp = *ptr;
        *ptr++ = *other_ptr;
        *other_ptr++ = temp;
      }
    }

    void SwapValues_(
      Precision *ptr,
      Precision *other_ptr,
      index_t count
    ) {
      const size_t BUF_SIZE = 64 / sizeof(Precision);
      Precision buf[BUF_SIZE];

      while (count > BUF_SIZE) {
        ::memcpy(buf, ptr, BUF_SIZE * sizeof(Precision));
        ::memcpy(ptr, other_ptr, BUF_SIZE * sizeof(Precision));
        ::memcpy(other_ptr, buf, BUF_SIZE * sizeof(Precision));

        count -= BUF_SIZE;
        ptr += BUF_SIZE;
        other_ptr += BUF_SIZE;
      }
      if (count) {
        ::memcpy(buf, ptr, count * sizeof(Precision));
        ::memcpy(ptr, other_ptr, count * sizeof(Precision));
        ::memcpy(other_ptr, buf, count * sizeof(Precision));
      }
    }

    template<typename Archive>
    void save(Archive &ar,
              const unsigned int file_version) const {
      try {
        ar << boost::serialization::make_nvp("ops",
            boost::serialization::base_object<fl::dense::ops>(*this));      
        ar << boost::serialization::make_nvp("n_rows", n_rows_);
        ar << boost::serialization::make_nvp("n_cols", n_cols_);
        ar << boost::serialization::make_nvp("n_elements", n_elements_);
        for (index_t i=0; i<n_elements_; ++i) {
          ar << BOOST_SERIALIZATION_NVP(ptr_[i]);
        }
      }
      catch(const boost::archive::archive_exception &e) {
        fl::logger->Die()<< "Matrix archiving (save): "<< e.what();
      }
    }

    template<typename Archive>
    void load(Archive &ar,
              const unsigned int file_version) {
      Destruct();
      try {
        ar >> boost::serialization::make_nvp("ops",
            boost::serialization::base_object<fl::dense::ops>(*this));      
        ar >> boost::serialization::make_nvp("n_rows", n_rows_);
        ar >> boost::serialization::make_nvp("n_cols", n_cols_);
        ar >> boost::serialization::make_nvp("n_elements", n_elements_);
      }
      catch(const boost::archive::archive_exception &e) {
        fl::logger->Die()<< "Matrix archiving (load): "<< e.what();
      }

      capacity_ = n_elements_ + (n_elements_ == 0);
      try {
        ptr_ = new Precision[capacity_];   
      }
      catch (const std::bad_alloc &e) {
         fl::logger->Die() << "Failed to allocate memory of :" <<
         capacity_ *sizeof(Precision) << " bytes";
      }
      try {
        for (index_t i=0; i<n_elements_; ++i) {
          ar >> BOOST_SERIALIZATION_NVP(ptr_[i]);
        }
      }
      catch(const boost::archive::archive_exception &e) {
        fl::logger->Die()<< "Matrix archiving (load): "<< e.what();
      }
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

} // namespace dense
} // namespace fl
#endif
