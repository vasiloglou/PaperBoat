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
#ifndef FASTLIB_DENSE_SMALL_MATRIX_H_
#define FASTLIB_DENSE_SMALL_MATRIX_H

#include "matrix.h"

namespace fl {
namespace dense {
/**
     * Low-overhead vector if length is known at compile time.
     */
template<typename Precision, int t_length>
class SmallVector : public Matrix<Precision, true> {
  private:
    Precision array_[t_length];

  public:
    SmallVector() {
      Alias(array_, t_length);
    }
    ~SmallVector() {}

  public:
    index_t length() const {
      return t_length;
    }

    Precision *ptr() {
      return array_;
    }

    const Precision *ptr() const {
      return array_;
    }

    Precision operator [](index_t i) const {
      DEBUG_BOUNDS(i, t_length);
      return array_[i];
    }

    Precision &operator [](index_t i) {
      DEBUG_BOUNDS(i, t_length);
      return array_[i];
    }

    Precision get(index_t i) const {
      DEBUG_BOUNDS(i, t_length);
      return array_[i];
    }
};

/** @brief A Vector is a Matrix of double's.
 */
typedef Matrix<double, true> Vector;

/**
 * Low-overhead matrix if size is known at compile time.
 */
template<typename Precision, int t_rows, int t_cols>
class SmallMatrix : public Matrix<Precision, false> {
  private:
    Precision array_[t_cols][t_rows];

  public:
    SmallMatrix() {
      Alias(array_[0], t_rows, t_cols);
    }
    ~SmallMatrix() {}

  public:
    const Precision *ptr() const {
      return array_[0];
    }

    Precision *ptr() {
      return array_[0];
    }

    Precision get(index_t r, index_t c) const {
      DEBUG_BOUNDS(r, t_rows);
      DEBUG_BOUNDS(c, t_cols);
      return array_[c][r];
    }

    void set(index_t r, index_t c, Precision v) {
      DEBUG_BOUNDS(r, t_rows);
      DEBUG_BOUNDS(c, t_cols);
      array_[c][r] = v;
    }

    Precision &ref(index_t r, index_t c) {
      DEBUG_BOUNDS(r, t_rows);
      DEBUG_BOUNDS(c, t_cols);
      return array_[c][r];
    }

    index_t n_cols() const {
      return t_cols;
    }

    index_t n_rows() const {
      return t_rows;
    }

    size_t n_elements() const {
      // TODO: putting the size_t on the outside may be faster (32-bit
      // versus 64-bit multiplication in cases) but is more likely to result
      // in bugs
      return size_t(t_rows) * size_t(t_cols);
    }

    Precision *GetColumnPtr(index_t col) {
      DEBUG_BOUNDS(col, t_cols);
      return array_[col];
    }

    const Precision *GetColumnPtr(index_t col) const {
      DEBUG_BOUNDS(col, t_cols);
      return array_[col];
    }
};

} // namespace dense
}  // namespace fl

#endif
