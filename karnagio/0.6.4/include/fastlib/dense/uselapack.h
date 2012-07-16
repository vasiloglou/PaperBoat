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
 * @file uselapack.h
 *
 * Integration with BLAS and LAPACK.
 */

#ifndef LA_USELAPACK_H
#define LA_USELAPACK_H

#include "matrix.h"
#include "fastlib/col/arraylist.h"

#define DEBUG_VECSIZE(a, b) \
  DEBUG_SAME_SIZE((a).length(), (b).length())
#define DEBUG_MATSIZE(a, b) \
  DEBUG_SAME_SIZE((a).n_rows(), (b).n_rows());\
  DEBUG_SAME_SIZE((a).n_cols(), (b).n_cols())
#define DEBUG_MATSQUARE(a) \
  DEBUG_SAME_SIZE((a).n_rows(), (a).n_cols())

/*
 * TODO: this could an overhaul; LAPACK failures may mean either input
 * problems or just some linear algebra result (e.g. singular matrix).
 *
 * Perhaps extend success_t?
 */
#define SUCCESS_FROM_LAPACK(info) \
  (likely((info) == 0) ? SUCCESS_PASS : SUCCESS_FAIL)

#ifndef NO_LAPACK
#define USE_LAPACK
#define USE_BLAS_L1
#endif

#if defined(DOXYGEN) && !defined(USE_LAPACK)
/* Use the doxygen from the blas level 1 code. */
#define USE_LAPACK
#endif

#ifdef USE_LAPACK
#include "cppblas.h"
#include "cpplapack.h"
#endif

namespace fl {
/**
 *  @brief Linear-algebra routines.
 *
 * This encompasses most basic real-valued vector and matrix math.
 * Most functions are written in a similar style, so after using a few it
 * should be clear how others are used.  For instance, input arguments
 * come first, and the output arguments are last.
 *
 * Many functions have several versions:
 * The functions are all templatized in terms of precision
 * Another template parameter is the initialization of the result
 * So in some cases we want to reuse the same matrix, so it is not necessary
 * to reallocate them
 *
 * All the algorithms are ignorant of Matrix/Vector
 *
 * @code
 * void SomeExampleCode<float, fl::la::Init>(const Matrix& a, const Matrix& b) {
 *   Matrix c;
 *   // c must not be initialized
 *   fl::la::Mul<Precision, fl::la::Trans, fl::la::NoTrans, fl::la::Init>(a, b, &c);
 * }
 * @endcode
 * The enums fl::la::Trans, fl::la::NoTrans are used to declare if the matrix
 * will be used in the transpose or non-transpose mode
 *
 * The enums fl::la::Init, fl::la::Overwrite state whether the resulting
 * container is going to be initialized by the function or if it is
 * already initialized
 *
 * - The Expert version, or power-user version.  We recommend you avoid this
 * version whenever possible.  These are the closest mapping to the LAPACK
 * or BLAS routine.  These will require that memory for destinations is
 * already allocated but the memory itself is not initalized.  These usually
 * take a Matrix or Vector pointer, or sometimes just a pointer to a slab of
 * doubles/floats.  Sometimes, one of the "inputs" is completely destroyed while
 * LAPACK performs its computations.
 *
 *
 * FINAL WARNING:
 * With all of these routines, be very careful about passing in an argument
 * as both a source and destination -- usually this is an error and
 * will result in garbage results, except for simple
 * addition/scaling/subtraction.
 */
namespace la {
/**
 * @brief Initialization of a container
 *        If you want the function to reuse an already initialized
 *        container, use la::Overwrite. Otherwise if you want the
 *        function to initialize the result, use la::Init
 * @code
 * enum MemoryAlloc {Init=0, Overwrite};
 * @endcode
 */
enum MemoryAlloc {Init = 0, Overwrite};
/**
 * @brief Sometimes in linear algebra we need the transpose
 *        of a matrix. It is in general a bad idea to transpose the
 *        matrix. BLAS has as an option Transpose/Non Transpose mode
 *        This enum lets you choose if you want the matrix as it is
 *        or in the transpose mode.
 * @code
 *  enum TransMode {NoTrans=0, Trans};
 * @endcode
 */
enum TransMode {NoTrans = 0, Trans};

/**
 * @brief We are using this trait so that we can handle
 *        at compile time the initialization of the result
 * @code
 *  template<MemoryAlloc T>
 *   class AllocationTrait {
 *    public:
 *     template<typename Container>
 *      static inline void Init(index_t length, Container *cont);
 *
 *     template<typename Container>
 *      static inline void Init(index_t dim1, index_t dim2, Container *cont);
 *   };
 * @endcode
 */
template<MemoryAlloc T>
class AllocationTrait {
  public:
    template<typename Container>
    static inline void Init(index_t length, Container *cont);

    static inline void Init(index_t length, float **cont);

    static inline void Init(index_t length, double **cont);

    static inline void Init(index_t length, long double **cont);

    template<typename Container>
    static inline void Init(index_t dim1, index_t dim2, Container *cont);
};

template<>
class AllocationTrait<Init> {
  public:
    template<typename Container>
    static inline void Init(index_t length, Container *cont) {
      cont->Init(length);
    }

    static inline void Init(index_t length, float **cont) {
      *cont =  new float[length];
    }

    static inline void Init(index_t length, double **cont) {
      *cont =  new double[length];
    }

    static inline void Init(index_t length, long double **cont) {
      *cont =  new long double[length];
    }

    template<typename Container>
    static inline void Init(index_t dim1, index_t dim2, Container *cont) {
      cont->Init(dim1, dim2);
    }
};

template<>
class AllocationTrait<Overwrite> {
  public:
    template<typename Container>
    static inline void Init(index_t length, Container *cont) {
      DEBUG_SAME(length, cont->length());
    }

    static inline void Init(index_t length, float **cont) {
      DEBUG_ASSERT_MSG(*cont != NULL, "Uninitialized pointer");
    }

    static inline void Init(index_t length, double  **cont) {
      DEBUG_ASSERT_MSG(*cont != NULL, "Uninitialized pointer");
    }

    static inline void Init(index_t length, long double **cont) {
      DEBUG_ASSERT_MSG(*cont != NULL, "Uninitialized pointer");
    }

    template<typename Container>
    static inline void Init(index_t dim1, index_t dim2, Container *cont) {
      DEBUG_SAME_SIZE(dim1, cont->n_rows());
      DEBUG_SAME_SIZE(dim2, cont->n_cols());
    }
};


/**
 * @brief Scales the rows of a column-major matrix by a different value for
 * each row.
 * @code
 *   template<Precision>
 *   void ScaleRows(index_t n_rows, index_t n_cols,
 *       const Precision *scales, const Precision *matrix);
 *   // example:
 *   double a[3][3]={{0, 1, 2}, {-1, 4, 2}, {-3, 0, -2}};
 *   double scales[3]={1, -3, 1};
 *   fl::la::Scale(3, 4, scales, a);
 * @endcode
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float double
 * @param n_rows, number of rows of the matrix
 * @param n_cols, number of columns of the matrix
 * @param scales, an array with the values that scale the rows
 * @param matrix, a column major matrix unfolded in a memory slab
 */
template<typename Precision>
inline void ScaleRows(index_t n_rows, index_t n_cols,
                      const Precision *scales, Precision *matrix) {
  do {
    for (index_t i = 0; i < n_rows; i++) {
      matrix[i] *= scales[i];
    }
    matrix += n_rows;
  }
  while (--n_cols);
}

/**
 * @brief Finds the square root of the dot product of a vector with itself.
 */
template<typename Precision>
inline Precision LengthEuclidean(index_t length, const Precision *x) {
  return CppBlas<Precision>::nrm2(length, x, 1);
}

/**
 * @brief Finds the square root of the dot product of a vector or matrix with itself
 *       (\f$\sqrt{x} \cdot x \f$).
 * @code
 *  template<typename Precision, bool IsVector>
 *  inline Precision LengthEuclidean(const GenMatrix<Precision, IsVector> &x);
 *  // example
 *  fl::la::GenMatrix<float> a;
 *  fl::la::Random(3,1, &a);
 *  float norm = fl::la::LengthEuclidean(a);
 * @endcode
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float double
 * @param IsVector, boolean variable, for backward compatibility
 * @param x, a matrix
 *
 */
template<typename Precision, bool IsVector>
inline Precision LengthEuclidean(const GenMatrix<Precision, IsVector> &x) {
  return LengthEuclidean<Precision>(x.length(), x.ptr());
}
/**
  * @brief Finds the dot-product of two arrays
  * (\f$\vec{x} \cdot \vec{y}\f$).
  */
template<typename Precision>
inline long double Dot(index_t length, const Precision *x, const Precision *y) {
  return CppBlas<Precision>::dot(length, x, 1, y, 1);
}


/**
 * @brief Finds the dot product of two arrays
 *        (\f$x \cdot y\f$).
 * @code
 *   template<typename Precision, bool IsVector>
 *   Precision Dot(const GenMatrix<Precision, IsVector> &x,
 *                 const GenMatrix<Precision, IsVector> &y);
 *   // example
 *   fl::la::GenMatrix<double> x;
 *   fl::la::GenMatrix<double> y;
 *   fl::la::Random(1, 4, &x);
 *   fl::la::Random(1, 4, &y);
 *   double dot_prod = fl::la::Dot(x, y);
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float, double
 * @param IsVector, template parameter for backward compatibility
 * @param x, a GenMatrix
 * @param y, a GenMatrix, with the same dimensions as x
 */
template<typename Precision, bool IsVector>
inline long double Dot(const GenMatrix<Precision, IsVector> &x,
                       const GenMatrix<Precision, IsVector> &y) {
  DEBUG_SAME_SIZE(x.length(), y.length());
  return Dot<Precision>(x.length(), x.ptr(), y.ptr());
}

/**
 * @brief Scales an array in-place by some factor
 * (\f$\vec{x} \gets \alpha \vec{x}\f$).
 */
template<typename Precision>
inline void ScaleExpert(index_t length, Precision alpha, Precision *x) {
  CppBlas<Precision>::scal(length, alpha, x, 1);
}

/**
 * @brief Scales an array in-place by some factor
 * (\f$ x \gets \alpha x\f$).
 * @code
 *  template<typename Precision, bool IsVector>
 *  void ScaleExpert(const Precision alpha, GenMatrix<Precision, IsVector> *x);
 *  //example
 *  fl::la::GenMatrix<double> x;
 *  fl::la::Random(3, 5, &x);
 *  double alpha=2.4;
 *  fl::la::ScaleExpert(alpha, &x);
 * @endcode
 *
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float,double
 * @param IsVector, template parameter for backward compatibility
*/
template<typename Precision, bool IsVector>
inline void ScaleExpert(const Precision alpha, GenMatrix<Precision, IsVector> *x) {
  ScaleExpert<Precision>(x->length(), alpha, x->ptr());
}


/**
 * @brief Scales each row of the matrix to a different scale.
 *         X <- diag(d) * X
 * @code
 *  template<typename Precision, bool IsVector1, bool IsVector2 >
 *  void ScaleRows(const GenMatrix<Precision, IsVector1>& d,
 *                       GenMatrix<Precision, IsVector2> *X);
 *  //example
 *  fl::la::GenMatrix<float> x;
 *  fl::la::Random(4,5, &x);
 *  fl::la::GenMatrix<fload> d;
 *  fl::la::Random(4, 1, &d); // since d must be one dimensional n_cols must
 *                            // be one. It must be a column vector
 *  fl::la::ScaleRows(d, &x);
 *  // we could alternative declare d
 *   GenMatrix<float, true> d;
 *
 * @endcode
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float double
 * @param IsVector1, IsVector2, boolean parameters for backward compatibility
 * @param d a length-M vector with each value corresponding
 * @param X the matrix to scale
 */
template<typename Precision, bool IsVector1, bool IsVector2 >
inline void ScaleRows(const GenMatrix<Precision, IsVector1>& d,
                      GenMatrix<Precision, IsVector2> *X) {
  DEBUG_SAME_SIZE(d.n_cols(), 1);
  DEBUG_SAME_SIZE(d.n_rows(), X->n_rows());
  ScaleRows<Precision>(d.n_rows(), X->n_cols(), d.ptr(), X->ptr());
}
/**
  * @brief Sets an array to another scaled by some factor
  *        (\f$ y \gets \alpha x\f$).
  * @code
  *  template<typename Precision, MemoryAlloc M, bool IsVector>
  *  void Scale(const Precision alpha, const GenMatrix<Precision, IsVector>
  *       &x, GenMatrix<Precision, IsVector> *y);
  *  //example
  *  fl::la::GenMatrix<double> x;
  *  fl::la::Random(4, 6, &x);
  *  fl::la::GenMatrix<double> y;
  *  double alpha = 3.44;
  * // The call sounds weird, but we used a C++ trick to achieve that
  * // You can use the following syntax works
  *  Scale<fl::la::Init>(alpha, x, &y);
  * // or initializr y first
  * y.Init(4, 6);
  *  Scale<fl::la::Overwrite> (alpha, x, &y);
  * @endcode
  *
  * @param Precision, template parameter for the precision, currently supports
  *        float, double.
  * @param IsVector, bool parametre for bakward compatibility with Vectors
  * @param M, this one is the only one you have to define, it can be
  *           fl::la::Init if you want the function to allocate space for the
  *           result or, fl::la::Overwrite if the result is already initialized
  * @param alpha, the scaling factor
  * @param x, the matrix to be scaled
  * @param y, the scaled matrix
  */
template<MemoryAlloc M>
class Scale {
  public:
    template<typename Precision, bool IsVector>
    Scale(const Precision alpha, const GenMatrix<Precision, IsVector>
          &x, GenMatrix<Precision, IsVector> *y) {
      DEBUG_SAME_SIZE(x.n_rows(), y->n_rows());
      DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
      AllocationTrait<M>::Init(x.n_rows(), y.n_cols(), y);
      Scale(x.length(), alpha, x.ptr(), y->ptr());
    }
};
/**
* @brief Adds a scaled array to an existing array
* (\f$\vec{y} \gets \vec{y} + \alpha \vec{x}\f$).
*/
template<typename Precision>
inline void AddExpert(index_t length,
                      Precision alpha, const Precision *x, Precision *y) {
  CppBlas<Precision>::axpy(length, alpha, x, 1, y, 1);
}

/**
 * @brief Sets an array to the sum of two arrays
 * (\f$\vec{z} \gets \vec{y} + \vec{x}\f$).
 */
template<typename Precision>
inline void AddExpert(index_t length,
                      const Precision *x, const Precision *y, Precision *z) {
  ::memcpy(z, y, length * sizeof(Precision));
  AddExpert<Precision>(length, 1.0, x, z);
}


/**
 * @brief Adds a scaled vector to an existing vector
 *        (\f$ y \gets y + \alpha x\f$).
 * @code
 *   template<typename Precision, bool IsVector>
 *   void AddExpert(Precision alpha,
 *                  const GenMatrix<Precision, IsVector> &x,
 *                  GenMatrix<Precision, IsVector> *y);
 *   // example
 *   float alpha
 *   fl::la::GenMatrix<float> x;
 *   fl::la::Random(5, 4, &x);
 *   fl::la::GenMatrix<float> y;
 *   fl::la::Random(5, 4, &y);
 *   fl::la::AddExpert(alpha, x, &y);
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float double
 * @param IsVector, boolean template parameter for backward compatibility with
 *                  Vector
 */
template<typename Precision, bool IsVector>
inline void AddExpert(Precision alpha,
                      const GenMatrix<Precision, IsVector> &x,
                      GenMatrix<Precision, IsVector> *y) {
  DEBUG_SAME_SIZE(x.n_rows(), y->n_rows());
  DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
  AddExpert(x.length(), alpha, x.ptr(), y->ptr());
}
/* --- Matrix/Vector Addition --- */

/**
 * @brief Adds a vector to an existing vector
 *        (\f$ y \gets y + x\f$);
 * @code
 *   template<typename Precision, bool IsVector>
 *   void AddTo(const GenMatrix<Precision, IsVector> &x,
 *              GenMatrix<Precision, IsVector> *y);
 *   // example
 *   fl::la::GenMatrix<double> x;
 *   fl::la::Random(7, 8, &x);
 *   fl::la::GenMatrix<double> y;
 *   fl::la::Random(7, 9, &y);
 *   fl::la::AddTo(x, y);
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float double
 * @param IsVector, template parameter, for backward compatibility with Vector
 */
template<typename Precision, bool IsVector>
inline void AddTo(const GenMatrix<Precision, IsVector> &x,
                  GenMatrix<Precision, IsVector> *y) {
  DEBUG_SAME_SIZE(x.n_rows(), y->n_rows());
  DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
  AddExpert(Precision(1.0), x, y);
}

/**
 * @brief Sets a vector to the sum of two vectors
 *        (\f$ z \gets y + x\f$).
 * @code
 * // Add is a function. We use the following trick though
 * // To use it as a constructor of a class.
 * // The Precision and IsVector are deduced automatically
 * // from the syntax. The M (Init, Overwrite) have to be defined
 * // explicitly. The example will shed more light on this issue
 *  template<MemoryAlloc M>
 *  class Add {
 *   public:
 *    template<typename Precision, , bool IsVector>
 *    Add(const GenMatrix<Precision, IsVector> &x,
 *       const GenMatrix<Precision, IsVector> &y,
 *             GenMatrix<Precision, IsVector> *z);
 *  };
 *  // example
 *  fl::la::GenMatrix<double> x;
 *  fl::la::GenMatrix<double> y;
 *  fl::la::Random(5, 6, &x);
 *  fl::la::Random(5, 6, &y);
 *  fl::la::GenMatrix<double> z;
 *  // Add will initialize z
 *  fl::la::Add<fl::la::Init>(x, y, &z);
 *  // alternatively
 *  fl::la::GenMatrix<double> w;
 *  w.Init(5, 6);
 * // Add desn't allocate space for w
 *  fl::la::Add<fl::la::Overwrite>(x, y, &w)/
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. This parameter is automatically deduced from the
 *        function arguments
 * @param IsVector, for backward compatibility with Vector. This parameter is automatically deduced from the
 *        function arguments
 */
template<MemoryAlloc M>
class Add {
  public:
    template<typename Precision, bool IsVector>
    Add(const GenMatrix<Precision, IsVector> &x,
        const GenMatrix<Precision, IsVector> &y,
        GenMatrix<Precision, IsVector> *z) {
      DEBUG_SAME_SIZE(x.n_rows(), y.n_rows());
      DEBUG_SAME_SIZE(x.n_cols(), y.n_cols());
      AllocationTrait<M>::Init(x.n_rows(), x.n_cols(), z);
      AddExpert(x.length(), x.ptr(), y.ptr(), z->ptr());
    }
};


/**
 * @brief Subtracts an array from an existing array
 * (\f$\vec{y} \gets \vec{y} - \vec{x}\f$).
 */
template<typename Precision>
inline void SubFrom(index_t length, const Precision *x, Precision *y) {
  AddExpert<Precision>(length, -1.0, x, y);
}

/**
 * @brief Sets an array to the difference of two arrays
 * (\f$\vec{x} \gets \vec{y} - \vec{x}\f$).
 */
template<typename Precision>
inline void SubExpert(index_t length,
                      const Precision *x, const Precision *y, Precision *z) {
  ::memcpy(z, y, length * sizeof(Precision));
  SubFrom<Precision>(length, x, z);
}
/* --- Matrix/Vector Subtraction --- */

/**
 * @brief Subtracts a vector from an existing vector
 *        (\f$ y \gets y - x \f$).
 * @code
 *  template<typename Precision, bool IsVector>
 *  void SubFrom(const GenMatrix<Precision, IsVector> &x,
 *                     GenMatrix<Precision, IsVector> *y);
 *   //example
 *   fl::la::GenMatrix<float> x;
 *   fl::la::GenMatrix<float> y;
 *   fl::la::Random(3, 4, &x);
 *   fl::la::Random(3, 4, &y);
 *   fl::la::SubFrom(x, &y);
 * @endcode
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. The type is automatically deduced by the function
 *        arguments
 * @param IsVector, boolean templated parameter for backward compatibility
 *                  with Vector. It is automatically deduced from the
 *                  function arguments
 */
template<typename Precision, bool IsVector>
inline void SubFrom(const GenMatrix<Precision, IsVector> &x,
                    GenMatrix<Precision, IsVector> *y) {
  DEBUG_SAME_SIZE(x.n_rows(), y->n_rows());
  DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
  SubFrom(x.length(), x.ptr(), y->ptr());
}

/**
 * @brief Sets a vector to the difference of two vectors
 *        (\f$ z \gets y - x\f$).
 * @code
 *   // Although Sub is a function it is declared as a class.
 *   // This is because the M argument has to be explicitly
 *   // defined while Precision, and IsVector can be
 *   // deduced automatically by the compiler
 *   // The following example will shed more light
 *   template<MemoryAlloc M>
 *   class Sub {
 *    public:
 *    template<typename Precision, bool IsVector>
 *    Sub(const GenMatrix<Precision, IsVector> &x,
 *        const GenMatrix<Precision, IsVector> &y,
 *              GenMatrix<Precision, IsVector> *z);
 *  };
 *  // example
 *  fl::la::GenMatrix<double> x;
 *  fl::la::GenMatrix<double> y;
 *  fl::la::Random(4, 4, &x);
 *  fl::la::Random(4, 4, &z);
 *  fl::la::GenMatrix z;
 *  // Sub will allocate space for z
 *  fl::la::Sub<fl::la::Init>(x, y, &z);
 *  fl::la::GenMatrix<double> w;
 *  w.Init(4, 4);
 *  // Sub will not allocate space for z
 *  fl::la::Sub<fl::la::Overwrite>(x, y, &z);
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is automatically deduced from function arguments
 * @param IsVector, bool parameter, for backward compatibility with Vector.
 *                  It is automatically deduced from the function arguments
 * @param M, it can be fl::la::Init if we want the function to initialize the
 *           result, or fl::la::Overwrite if the result has already been initialized
 */
template<MemoryAlloc M>
class Sub {
  public:
    template<typename Precision, bool IsVector>
    Sub(const GenMatrix<Precision, IsVector> &x,
        const GenMatrix<Precision, IsVector> &y,
        GenMatrix<Precision, IsVector> *z) {
      DEBUG_SAME_SIZE(x.n_rows(), y.n_rows());
      DEBUG_SAME_SIZE(x.n_cols(), y.n_cols());
      AllocationTrait<M>::Init(x.n_rows(), x.n_cols(), z);
      SubExpert<Precision>(x.length(), x.ptr(), y.ptr(), z->ptr());
    }
};

/* --- Matrix Transpose --- */

/*
 * TODO: These could be an order of magnitude faster if we use the
 * cache-efficient Morton layout matrix transpose algoritihm.
 */

/**
 * @brief Computes a square matrix transpose in-place
 *        (\f$X \gets X'\f$).
 * @code
 *   template<typename Precision>
 *   void TransposeSquare(GenMatrix<Precision, false> *X);
 *   // example
 *   fl::la::GenMatrix<float> x;
 *   fl::la::Random(4, 4, &x);
 *   fl::la::TransposeSquare(&x);
 * @endcode
 *
 * @param Precision, template parameter for the precision, Automatically
 *                   deduced by the function arguments
 */
template<typename Precision>
inline void TransposeSquare(GenMatrix<Precision, false> *X) {
  DEBUG_MATSQUARE(*X);
  index_t nr = X->n_rows();
  for (index_t r = 1; r < nr; r++) {
    for (index_t c = 0; c < r; c++) {
      Precision temp = X->get(r, c);
      X->set(r, c, X->get(c, r));
      X->set(c, r, temp);
    }
  }
}

/**
 * @brief Sets a matrix to the transpose of another
 *        (\f$Y \gets X'\f$).
 * @code
 *   // Although Transpose is a function we declare it as a class and
 *   // we use it as a constructor. This is because the template M parameter
 *   // has to be explicitly defined, while the others  can be deduced.
 *   // The following example will shed some light
 *   template<MemoryAlloc M>
 *   class Transpose {
 *    public:
 *     template<typename Precision>
 *     Transpose(const GenMatrix<Precision, false> &X,
 *                     GenMatrix<Precision, false> *Y);
 *   };
 *   // example
 *   fl::la::GenMatrix<double> x;
 *   fl::la::GenMatrix<double> y;
 *   fl::la::Random(4, 6, &x);
 *   fl::la::Transpse<fl::la::init>(x, &y);
 *   // or if we want to initialize on our own
 *   fl::la::GenMatrix w;
 *   w.Init(6, 4);
 *   fl::la::Transpose<fl::la::Overwrite>(x, &y);
 *
 * @endcode
 * @param Precision, template parameter for the precision, automatically
 *                   deduced from the functtion arguments
 * @param M, it can be fl::la::Init if we want the function to initialize the
 *           result, or fl::la::Overwrite if the result has already been initialized
 */
template<MemoryAlloc M>
class Transpose {
  public:
    template<typename Precision>
    Transpose(const GenMatrix<Precision, false> &X,
              GenMatrix<Precision, false> *Y) {
      AllocationTrait<M>::Init(X.n_cols(), X.n_rows(), Y);
      index_t nr = X.n_rows();
      index_t nc = X.n_cols();
      for (index_t r = 0; r < nr; r++) {
        for (index_t c = 0; c < nc; c++) {
          Y->set(c, r, X.get(r, c));
        }
      }
    }
};




/**
 * @brief Matrix Multiplication, close to the Blas format
 *        (\f$  C = A *B + b * C \f$)
 *        A, B  can be in normal or transpose mode
 * @code
 *  // Although MulExpert is a function, we declare it as a
 *  // class constructor, only because IsTransA, IsTransB
 *  // have to be declared explicitly. The other template
 *  // parameters can be automatically deduced by the function arguments
 *  // The following example will shed some light
 *  template<TransMode IsTransA, TransMode IsTransB>
 *  class MulExpert {
 *   public:
 *    template<typename Precision>
 *    MulExpert(Precision alpha,
 *              const GenMatrix<Precision, false> &a,
 *              const GenMatrix<Precision, false> &b,
 *              Precision beta,
 *              GenMatrix<Precision, false> *c);
 *     template<typename Precision>
 *  MulExpert(const Precision alpha,
 *            const GenMatrix<Precision, false> &A,
 *            const GenMatrix<Precision, true> &x,
 *                  Precision beta,
 *                  GenMatrix<Precision, true> *y);
 *  };
 *  // example
 *  fl::la::GenMatrix<double> a;
 *  fl::la::GenMatrix<double> b;
 *  fl::la::GenMatrix<double> c;
 *  fl::la::Random(4, 6, &a);
 *  fl::la::Random(5, 6, &b);
 *  fl::la::Random(4, 5, &c);
 *  double beta=5.33;
 *  fl::la::MulExpert<fl::la::NoTrans, fl::la::Trans>(a, b, beta, c);
 * @endcode
 *
 * @param IsTransA, it can be fl::la::NoTrans, fl::la::Trans, if we are using
 *                  A or A'
 * @param IsTransB it can be fl::la::NoTrans, fl::la::Trans, if we are using
 *                  B or B'
 * @param Precision, template parameter for the precision, currently supports
 *        float,double. It is automatically deduced from the function arguments
 */
template < TransMode IsTransA, TransMode IsTransB = NoTrans >
class MulExpert {
  public:
    template<typename Precision>
    MulExpert(const Precision alpha,
              const GenMatrix<Precision, false> &A,
              const GenMatrix<Precision, true> &x,
              Precision beta,
              GenMatrix<Precision, true> *y) {
      DEBUG_ASSERT(x.ptr() != y->ptr());
      index_t n;
      if (IsTransA == true) {
        n = A.n_cols();
      }
      else {
        n = A.n_rows();
      }
      DEBUG_SAME_SIZE(n, y->n_rows());
      DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
      if (IsTransA == true) {
        DEBUG_SAME_SIZE(A.n_rows(), x.n_rows());
        CppBlas<Precision>::gemv("T", A.n_rows(), A.n_cols(),
                                 alpha, A.ptr(), A.n_rows(), x.ptr(), 1,
                                 beta, y->ptr(), 1);
      }
      else {
        DEBUG_SAME_SIZE(A.n_cols(), x.n_rows());
        CppBlas<Precision>::gemv("N", A.n_rows(), A.n_cols(),
                                 alpha, A.ptr(), A.n_rows(), x.ptr(), 1,
                                 beta, y->ptr(), 1);
      }
    }

    template<typename Precision>
    MulExpert(Precision alpha,
              const GenMatrix<Precision, false> &A,
              const GenMatrix<Precision, false> &B,
              Precision beta,
              GenMatrix<Precision, false> *C) {
      DEBUG_ASSERT(B.ptr() != C->ptr());
      if (IsTransB == true) {
        if (IsTransA == true) {
          DEBUG_SAME_SIZE(A.n_rows(), B.n_cols());
          DEBUG_SAME_SIZE(A.n_cols(), C->n_rows());
          DEBUG_SAME_SIZE(C->n_cols(), B.n_rows());
          CppBlas<Precision>::gemm("T", "T",
                                   C->n_rows(), C->n_cols(),  A.n_rows(),
                                   alpha, A.ptr(), A.n_rows(), B.ptr(), B.n_rows(),
                                   beta, C->ptr(), C->n_rows());
        }
        else {
          DEBUG_SAME_SIZE(A.n_cols(), B.n_cols());
          DEBUG_SAME_SIZE(A.n_rows(), C->n_rows());
          DEBUG_SAME_SIZE(C->n_cols(), B.n_rows());
          CppBlas<Precision>::gemm("N", "T",
                                   C->n_rows(), C->n_cols(),  A.n_cols(),
                                   alpha, A.ptr(), A.n_rows(), B.ptr(), B.n_rows(),
                                   beta, C->ptr(), C->n_rows());
        }
      }
      else {
        if (IsTransA == true) {
          DEBUG_SAME_SIZE(A.n_rows(), B.n_rows());
          DEBUG_SAME_SIZE(A.n_cols(), C->n_rows());
          DEBUG_SAME_SIZE(C->n_cols(), B.n_cols());
          CppBlas<Precision>::gemm("T", "N",
                                   C->n_rows(), C->n_cols(),  A.n_rows(),
                                   alpha, A.ptr(), A.n_rows(), B.ptr(), B.n_rows(),
                                   beta, C->ptr(), C->n_rows());
        }
        else {
          DEBUG_SAME_SIZE(A.n_cols(), B.n_rows());
          DEBUG_SAME_SIZE(A.n_rows(), C->n_rows());
          DEBUG_SAME_SIZE(C->n_cols(), B.n_cols());
          CppBlas<Precision>::gemm("N", "N",
                                   C->n_rows(), C->n_cols(),  A.n_cols(),
                                   alpha, A.ptr(), A.n_rows(), B.ptr(), B.n_rows(),
                                   beta, C->ptr(), C->n_rows());
        }
      }
    }
};


/**
 * @brief Matrix Multiplication
 *       (\f$ C = A^{(T)} * B^{(T)}   \f$)
 *       A and B can be in normal or transpose mode
 * @code
 *   // Although Mul is a function, it is being used as a class constructor
 *   // This is because M, IsTransA have to be expliitly declared, while
 *   // the other parameters are automatically deduced by the function
 *   // arguments, The following example will shed some light
 *   template<MemoryAlloc M,
 *            TransMode IsTransA,
 *            TransMode IsTransB>
 *   class Mul {
 *   public:
 *    template<typename Precision,
 *             bool IsVector>
 *    Mul(const GenMatrix<Precision, false> &a,
 *        const GenMatrix<Precision, IsVector> &b,
 *        GenMatrix<Precision, IsVector> *c);
 *   };
 *   // example
 *   fl::la::GenMatrix<double> a;
 *   fl::la::GenMatrix<double> b;
 *   fl::la::GenMatrix<double> c;
 *   fl::la::Random(4, 3, &a);
 *   fl::la::Random(3, 4, &b);
 *   Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(a, b, &c);
 * @endcode
 *
 * @param M, It can be fl::la::Init, or fl::la::Overwrite, depending on whether
 *           we want the result to be initialized by Mul or not.
 * param IsTransA, It can be fl::la::NoTrans or fl::la::Trans, depending on
 *                 whether we want to use A's transpose or not.
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is automatically deduced by the funtion arguments
 * @param IsVector, bool parameter for backward compatibility,
 *                  automatically deduced from the function arguments
 *
 */
template < MemoryAlloc M,
TransMode IsTransA = NoTrans,
TransMode IsTransB = NoTrans >
class Mul {
  public:
    template < typename Precision,
    bool IsVector >
    Mul(const GenMatrix<Precision, false> &A,
        const GenMatrix<Precision, IsVector> &B,
        GenMatrix<Precision, IsVector> *C) {
      index_t n;
      if (IsTransA == true) {
        n = A.n_cols();
      }
      else {
        n = A.n_rows();
      }
      AllocationTrait<M>::Init(n, B.n_cols(), C);
      MulExpert<IsTransA>(
        Precision(1.0), A, B, Precision(0.0), C);
    }
    template<typename Precision>
    Mul(const GenMatrix<Precision, false> &A,
        const GenMatrix<Precision, false> &B,
        GenMatrix<Precision, false> *C) {
      index_t nr = 0;
      index_t nc = 0;
      if (IsTransA == true) {
        nr = A.n_cols();
      }
      else {
        nr = A.n_rows();
      }
      if (IsTransB == true) {
        nc = B.n_rows();
      }
      else {
        nc = B.n_cols();
      }

      AllocationTrait<M>::Init(nr, nc, C);
      MulExpert<IsTransA, IsTransB>(
        Precision(1.0), A, B, Precision(0.0), C);
    }
};




/* --- Wrappers for LAPACK --- */
/**
 * @brief Destructively computes an LU decomposition of a matrix.
 *
 * Stores L and U in the same matrix (the unitary diagonal of L is
 * implicit).
 * @param Precision, template parameter for the precision, currently supports
 *        float, double
 *
 * @param pivots a size min(M, N) array to store pivotes
 * @param A_in_LU_out an M-by-N matrix to be decomposed; overwritten
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<typename Precision>
inline success_t PLUExpert(f77_integer *pivots, GenMatrix<Precision, false> *A_in_LU_out) {
  f77_integer info;
  CppLapack<Precision>::getrf(A_in_LU_out->n_rows(),
                              A_in_LU_out->n_cols(),
                              A_in_LU_out->ptr(), A_in_LU_out->n_rows(), pivots, &info);
  return SUCCESS_FROM_LAPACK(info);
}
/**
 * Pivoted LU decomposition of a matrix.
 * @code
 *   // We use PLU as a class constructor because M has to be explicitly
 *   // defined, while Precision is automatically deduced by the
 *   // function arguments. The following example will shed some light
 *   template<tMemoryAlloc M>
 *   class PLU {
 *   public:
 *    template<MemoryAlloc M>
 *     PLU(const GenMatrix<Precision, false> &A,
 *         ArrayList<f77_integer> *pivots, GenMatrix<Precision, false> *L,
 *         GenMatrix<Precision, false> *U);
 *     success_t success;
 *   };
 *   fl::la::GenMatrix<double> a;
 *   fl::la::Random(5, 5, &a);
 *   fl::la::ArrayList<double> pivots;
 *   fl::la::GenMatrix<double> l;
 *   fl::la::GenMatrix<double> u;
 *   success_t success =
 *     fl::la::PLU<fl::la::Init>(a, &pivots, &l, &u).success;
 *   // The strange syntax ().success in the end was necessary
 *   // since the constructors don't return anything. If you
 *   // just want to do PLU without carring about the success
 *   // you don't have to put the ()..success in the end
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is automatically deduced by the function arguments
 * @param M, it can be fl::la::Init if PLU will initialize the result or
 *           fl::la::Overwrite if the soace has already been allocated
 * @param A the matrix to be decomposed
 * @param pivots a fresh array to be initialized to length min(M,N)
 *        and filled with the permutation pivots
 * @param L a fresh matrix to be initialized to size M-by-min(M,N)
 *        and filled with the lower triangular matrix
 * @param U a fresh matrix to be initialized to size min(M,N)-by-N
 *        and filled with the upper triangular matrix
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<MemoryAlloc M>
class PLU {
  public:
    template<typename Precision>
    PLU(const GenMatrix<Precision, false> &A,
        ArrayList<f77_integer> *pivots, GenMatrix<Precision, false> *L,
        GenMatrix<Precision, false> *U) {
      index_t m = A.n_rows();
      index_t n = A.n_cols();

      if (m > n) {
        pivots->Init(n);
        L->Copy(A);
        AllocationTrait<M>::Init(n, n, U);
        success = PLUExpert(pivots->begin(), L);

        if (!PASSED(success)) {
          return;
        }

        for (index_t j = 0; j < n; j++) {
          Precision *lcol = L->GetColumnPtr(j);
          Precision *ucol = U->GetColumnPtr(j);
          ::memcpy(ucol, lcol, (j + 1) * sizeof(Precision));
          ::memset(ucol + j + 1, 0, (n - j - 1) * sizeof(Precision));
          ::memset(lcol, 0, j * sizeof(Precision));
          lcol[j] = 1.0;
        }
      }
      else {
        pivots->Init(m);
        L->Init(m, m);
        AllocationTrait<M>::Init(A.n_rows(), A.n_cols(), U);
        U->CopyValues(A);
        success = PLUExpert(pivots->begin(), U);

        if (!PASSED(success)) {
          success;
        }

        for (index_t j = 0; j < m; j++) {
          Precision *lcol = L->GetColumnPtr(j);
          Precision *ucol = U->GetColumnPtr(j);

          ::memset(lcol, 0, j * sizeof(Precision));
          lcol[j] = 1.0;
          ::memcpy(lcol + j + 1, ucol + j + 1, (m - j - 1) * sizeof(Precision));
          ::memset(ucol + j + 1, 0, (m - j - 1) * sizeof(Precision));
        }
      }
      return ;
    }
    success_t success;
};
/**
 * @brief Destructively computes an inverse from a PLU decomposition.
 * @param Precision, template parameter for the precision, currently supports
 *        floati, double
 *
 * @param pivots the pivots array from PLU decomposition
 * @param LU_in_B_out the LU decomposition; overwritten with inverse
 * @return SUCCESS_PASS if invertible, SUCCESS_FAIL otherwise
 */
template<typename Precision>
inline success_t InverseExpert(f77_integer *pivots, GenMatrix<Precision, false> *LU_in_B_out) {
  f77_integer info;
  f77_integer n = LU_in_B_out->n_rows();
  f77_integer lwork = CppLapack<Precision>::getri_block_size * n;
  boost::scoped_array<Precision> work(new Precision[lwork]);
  DEBUG_MATSQUARE(*LU_in_B_out);
  CppLapack<Precision>::getri(n, LU_in_B_out->ptr(), n, pivots, work.get(), lwork, &info);
  return SUCCESS_FROM_LAPACK(info);
}
/**
 * @brief Inverts a matrix in place
 * (\f$A \gets A^{-1}\f$).
 *
 * @code
 *  template<typename Precision>
 *  success_t InverseExpert(GenMatrix<Precision, false> *A)
 *  // example
 *  fl::la::GenMatrix a;
 *  a.Init(2, 2);
 *  // assign a to [4 0; 0 0.5]
 *  a[0]=4; a[1]=0; a[2]=0; a[3]=0.5
 *  fl::la::Inverse(&a);
 *  //a is now [0.25 0; 0 2.0]
 * @endcode
  * @param Precision, template parameter for the precision, currently supports
 *        float double
 *
 * @param A an N-by-N matrix to invert
 * @return SUCCESS_PASS if invertible, SUCCESS_FAIL otherwise
 */
template<typename Precision>
success_t InverseExpert(GenMatrix<Precision, false> *A) {
  boost::scoped_array<f77_integer> pivots(new f77_integer[A->n_rows()]);

  success_t success = PLUExpert(pivots.get(), A);

  if (!PASSED(success)) {
    return success;
  }

  return InverseExpert(pivots.get(), A);
}


/**
 * @brief Set a matrix to the inverse of another matrix
 *        (\f$B \gets A^{-1}\f$).
 *
 * @code
 *   // Inverse is a function but we call it as a class constructor
 *   // This is because M has to be explicitly defined, while Precision
 *   // can be deduced by input. Since constructors don't return values
 *   // we store the success of the function in success member variable
 *   // The following result will show you how to use .
 *   template<typename Precision, MemoryAlloc M>
 *    class {
 *     public:
 *      template<typename Precision>
 *      Inverse(const GenMatrix<Precision, false> &A,
 *                    GenMatrix<Precision, false> *B);
 *      success_t success;
 *    };
 *    //example
 *    fl::la::GenMatrix a;
 *    a.Init(2, 2);
 *    // assign a to [4 0; 0 0.5]
 *    a[0]=4; a[1]=0; a[2]=0; a[3]=0.5;
 *    fl::la::GenMatrix<double> b;
 *    success_t success = fl::la::Inverse<fl::la::Init>(a, &b).success;
 *    //b is now [0.25 0; 0 2.0]
 *    // Another example showing the usage of fl::la::Overwrite
 *    fl::la::GenMatrix<double> c;
 *    c.Init(2, 2);
 *    success = fl::la::Inverse<fl::la::Overwrite>(a, &c);
 *   // c is now [0.25 0; 0 2.0]
 * @endcode
 *
 * @param M, It can be fl::la::Init if the result will be initialized by
 *           the function, or fl::la::Overwrite, if the result is initialized
 *           externally and the function just overwrites it
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is automatically deduced by the function arguments
 *
 * @param A an N-by-N matrix to invert
 * @param B an N-by-N matrix to store the results
 * @return SUCCESS_PASS if invertible, SUCCESS_FAIL otherwise
 */
template<MemoryAlloc M>
class Inverse {
  public:
    template<typename Precision>
    Inverse(const GenMatrix<Precision, false> &A,
            GenMatrix<Precision, false> *B) {
      boost::scoped_array<f77_integer> pivots(new f77_integer[A.n_rows()]);
      AllocationTrait<M>::Init(A.n_rows(), A.n_cols(), B);
      if (likely(A.ptr() != B->ptr())) {
        B->CopyValues(A);
      }
      success = PLUExpert(pivots.get(), B);

      if (!PASSED(success)) {
        return ;
      }
      success = InverseExpert(pivots.get(), B);
    }
    success_t success;
};

/**
 * @bried Returns the determinant of a matrix
 *       (\f$\det A\f$).
 *
 * @code
 *   template<typename Precision>
 *   long double Determinant(const GenMatrix<Precision, false> &A);
 *   // example
 *   fl::la::GenMatrix a;
 *   a.Init(2, 2);
 *   // assign a to [4 0; 0 0.5]
 *   a[0]=4; a[1]=0; a[2]=0; a[3]=0.5;
 *   double det = fl::la::Determinant(a);
 * // ... det is equal to 2.0
 * @endcode
  * @param Precision, template parameter for the precision, currently supports
 *        float double
 *
 * @param A the matrix to find the determinant of
 * @return the determinant; note long double for large exponents
 */
template<typename Precision>
long double Determinant(const GenMatrix<Precision, false> &A) {
  DEBUG_MATSQUARE(A);
  int n = A.n_rows();
  boost::scoped_array<f77_integer> pivots(new f77_integer[n]);
  GenMatrix<Precision> LU;

  LU.Copy(A);
  PLUExpert<Precision>(pivots.get(), &LU);

  long double det = 1.0;

  for (index_t i = 0; i < n; i++) {
    if (pivots[i] != i + 1) {
      // pivoting occured (note FORTRAN has 1-based indexing)
      det = -det;
    }
    det *= LU.get(i, i);
  }

  return det;
}

/**
 * @brief Returns the log-determinant of a matrix
 *        (\f$\ln |\det A|\f$).
 *
 * This is effectively log(fabs(Determinant(A))).
 * @code
 *   template<typename Precision>
 *   Precision DeterminantLog(const GenMatrix<Precision, false> &A,
 *                            int *sign_out);
 *   // example
 *   fl::la::GenMatrix a;
 *   a.Init(2, 2);
 *   // assign a to [4 0; 0 0.5]
 *   a[0]=4; a[1]=0; a[2]=0; a[3]=0.5;
 *   double det = fl::la::DeterminantLog(a, NULL);
 *   int sign;
 *   det = fl::la::DeterminantLog(a, &sign);
 *   // det is equal to log(2.0)
 *   // sign is 1;
 * @endcode
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is automatically deduced by the function arguments
 *
 * @param A the matrix to find the determinant of
 * @param sign_out set to -1, 1, or 0; pass NULL to disable
 * @return the log of the determinant or NaN if A is singular
 */
template<typename Precision>
Precision DeterminantLog(const GenMatrix<Precision, false> &A, int *sign_out) {
  DEBUG_MATSQUARE(A);
  int n = A.n_rows();
  boost::scoped_array<f77_integer> pivots(new f77_integer[n]);
  GenMatrix<Precision, false> LU;

  LU.Copy(A);
  PLUExpert<Precision>(pivots.get(), &LU);

  Precision log_det = 0.0;
  int sign_det = 1;

  for (index_t i = 0; i < n; i++) {
    if (pivots[i] != i + 1) {
      // pivoting occured (note FORTRAN has one-based indexing)
      sign_det = -sign_det;
    }

    Precision value = LU.get(i, i);

    if (value < 0) {
      sign_det = -sign_det;
      value = -value;
    }
    else if (!(value > 0)) {
      sign_det = 0;
      log_det = std::numeric_limits<Precision>::quiet_NaN();
      break;
    }

    log_det += log(value);
  }

  if (sign_out) {
    *sign_out = sign_det;
  }

  return log_det;
}

/**
 * @brief Destructively solves a system of linear equations (X st A * X = B).
 *
 * This computes the PLU factorization in A as a side effect, but
 * you are free to ignore it.
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is automatically deduced by the function arguments
 *
 * @param pivots a size N array to store pivots of LU decomposition
 * @param A_in_LU_out an N-by-N matrix multiplied by x; overwritten
 *        with its LU decomposition
 * @param k the number of columns on the right-hand side
 * @param B_in_X_out an N-by-K matrix ptr of desired products;
 *        overwritten with solutions (must not alias A_in_LU_out)
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<typename Precision>
inline success_t SolveExpert(
  f77_integer *pivots, GenMatrix<Precision, false> *A_in_LU_out,
  index_t k, Precision *B_in_X_out) {
  DEBUG_MATSQUARE(*A_in_LU_out);
  f77_integer info;
  f77_integer n = A_in_LU_out->n_rows();
  CppLapack<Precision>::gesv(n, k, A_in_LU_out->ptr(), n, pivots,
                             B_in_X_out, n, &info);
  return SUCCESS_FROM_LAPACK(info);
}
/**
 * @brief Inits a matrix to the solution of a system of linear equations
 *        (X st A * X = B).
 *
 * @code
 *   // Although Solve is a function it is being used as a function constructor
 *   // because M has to be explicitly defined, but the other templated arguments
 *   // can be deduced by the function arguments. The following example will
 *   // shed some light
 *   template<MemoryAlloc M>
 *   class Solve {
 *    public:
 *     template<typename Precision, bool IsVector>
 *     Solve(const GenMatrix<Precision, false> &A,
 *           const GenMatrix<Precision, IsVector> &B,
 *                 GenMatrix<Precision, IsVector> *X);
 *     success_t success;
 *   };
 *   // example
 *   fl::la::GenMatrix<double> a;
 *   a.Init(2, 2);
 *   fl::la::GenMatrix<double> b;
 *   b.Init(2, 2);
 *   // assign A to [1 3; 2 10]
 *   a[0]=1; a[1]=2; a[2]=3; a[3]=10;
 *   // assign B to [2 3; 8 10]
 *   b[0]=2; b[1]=3; b[2]=8; b[3]=10;
 *   fl::la::GenMatrix<double> x; // Not initialized
 *   success_t success = fl::la::Solve<fl::la::Init>(a, b, &x).success;
 *   // x is now [-1 0; 1 1]
 *   fl::la::GenMatrix<double> c;
 *   fl::la::Mul<fl::la::Init>(a, x, &c);
 *   // b and c should be equal (but for round-off)
 * @endcode
 *
 * @param M, it can be fl::la::Init, if the function will initialize the
 *           result, or fl::la::Overwrite if the result has already been
 *           initialized and it is going to be overwritten
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It can be deduced from the function arguments
 * @param A an N-by-N matrix multiplied by x
 * @param B a size N-by-K matrix of desired products
 * @param X a fresh matrix to be initialized to size N-by-K
 *        and filled with solutions
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<MemoryAlloc M>
class Solve {
  public:
    template<typename Precision, bool IsVector>
    Solve(const GenMatrix<Precision, false> &A,
          const GenMatrix<Precision, IsVector> &B,
          GenMatrix<Precision, IsVector> *X) {
      DEBUG_MATSQUARE(A);
      DEBUG_SAME_SIZE(A.n_rows(), B.n_rows());
      GenMatrix<Precision, false> tmp;
      index_t n = B.n_rows();
      boost::scoped_array<f77_integer> pivots(new f77_integer[n]);
      tmp.Copy(A);
      AllocationTrait<M>::Init(B.n_rows(), B.n_cols(), X);
      X->CopyValues(B);
      success = SolveExpert<Precision>(pivots.get(), &tmp, B.n_cols(), X->ptr());
    }
    success_t success;
};

/**
 * @brief Destructively performs a QR decomposition (A = Q * R).
 *        Factorizes a matrix as a rotation matrix (Q) times a reflection
 *        matrix (R); generalized for rectangular matrices.
 * @code
 *   template<typename Precision>
 *   success_t QRExpert(GenMatrix<Precision, false> *A_in_Q_out,
 *                      GenMatrix<Precision, false>  *R);
 *   // example
 *   // This is matrix a, but after QR decomposition, it will store q
 *   fl::la::GenMatrix<double> a_in_q_out;
 *   fl::la::Random(4, 4, &a_in_q_out);
 *   fl::la::GenMatrix<double> r;
 *   // it must be initialized;
 *   r.Init(4, 4);
 *   fl::la::QRExpert(&a_in_q_out, &r);
 * @endcode
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It can be deduced by the function arguments
 * @param A_in_Q_out an M-by-N matrix to factorize; overwritten with
 *        Q, an M-by-min(M,N) matrix (remaining columns are garbage,
 *        but are not removed from the matrix)
 * @param R a min(M,N)-by-N matrix to store results (must not be
 *        A_in_Q_out)
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<typename Precision>
success_t QRExpert(GenMatrix<Precision, false> *A_in_Q_out,
                   GenMatrix<Precision, false>  *R) {
  f77_integer info;
  f77_integer m = A_in_Q_out->n_rows();
  f77_integer n = A_in_Q_out->n_cols();
  f77_integer k = std::min(m, n);
  f77_integer lwork = n * CppLapack<Precision>::geqrf_dorgqr_block_size;
  boost::scoped_array<Precision> tau(new Precision[k + lwork]);
  Precision *work = tau.get() + k;

  // Obtain both Q and R in A_in_Q_out
  CppLapack<Precision>::geqrf(m, n, A_in_Q_out->ptr(), m,
                              tau.get(), work, lwork, &info);

  if (info != 0) {
    return SUCCESS_FROM_LAPACK(info);
  }

  // Extract R
  for (index_t j = 0; j < n; j++) {
    Precision *r_col = R->GetColumnPtr(j);
    Precision *q_col = A_in_Q_out->GetColumnPtr(j);
    int i = std::min(j + 1, index_t(k));
    ::memcpy(r_col, q_col, i * sizeof(Precision));
    ::memset(r_col + i, 0, (k - i) * sizeof(Precision));
  }

  // Fix Q
  CppLapack<Precision>::orgqr(m, k, k, A_in_Q_out->ptr(), m,
                              tau.get(), work, lwork, &info);

  return SUCCESS_FROM_LAPACK(info);

}
/**
 * @brief Init matrices to a QR decomposition (A = Q * R).
 *         Factorizes a matrix as a rotation matrix (Q) times a reflection
 *         matrix (R); generalized for rectangular matrices.
 *
 * @code
 *   // Although QR is a function we are using it as a class constructor. This
 *   // is becauce M has to be explicitly defined, while other template
 *   // arguments can be deduced from the function arguments. The following
 *   // result will make it more clear
 *   template<MemoryAlloc M>
 *   class QR {
 *    public:
 *     template<typename Precision>
 *     QR(const GenMatrix<Precision, false> &A,
 *              GenMatrix<Precision, false> *Q,
 *              GenMatrix<Precision, false> *R);
 *     success_t success;
 *   };
 *   fl::la::GenMatrix<double> a;
 *   // assign A to [3 5; 4 12]
 *   a.Init(2,2);
 *   a[0]=3; a[1]=4; a[2]=5; a[3]=12;
 *   fl::la::GenMatrix<double> q;
 *   fl::la::GenMatrix<double> r;
 *   success_t success = fl::la::QR<fl::la::Init>(a, &q, &r).success;
 *   // q is now [-0.6 -0.8; -0.8 0.6]
 *   // r is now [-5.0 -12.6; 0.0 3.2]
 *   fl::la::GenMatrix<double> b;
 *   fl::la::Mul<fl::la::Init>(q, r, &b)
 *   // a and b should be equal (but for round-off)
 * @endcode
  * @param Precision, template parameter for the precision, currently supports
 *        float double
 *
 * @param A an M-by-N matrix to factorize
 * @param Q a fresh matrix to be initialized to size M-by-min(M,N)
          and filled with the rotation matrix
 * @param R a fresh matrix to be initialized to size min(M,N)-by-N
          and filled with the reflection matrix
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<MemoryAlloc M>
class QR {
  public:
    template<typename Precision>
    QR(const GenMatrix<Precision, false> &A,
       GenMatrix<Precision, false> *Q,
       GenMatrix<Precision, false> *R) {
      index_t k = std::min(A.n_rows(), A.n_cols());
      AllocationTrait<M>::Init(A.n_rows(), A.n_cols(), Q);
      Q->CopyValues(A);
      AllocationTrait<M>::Init(k, A.n_cols(), R);
      success = QRExpert(Q, R);
      Q->ResizeNoalias(k);
    }
    success_t success;
};
/**
 * @brief Destructive Schur decomposition (A = Z * T * Z').
 *        This uses DGEES to find a Schur decomposition, but is also the best
 *         way to find just eigenvalues.
 * @code
 *   template<typename Precision>
 *   success_t SchurExpert(GenMatrix<Precision, false> *A_in_T_out,
 *   Precision *w_real, Precision *w_imag, Precision *Z);
 *   // example
 *
 *   // Here we store the initial matrix A, but after decomposition
 *   // T matrix will be stored there
 *   fl::la::GenMatrix<double> a_in_t_out;
 *   fl::la::Random(3, 3, &a_in_t_out);
 *   double w_real[3];
 *   double w_imag[3];
 *   double z[3][3];
 *   fl::la::ShurExpert(&a_in_t_out, w_real, w_imag, z);
 * @endcode
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is deduced by the function arguments
 *
 * @param A_in_T_out am N-by-N matrix to decompose; overwritten
 *        with the Schur form
 * @param w_real a length-N array to store real eigenvalue components
 * @param w_imag a length-N array to store imaginary components
 * @param Z an N-by-N matrix ptr to store the Schur vectors, or NULL
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<typename Precision>
success_t SchurExpert(GenMatrix<Precision, false> *A_in_T_out,
                      Precision *w_real, Precision *w_imag, Precision *Z) {
  DEBUG_MATSQUARE(*A_in_T_out);
  f77_integer info;
  f77_integer n = A_in_T_out->n_rows();
  f77_integer sdim;
  const char *job = Z ? "V" : "N";
  Precision d; // for querying optimal work size

  CppLapack<Precision>::gees(job, "N", NULL,
                             n, A_in_T_out->ptr(), n, &sdim, w_real, w_imag,
                             Z, n, &d, -1, NULL, &info);
  {
    f77_integer lwork = (f77_integer)d;
    boost::scoped_array<Precision> work(new Precision[lwork]);

    CppLapack<Precision>::gees(job, "N", NULL,
                               n, A_in_T_out->ptr(), n, &sdim, w_real, w_imag,
                               Z, n, work.get(), lwork, NULL, &info);
  }

  return SUCCESS_FROM_LAPACK(info);
}

/**
 * @brief Init matrices to a Schur decompoosition (A = Z * T * Z').
 * @code
 *   // Although Shur is a function, we use it as a class constructor,
 *   // because some of the template parameters have to be explicitly defined
 *   // while others can be deduced from context. The following example
 *   // will make it clear
 *
 *   template<MemoryAlloc M>
 *   class Schur {
 *    public:
 *     template<typename Precision, bool IsVector>
 *      Schur(const GenMatrix<Precision, false> &A,
 *                  GenMatrix<Precision, IsVector> *w_real,
 *                  GenMatrix<Precision, IsVector> *w_imag,
 *                  GenMatrix<Precision, false> *T,
 *                  GenMatrix<Precision, false> *Z);
 *      success_t success;
 *    };
 *    // example
 *    fl::la::GenMatrix<double> a;
 *    fl::la::Random(4, 4, &a);
 *    fl::la::GenMatrix<double, true> w_real;
 *    fl::la::GenMatrix<double, true> w_imag;
 *    fl::la::GenMatrix<double> t;
 *    fl::la::GenMatrix<double> z;
 *    success_t success = fl::la::Schur<fl::la::Init>(a,
 *        &w_real, &w_imag, &t, &z);
 * @endcode
 * @param M, if it is fl::la::Init, then the function initializes the results
 *           if it is fl::la::Overwrite, then the function just overwrites
 *           the already allocated results
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is deduced by the function arguments
 * @param IsVector, boolean template parameter for backward compatibility
 *                  with Vector. It is deduced by the function arguments
 * @param A an N-by-N matrix to decompose
 * @param w_real a fresh vector to be initialized to length N
 *        and filled with the real eigenvalue components
 * @param w_imag a fresh vector to be initialized to length N
 *        and filled with the imaginary components
 * @param T a fresh matrix to be initialized to size N-by-N
 *        and filled with the Schur form
 * @param Z a fresh matrix to be initialized to size N-by-N
 *        and filled with the Schur vectors
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<MemoryAlloc M>
class Schur {
  public:
    template<typename Precision, bool IsVector>
    Schur(const GenMatrix<Precision, false> &A,
          GenMatrix<Precision, IsVector> *w_real,
          GenMatrix<Precision, IsVector> *w_imag,
          GenMatrix<Precision, false> *T,
          GenMatrix<Precision, false> *Z) {
      index_t n = A.n_rows();
      AllocationTrait<M>::Init(A.n_rows(), A.n_cols(), T);
      T->CopyValues(A);
      AllocationTrait<M>::Init(n, 1, w_real);
      AllocationTrait<M>::Init(n, 1, w_imag);
      // w_real->Init(n);
      // w_imag->Init(n);
      AllocationTrait<M>::Init(n, n, Z);
      success = SchurExpert(T, w_real->ptr(), w_imag->ptr(), Z->ptr());
    }
    success_t success;
};
/**
 * @brief Destructive, unprocessed eigenvalue/vector decomposition.
 *      Real eigenvectors are stored in the columns of V, while imaginary
 *      eigenvectors occupy adjacent columns of V with conjugate pairs
 *      given by V(:,j) + i*V(:,j+1) and V(:,j) - i*V(:,j+1).
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It can be deduced by the function arguments
 * @param A_garbage an N-by-N matrix to be decomposed; overwritten
 *        with garbage
 * @param w_real a length-N array to store real eigenvalue components
 * @param w_imag a length-N array to store imaginary components
 * @param V_raw an N-by-N matrix ptr to store eigenvectors, or NULL
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<typename Precision>
success_t EigenExpert(GenMatrix<Precision, false> *A_garbage,
                      Precision *w_real, Precision *w_imag, Precision *V_raw) {
  DEBUG_MATSQUARE(*A_garbage);
  f77_integer info;
  f77_integer n = A_garbage->n_rows();
  const char *job = V_raw ? "V" : "N";
  Precision d; // for querying optimal work size

  CppLapack<Precision>::geev("N", job, n, A_garbage->ptr(), n,
                             w_real, w_imag, NULL, 1, V_raw, n, &d, -1, &info);
  {
    f77_integer lwork = (f77_integer)d;
    boost::scoped_array<Precision> work(new Precision[lwork]);

    CppLapack<Precision>::geev("N", job, n, A_garbage->ptr(), n,
                               w_real, w_imag, NULL, 1, V_raw, n, work.get(), lwork, &info);
  }

  return SUCCESS_FROM_LAPACK(info);
}

/**
 * @brief Finds the eigenvalues of a matrix.
 * @code
 *   // Although Eigenvalues is a function we use it as a clss constructor
 *   // because some of the template parameters have to be explicitly deifned
 *   // while others can be deduced. The following example will make it clear
 *
 *   template<MemoryAlloc M>
 *   class Eigenvalues {
 *    public:
 *     // returns imaginary and real eigenvalues
 *     template<typename Precision, bool IsVector>
 *     Eigenvalues(const GenMatrix<Precision, false> &A,
 *                       GenMatrix<Precision, IsVector> *w_real,
 *                       GenMatrix<Precision, IsVector> *w_imag);
 *     // returns only real eigenvalues
 *     template<typename Precision, bool IsVector>
 *     Eigenvalues(const GenMatrix<Precision, false> &A,
 *                       GenMatrix<Precision, IsVector> *w);
 *     success_t success;
 *   };
 *
 *  // example
 *  fl::la::GenMatrix<double> a;
 *  fl::la::Random(6, 6, &a);
 *  fl::la::GenMatrix<double> w_real;
 *  fl::la::GenMatrix<double> w_imag;
 *  success_t success=fl::la::Eigenvalues<fl::la::Init>(a,
 *                                &w_real, &w_imag).success;
 *  // we can just get the real eigenvalues
 *  // notice now that we use fl::la::Overwrite since w_real
 *  // has been initialized by the previous call of Eigenvalues
 *  success = fl::la::Eigenvalues<fl::la::Overwrite>(a, &w_real);
 * @endcode
 * @param M, if it is fl::la::Init, then the function initializes the results
 *           if it is fl::la::Overwrite, then the function just overwrites
 *           the already allocated results
 * @param Precision, template parameter for the precision, currently supports
 *        float double
 * @param A an N-by-N matrix to find eigenvalues for
 * @param w_real a fresh vector to be initialized to length N
 *        and filled with the real eigenvalue components
 * @param w_imag a fresh vector to be initialized to length N
 *        and filled with the imaginary components
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<MemoryAlloc M>
class Eigenvalues {
  public:
    template<typename Precision, bool IsVector>
    Eigenvalues(const GenMatrix<Precision, false> &A,
                GenMatrix<Precision, IsVector> *w_real,
                GenMatrix<Precision, IsVector> *w_imag) {
      DEBUG_MATSQUARE(A);
      int n = A.n_rows();
      AllocationTrait<M>::Init(n, 1, w_real);
      AllocationTrait<M>::Init(n, 1, w_imag);
      GenMatrix<Precision, false> tmp;
      tmp.Copy(A);
      success = SchurExpert<Precision>(&tmp, w_real->ptr(), w_imag->ptr(), NULL);
    };

    template<typename Precision, bool IsVector>
    Eigenvalues(const GenMatrix<Precision, false> &A,
                GenMatrix<Precision, IsVector> *w) {
      DEBUG_MATSQUARE(A);
      int n = A.n_rows();
      AllocationTrait<M>::Init(n, 1, w);
      boost::scoped_array<Precision> w_imag(new Precision[n]);

      GenMatrix<Precision, false> tmp;
      tmp.Copy(A);
      success = SchurExpert<Precision>(&tmp, w->ptr(), w_imag.get(), NULL);

      if (!PASSED(success)) {
        return ;
      }

      for (index_t j = 0; j < n; j++) {
        if (unlikely(w_imag[j] != 0.0)) {
          (*w)[j] = std::numeric_limits<Precision>::quiet_NaN();
        }
      }
    }
    success_t success;
};



/**
 * @brief Inits vectors and matrices to the eigenvalues/vectors of a matrix.
 *        Complex eigenvalues/vectors are processed into components rather
 *        than raw conjugate form.
 *
 * @code
 *   // although Eigenvectors is a function, we use it as a class constructor
 *   // because some of the template arguments have to be explicitly set,
 *   // while others are deduced by the contex. The following result will make
 *   // it clear.
 *   template<MemoryAlloc M>
 *   class Eigenvectors {
 *    public:
 *     template<typename Precision, bool IsVector>
 *     Eigenvectors(const GenMatrix<Precision, false> &A,
 *                        GenMatrix<Precision, IsVector> *w_real,
 *                        GenMatrix<Precision, IsVector> *w_imag,
 *                        GenMatrix<Precision, false> *V_real,
 *                        GenMatrix<Precision, false> *V_imag);
 *
 *     template<typename Precision, bool IsVector>
 *     Eigenvectors(const GenMatrix<Precision, false> &A,
 *                        GenMatrix<Precision, IsVector> *w,
 *                        GenMatrix<Precision, false> *V);
 *    success_t succes;
 *  };
 *  // example
 *  fl::la::GenMatrix<double> A;
 *  a.Init(2, 2);
 *  // assign A to [0 1; -1 0]
 *  a[0]=0; a[1]=-1;a[2]=1;a[3]=0;
 *  fl::la::GenMatrix<double> w_real;
 *  fl::la::GenMatrix<double> w_imag; // Not initialized
 *  fl::la::GenMatrix<double> v_real; // Not initialized
 *  fl::la::GenMatrix<double> v_imag; // Not initialized
 *  success_t success = fl::la::Eigenvectors<fl::la::Init>(a, &w_real,
 *      &w_imag, &v_real, &v_imag);
 *  // w_real and w_imag are [0 0] + i*[1 -1]
 *  // V_real and V_imag are [0.7 0.7; 0 0] + i*[0 0; 0.7 -0.7]
 * @endcode
 *
 * @param M, if it is fl::la::Init, then the function initializes the results
 *           if it is fl::la::Overwrite, then the function just overwrites
 *           the already allocated results
 * @param Precision, template parameter for the precision, currently supports
 *        floati, double. It is automatically deduced by the function arguments
 * @param IsVector, bool template argument for backward compatibility with
 *                  Vector. It is deduced by the function arguments
 * @param A an N-by-N matrix to be decomposed
 * @param w_real a fresh vector to be initialized to length N
 *        and filled with the real eigenvalue components
 * @param w_imag a fresh vector to be initialized to length N
 *        and filled with the imaginary eigenvalue components
 * @param V_real a fresh matrix to be initialized to size N-by-N
 *        and filled with the real eigenvector components
 * @param V_imag a fresh matrix to be initialized to size N-by-N
 *        and filled with the imaginary eigenvector components
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<MemoryAlloc M>
class Eigenvectors {
  public:
    template<typename Precision, bool IsVector>
    Eigenvectors(const GenMatrix<Precision, false> &A,
                 GenMatrix<Precision, IsVector> *w_real,
                 GenMatrix<Precision, IsVector> *w_imag,
                 GenMatrix<Precision, false> *V_real,
                 GenMatrix<Precision, false> *V_imag) {
      DEBUG_MATSQUARE(A);
      index_t n = A.n_rows();
      AllocationTrait<M>::Init(n, w_real);
      AllocationTrait<M>::Init(n, w_imag);
      AllocationTrait<M>::Init(n, n, V_real);
      AllocationTrait<M>::Init(n, n, V_imag);

      GenMatrix<Precision, false> tmp;
      tmp.Copy(A);
      success = EigenExpert(&tmp,
                            w_real->ptr(), w_imag->ptr(), V_real->ptr());

      if (!PASSED(success)) {
        return;
      }

      V_imag->SetZero();
      for (index_t j = 0; j < n; j++) {
        if (unlikely(w_imag->get(j) != 0.0)) {
          Precision *r_cur = V_real->GetColumnPtr(j);
          Precision *r_next = V_real->GetColumnPtr(j + 1);
          Precision *i_cur = V_imag->GetColumnPtr(j);
          Precision *i_next = V_imag->GetColumnPtr(j + 1);

          for (index_t i = 0; i < n; i++) {
            i_next[i] = -(i_cur[i] = r_next[i]);
            r_next[i] = r_cur[i];
          }

          j++; // skip paired column
        }
      }
    }

    template<typename Precision, bool IsVector>
    Eigenvectors(const GenMatrix<Precision, false> &A,
                 GenMatrix<Precision, IsVector> *w,
                 GenMatrix<Precision, false> *V) {
      DEBUG_MATSQUARE(A);
      index_t n = A.n_rows();
      AllocationTrait<M>::Init(n, w);
      boost::scoped_array<Precision> w_imag(new Precision[n]);
      AllocationTrait<M>::Init(n, n, V);

      GenMatrix<Precision, false> tmp;
      tmp.Copy(A);
      success = EigenExpert(&tmp, w->ptr(), w_imag.get(), V->ptr());

      if (!PASSED(success)) {
        return;
      }

      for (index_t j = 0; j < n; j++) {
        if (unlikely(w_imag[j] != 0.0)) {
          (*w)[j] = std::numeric_limits<Precision>::quiet_NaN();
        }
      }
      return;
    }

    success_t success;
};

/**
 * @brief Generalized eigenvalue/vector decomposition of two symmetric matrices.
 *        Compute all the eigenvalues, and optionally, the eigenvectors
 *        of a real generalized eigenproblem, of the form
 *        A*x=(lambda)*B*x, A*Bx=(lambda)*x, or B*A*x=(lambda)*x, where
 *        A and B are assumed to be symmetric and B is also positive definite.
 * @code
 *   template<typename Precision, bool IsVector>
 *   success_t GenEigenSymmetric(int itype,
 *                               GenMatrix<Precision, false> *A_eigenvec,
 *                               GenMatrix<Precision, false> *B_chol,
 *                               GenMatrix<Precision, IsVector> *w);
 *   // example
 *   fl::la::GenMatrix<double> a;
 *   fl::la::RandomSymmetric(5, 5, &a);
 *   fl::la::GenMatrix<double> b;
 *   fl::la::RandomSymmetric(5, 5, &b);
 *   fl::la::GenMatrix<double> w;
 *   w.Init(5, 1);
 *   fl::la:GenEigenSymmetric(1, &a, &b, &w);
 *
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It can be deduced by the function arguments
 * @param IsVector, bool template parameter, for backward compatibility
 *                  with vectors. It is deduced by context
 * @param itype an integer that specifies the problem type to be solved:
 *        = 1: A*x = (lambda)*B*x
 *        = 2: A*B*x = (lambda)*x
 *        = 3: B*A*x = (lambda)*x
 * @param A_eigenvec an N-by-N matrix to be decomposed; overwritten
 *        with the matrix of eigenvectors
 * @param B_chol an N-by-N matrix to be decomposed; overwritten
 *        with triangular factor U or L from the Cholesky factorization
 *        B=U**T*U or B=L*L**T.
 * @param w a length-N array of eigenvalues in ascending order.
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<typename Precision, bool IsVector>
success_t GenEigenSymmetric(int itype, GenMatrix<Precision, false> *A_eigenvec,
                            GenMatrix<Precision, false> *B_chol,
                            GenMatrix<Precision, IsVector>  *w) {
  DEBUG_MATSQUARE(*A_eigenvec);
  DEBUG_MATSQUARE(*B_chol);
  DEBUG_ASSERT(A_eigenvec->n_rows() == B_chol->n_rows());
  f77_integer itype_f77 = itype;
  f77_integer info;
  f77_integer n = A_eigenvec->n_rows();
  const char *job = "V"; // Compute eigenvalues and eigenvectors.
  Precision d; // for querying optimal work size

  CppLapack<Precision>::sygv(&itype_f77, job, "U", n,
                             A_eigenvec->ptr(), n, B_chol->ptr(), n, w->ptr(),
                             &d, -1, &info);
  {
    f77_integer lwork = (f77_integer)d;
    boost::scoped_array<Precision> work(new Precision[lwork]);

    CppLapack<Precision>::sygv(&itype_f77, job, "U", n, A_eigenvec->ptr(), n, B_chol->ptr(), n, w,
                               work.get(), lwork, &info);
  }

  return SUCCESS_FROM_LAPACK(info);

}

/**
 * @brief Generalized eigenvalue/vector decomposition of two nonsymmetric matrices.
 *        Compute all the eigenvalues, and optionally, the eigenvectors
 *        of a real generalized eigenproblem, of the form
 *        A*x=(lambda)*B*x, A*B*x=(lambda)*x, or B*A*x=(lambda)*x
 *        (alpha_real(j) + alpha_imag(j)*i)/beta(j), j=1,...,N, will be the
 *        generalized eigenvalues.
 * Note: the quotients alpha_real(j)/beta(j) and alpha_imag(j)/beta(j) may
 * easily over- or underflow, and beta(j) may even be zero.  Thus, the
 * user should avoid naively computing the ratio alpha/beta.  However,
 * alpha_real and alpha_imag will be always less than and usually comparable
 * with norm(A) in magnitude, and beta always less than and usually
 * comparable with norm(B).
 *
 * Real eigenvectors are stored in the columns of V, while imaginary
 * eigenvectors occupy adjacent columns of V with conjugate pairs
 * given by V(:,j) + i*V(:,j+1) and V(:,j) - i*V(:,j+1).
 *
 * @code
 *   template<typename Precision, bool IsVector>
 *   success_t GenEigenNonSymmetric(GenMatrix<Precision, false> *A_garbage,
 *                                  GenMatrix<Precision, false> *B_garbage,
 *                                  GenMatrix<Precision, IsVector> *alpha_real,
 *                                  GenMatrix<Precision, IsVector> *alpha_imag,
 *                                  GenMatrix<Precision, IsVector> *beta,
 *                                  GenMatrix<Precision, IsVector> *V_raw);
 *  // example
 *  fl:la::GenMatrix<double> a;
 *  fl::la::Random(4, 4, &a);
 *  fl::la::GenMatrix<double> b;
 *  fl::la::Random(4, 4, &b);
 *  fl::la::GenMatrix<double> alpha_real;
 *  alpha_real.Init(4, 1);
 *  fl::la::GenMatrix<double> alpha_imag;
 *  alpha_imag.Init(4, 1);
 *  fl::la::GenMatrix<double> beta;
 *  beta.Init(4, 1);
 *  fl::la::GenMatrix<double> v_raw;
 *  v_raw.Init(4, 1);
 *  fl::la::GenEigenNonsymmetric(&a, &b, &alpha_real,
 *         &alpha_image, &beta, &v_raw);
 * @endcode
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is deduced by the function arguments
 * @param  IsVector, boolean template parameter, for backward compatibility
 *                   with Vector. It is deduced by the funciton arguments
 * @param A_garbage an N-by-N matrix to be decomposed; overwritten
 *        with garbage
 * @param B_garbage an N-by-N matrix to be decomposed; overwritten
 *        with garbage
 * @param alpha_real a length-N array
 * @param alpha_imag a length-N array
 * @param beta a length-N array
 * @param V_raw an N-by-N matrix ptr to store eigenvectors, or NULL
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<typename Precision, bool IsVector>
success_t GenEigenNonSymmetric(GenMatrix<Precision, false> *A_garbage,
                               GenMatrix<Precision, false> *B_garbage,
                               GenMatrix<Precision, IsVector> *alpha_real,
                               GenMatrix<Precision, IsVector> *alpha_imag,
                               GenMatrix<Precision, IsVector> *beta,
                               GenMatrix<Precision, IsVector> *V_raw) {
  DEBUG_MATSQUARE(*A_garbage);
  DEBUG_MATSQUARE(*B_garbage);
  DEBUG_ASSERT(A_garbage->n_rows() == B_garbage->n_rows());
  f77_integer info;
  f77_integer n = A_garbage->n_rows();
  const char *job = V_raw ? "V" : "N";
  Precision d; // for querying optimal work size

  CppLapack<Precision>::gegv("N", job, n, A_garbage->ptr(), n, B_garbage->ptr(), n,
                             alpha_real->ptr(), alpha_imag->ptr(),
                             beta->ptr(), NULL, 1, V_raw->ptr(), n, &d, -1, &info);
  {
    f77_integer lwork = (f77_integer)d;
    boost::scoped_array<Precision> work(new Precision[lwork]);

    CppLapack<Precision>::gegv("N", job, n, A_garbage->ptr(), n, B_garbage->ptr(), n,
                               alpha_real->ptr(), alpha_imag->ptr(),
                               beta->ptr(), NULL, 1, V_raw->ptr(), n, work.get(), lwork, &info);
  }

  return SUCCESS_FROM_LAPACK(info);

}

/**
 * @brief Destructive SVD (A = U * S * VT).
 * Finding U and VT is optional (just pass NULL), but you must solve
 * either both or neither.
 *  template<typename Precision>
 * @code
 *   template<typename Precision>
 *   success_t SVDExpert(GenMatrix<Precision, false>* A_garbage,
 *                       Precision *s,
 *                       Precision *U,
 *                       Precision *VT);
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float double
 *
 * @param A_garbage an M-by-N matrix to be decomposed; overwritten
 *        with garbage
 * @param s a length-min(M,N) array to store the singluar values
 * @param U an M-by-min(M,N) matrix ptr to store left singular
 *        vectors, or NULL for neither U nor VT
 * @param VT a min(M,N)-by-N matrix ptr to store right singular
 *        vectors, or NULL for neither U nor VT
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<typename Precision>
success_t SVDExpert(GenMatrix<Precision, false>* A_garbage,
                    Precision *s,
                    Precision *U,
                    Precision *VT) {
  DEBUG_ASSERT_MSG((U == NULL) == (VT == NULL),
                   "You must fill both U and VT or neither.");
  f77_integer info;
  f77_integer m = A_garbage->n_rows();
  f77_integer n = A_garbage->n_cols();
  f77_integer k = std::min(m, n);
  boost::scoped_array<f77_integer> iwork(new f77_integer[8 * k]);
  const char *job = U ? "S" : "N";
  Precision d; // for querying optimal work size

  CppLapack<Precision>::gesdd(job, m, n, A_garbage->ptr(), m,
                              s, U, m, VT, k, &d, -1, iwork.get(), &info);
  {
    f77_integer lwork = (f77_integer)d;
    // work for DGESDD can be large, we really do need to malloc it
    boost::scoped_array<Precision> work(new Precision[lwork]);

    CppLapack<Precision>::gesdd(job, m, n, A_garbage->ptr(), m,
                                s, U, m, VT, k, work.get(), lwork, iwork.get(), &info);
  }

  return SUCCESS_FROM_LAPACK(info);

}

/**
 * @brief Inits a vector and matrices to a singular value decomposition
 *        (A = U * S * VT).
 *
 * @code
 *   // Although SVD is a function, we use it as a class constructor, because
 *   // some of the template arguments have to be be explicitly defined
 *   // while others can be deduced. The following example will make it
 *   // more clear.
 *   template<MemoryAlloc M>
 *   class SVD {
 *    public:
 *     template<typename Precision, bool IsVector>
 *     SVD(const GenMatrix<Precision, false> &A,
 *               GenMatrix<Precision, IsVector> *s);
 *     template<typename Precision, bool IsVector>
 *     SVD(const GenMatrix<Precision, false> &A,
 *               GenMatrix<Precision, IsVector> *s,
 *               GenMatrix<Precision, false> *U,
 *               GenMatrix<Precision, false> *VT);
 *     success_t success;
 *   };
 *   // example
 *   fl::la::GenMatrix<double> a;
 *   // assign A to [0 1; -1 0]
 *   a.Init(2, 2);
 *   a[0]=0; a[1]=-1; a[2]=1; a[3]=0;
 *   fl::la::GenMatrix<double> s; // Not initialized
 *   fl::la::GenMatrix<double> U, VT; // Not initialized
 *   success_t success = fl::la::SVD<fl::la::Init>(a, &s, &U, &VT).success;
 *   // s is now [1 1]
 *   // U is now [0 1; 1 0]
 *   //  V is now [-1 0; 0 1]
 *   fl::la::GenMatrix<double> S;
 *   S.Init(s.length(), s.length());
 *   S.SetDiagonal(s);
 *   fl::la::GenMatrix<double> tmp, result;
 *   fl::la::Mul<fl::la::Init>(U, S, &tmp);
 *   fl::la::MulInit(tmp, VT, &result);
 *   // A and result should be equal (but for round-off)
 * @endcode
 * @param M, if it is fl::la::Init, then the function initializes the results
 *           if it is fl::la::Overwrite, then the function just overwrites
 *           the already allocated results
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is automatically deduced by the function arguments
 * @param IsVector, bool template argument for backward compatibility with
 *                  Vector. It is deduced by the function arguments
 * @param A an M-by-N matrix to be decomposed
 * @param s a fresh vector to be initialized to length min(M,N)
 *        and filled with the singular values
 * @param U a fresh matrix to be initialized to size M-by-min(M,N)
 *        and filled with the left singular vectors
 * @param VT a fresh matrix to be initialized to size min(M,N)-by-N
 *        and filled with the right singular vectors
 * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
 */
template<MemoryAlloc M>
class SVD {
  public:
    template<typename Precision, bool IsVector>
    SVD(const GenMatrix<Precision, false> &A,
        GenMatrix<Precision, IsVector> *s) {
      AllocationTrait<M>::Init(std::min(A.n_rows(), A.n_cols()), s);
      GenMatrix<Precision, false> tmp;
      tmp.Copy(A);
      success = SVDExpert<Precision>(&tmp, s->ptr(), NULL, NULL);
    }
    template<typename Precision, bool IsVector>
    SVD(const GenMatrix<Precision, false> &A,
        GenMatrix<Precision, IsVector> *s,
        GenMatrix<Precision, false> *U,
        GenMatrix<Precision, false> *VT) {
      index_t k = std::min(A.n_rows(), A.n_cols());
      AllocationTrait<M>::Init(k, s);
      AllocationTrait<M>::Init(A.n_rows(), k, U);
      AllocationTrait<M>::Init(k, A.n_cols(), VT);
      GenMatrix<Precision, false> tmp;
      tmp.Copy(A);
      success =  SVDExpert(&tmp, s->ptr(), U->ptr(), VT->ptr());
    }
    success_t success;
};

/**
 * @brief Destructively computes the Cholesky factorization (A = U' * U).
 * @code
 *   template<typename Precision>
 *   success_t CholeskyExpert(GenMatrix<Precision, false> *A_in_U_out);
 *   // example
 *   fl::la::GenMatrix<double> a_in_u_out;
 *   a.Init(2, 2);
 *   a[0]=4; a[1]=1; a[2]=1; a[3]=2;
 *   fl::la::CholeskyExpert(&a_in_u_out);
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is automatically deduced by the function arguments
 * @param A_in_U_out an N-by-N matrix to factorize; overwritten
 *        with result
 * @return SUCCESS_PASS if the matrix is symmetric positive definite,
 *         SUCCESS_FAIL otherwise
 */
template<typename Precision>
success_t CholeskyExpert(GenMatrix<Precision, false> *A_in_U_out) {
  DEBUG_MATSQUARE(*A_in_U_out);
  f77_integer info;
  f77_integer n = A_in_U_out->n_rows();

  CppLapack<Precision>::potrf("U", n, A_in_U_out->ptr(), n, &info);

  /* set the garbage part of the matrix to 0. */
  for (f77_integer j = 0; j < n; j++) {
    ::memset(A_in_U_out->GetColumnPtr(j) + j + 1, 0, (n - j - 1) * sizeof(Precision));
  }

  return SUCCESS_FROM_LAPACK(info);
}


/**
 * @brief Inits a matrix to the Cholesky factorization (A = U' * U).
 *
 * @code
 *   // Although Cholesky is a function, we use it here as a class constructor
 *   // because some of the template arguments have to be explicitly defined
 *   // while others can be deduced. The following example will make it more clear
 *
 *   template<MemoryAlloc M>
 *   class Cholesky {
 *    public:
 *     template<typename Precision>
 *     Cholesky(const GenMatrix<Precision, false> &A,
 *                    GenMatrix<Precision, false> *U);
 *     success_t success;
 *   };
 *   // example
 *   fl::la::GenMatrix<double> a;
 *   //  assign a to [1 1; 1 2]
 *   a.Init(2, 2);
 *   a[0]=1; a[1]=1; a[2]=1; a[3]=2;
 *   fl::la::GenMatrix<double> u; // Not initialized
 *   success_t success = fl::la::Cholesky<fl::la::Init>(a, &u).success;
 *   // u is now [1 1; 0 1]
 *   fl::la::GenMatrix<double> result;
 *   fl::la::Mul<fl::la::Init, fl::la::Trans, fl::la::NoTrans>(U, U, result);
 *   // A and result should be equal (but for round-off)
 * @endcode
 * @param M, if it is fl::la::Init, then the function initializes the results
 *           if it is fl::la::Overwrite, then the function just overwrites
 *           the already allocated results
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is automatically deduced by the function arguments
 *
 * @param A an N-by-N matrix to factorize
 * @param U a fresh matrix to be initialized to size N-by-N
 *        and filled with the factorization
 * @return SUCCESS_PASS if the matrix is symmetric positive definite,
 *         SUCCESS_FAIL otherwise
 */
template<MemoryAlloc M>
class Cholesky {
  public:
    template<typename Precision>
    Cholesky(const GenMatrix<Precision, false> &A,
             GenMatrix<Precision, false> *U) {
      AllocationTrait<M>::Init(A.n_rows(), A.n_cols(), U);
      U->CopyValues(A);
      success = CholeskyExpert(U);
    }
    success_t success;
};

/*
 * TODO:
 *   symmetric eigenvalue
 *   http://www.netlib.org/lapack/lug/node71.html
 *   dgels for least-squares problems
 *   dgesv for linear equations
 */
}; //la namespace
};
#endif
