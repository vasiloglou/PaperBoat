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
 * @file fl_math.h
 * @author Nikolaos Vasiloglou
 *
 */

#ifndef FASTLIB_MATH_MATH_LIB_H
#define FASTLIB_MATH_MATH_LIB_H

#include "boost/type_traits/is_same.hpp"
#include <boost/math/special_functions/fpclassify.hpp>
#include "fastlib/base/base.h"
#include "gen_range.h"
#include <cmath>
#include <cstdlib>
#include <vector>

namespace fl {
/**
 * @brief namespace math contains basic mathematical operations
 *
 */
namespace math {
template<typename Precision>
class Const {
  public:
    /** The square root of 2. */
    static const Precision SQRT2;
    /** Base of the natural logarithm. */
    static const Precision E;
    /** Log base 2 of E. */
    static const Precision LOG2_E;
    /** Log base 10 of E. */
    static const Precision LOG10_E;
    /** Natural log of 2. */
    static const Precision LN_2;
    /** Natural log of 10. */
    static const Precision LN_10;
    /** The ratio of the circumference of a circle to its diameter. */
    static const Precision PI;
    /** The ratio of the circumference of a circle to its radius. */
    static const Precision PI_2;
};
template<typename Precision>
const Precision Const<Precision>::SQRT2 =  1.41421356237309504880;

template<typename Precision>
const Precision Const<Precision>::E = 2.7182818284590452354;

template<typename Precision>
const Precision Const<Precision>::LOG2_E = 1.4426950408889634074;

template<typename Precision>
const Precision Const<Precision>::LOG10_E = 0.43429448190325182765;

template<typename Precision>
const Precision Const<Precision>::LN_2 = 0.69314718055994530942;

template<typename Precision>
const Precision Const<Precision>::LN_10 = 2.30258509299404568402;

template<typename Precision>
const Precision Const<Precision>::PI = 3.141592653589793238462643383279;

template<typename Precision>
const Precision Const<Precision>::PI_2 = 1.57079632679489661923;

/** Squares a number. */
template<typename T>
inline T Sqr(T v);
/**
 * Rounds a double-precision to an integer, casting it too.
 */
inline int64 RoundInt(double d);

/**
 * Forces a number to be non-negative, turning negative numbers into zero.
 *
 * Avoids branching costs (yes, we've discovered measurable improvements).
 */
template<typename Precision>
inline Precision ClampNonNegative(Precision d);

/**
 * Forces a number to be non-positive, turning positive numbers into zero.
 *
 * Avoids branching costs (yes, we've discovered measurable improvements).
 */
template<typename Precision>
inline double ClampNonPositive(Precision d);

/**
 * Clips a number between a particular range.
 *
 * @param value the number to clip
 * @param range_min the first of the range
 * @param range_max the last of the range
 * @return max(range_min, min(range_max, d))
 */
template<typename Precision>
inline Precision ClampRange(Precision value, Precision range_min, Precision range_max);

/**
 * Generates a uniform random number between 0 and 1.
 */
template<typename Precision>
inline Precision Random();
/**
 * Generates a uniform random number in the specified range.
 */

inline double Random(double lo, double hi);

inline float Random(float lo, float hi);

inline int32 Random(int32 lo, int32 hi);

inline int64 Random(int64 lo, int64 hi);

inline uint32 Random(uint32 lo, uint32 hi);

inline uint64 Random(uint64 lo, uint64 hi);

inline unsigned char Random(unsigned char lo, unsigned char hi);

inline signed char Random(signed char lo, signed char hi);

/**
 * Generates a uniform random integer.
 */
inline int Random(int hi_exclusive);
/**
* Generates a uniform random integer.
*/
inline int Random(int lo, int hi_exclusive);

/**
 * Generate a normal (gaussian) random double
 */
inline double RandomNormal();
inline double RandomNormal(double mean, double variance);
}; // namespace math
}; // namespace fl

namespace fl {
namespace math {
/**
 * Calculates a relatively small power using template metaprogramming.
 *
 * This allows a numerator and denominator.  In the case where the
 * numerator and denominator are equal, this will not do anything, or in
 * the case where the denominator is one.
 */
template<typename Precision, int t_numerator, int t_denominator>
inline Precision Pow(Precision d);
/**
 * Calculates a small power of the absolute value of a number
 * using template metaprogramming.
 *
 * This allows a numerator and denominator.  In the case where the
 * numerator and denominator are equal, this will not do anything, or in
 * the case where the denominator is one.  For even powers, this will
 * avoid calling the absolute value function.
 */
template<typename Precision, int t_numerator, int t_denominator>
inline Precision PowAbs(Precision d);

/**
 * A value which is the min or max of multiple other values.
 *
 * Comes with a highly optimized version of x = max(x, y).
 *
 * The template argument should be something like double, with greater-than,
 * less-than, and equals operators.
 */
template<typename TValue>
class MinMaxVal;
/**
 * Computes the hyper-volume of a hyper-sphere of dimension d.
 *
 * @param r the radius of the hyper-sphere
 * @param d the number of dimensions
 */
template<typename Precision>
Precision SphereVolume(Precision r, int d);

template<typename Precision, bool USE_NORMALIZATION>
class GaussianKernel;


/**
 * Standard multivariate Gaussian kernel.
 *
 */
template<typename Precision, bool USE_NORMALIZATION>
class GaussianStarKernel;

/**
 * Multivariate Epanechnikov kernel.
 *
 * To use, first get an unnormalized density, and divide by the
 * normalizeation factor.
 */
template<typename Precision>
class EpanKernel;

/**
 * Computes the factorial of an integer.
 */
template<typename Precision>
Precision Factorial(int d);

/**
 * Computes the binomial coefficient, n choose k for nonnegative integers
 * n and k
 *
 * @param n the first nonnegative integer argument
 * @param k the second nonnegative integer argument
 * @return the binomial coefficient n choose k
 */

template<typename Precision>
Precision BinomialCoefficient(int n, int k);

/**
 * Creates an identity permutation where the element i equals i.
 *
 * Low-level pointer version -- preferably use the @c std::vector
 * version instead.
 *
 * For instance, result[0] == 0, result[1] == 1, result[2] == 2, etc.
 *
 * @param size the number of elements in the permutation
 * @param array a place to store the permutation
 */
void MakeIdentityPermutation(index_t size, index_t *array);

/**
 * Creates an identity permutation where the element i equals i.
 *
 * For instance, result[0] == 0, result[1] == 1, result[2] == 2, etc.
 *
 * @param size the size to initialize the result to
 * @param result will be initialized to the identity permutation
 */
inline void MakeIdentityPermutation(index_t size, std::vector<index_t> *result);

/**
 * Creates a random permutation and stores it in an existing C array
 * (power user version).
 *
 * The random permutation is over the integers 0 through size - 1.
 *
 * @param size the number of elements
 * @param array the array to store a permutation in
 */
void MakeRandomPermutation(index_t size, index_t *array);

/**
 * Creates a random permutation over integers 0 throush size - 1.
 *
 * @param size the number of elements
 * @param result will be initialized to a permutation array
 */
inline void MakeRandomPermutation(
  index_t size, std::vector<index_t> *result);

/**
 * Inverts or transposes an existing permutation.
 */
void MakeInversePermutation(index_t size,
                            const index_t *original, index_t *reverse);
/**
 * Inverts or transposes an existing permutation.
 */
inline void MakeInversePermutation(
  const std::vector<index_t>& original, std::vector<index_t> *reverse);

template<typename TAnyIntegerType>
inline bool IsPowerTwo(TAnyIntegerType i);

/**
 * Finds the log base 2 of an integer.
 *
 * This integer must absolutely be a power of 2.
 */
inline unsigned int IntLog2(unsigned int i);

}; // namespace math
}; // namespace fl

#include "fl_math_impl.h"

#endif
