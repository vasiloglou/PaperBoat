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

#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"
#include "boost/random/uniform_real.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/variate_generator.hpp"

namespace fl {
  extern boost::mt19937 mtn19937_gen;
namespace math {

/** Squares a number. */
template<typename T>
inline T Sqr(T v) {
  return v * v;
}

/**
 * Rounds a double-precision to an integer, casting it too.
 */
inline int64 RoundInt(double d) {
//  return int64(nearbyint(d));
  return int64(d);
}

/**
 * Forces a number to be non-negative, turning negative numbers into zero.
 *
 * Avoids branching costs (yes, we've discovered measurable improvements).
 */
template<typename Precision>
inline Precision ClampNonNegative(Precision d) {
  return (d + fabs(d)) / 2;
}

/**
 * Forces a number to be non-positive, turning positive numbers into zero.
 *
 * Avoids branching costs (yes, we've discovered measurable improvements).
 */
template<typename Precision>
inline double ClampNonPositive(Precision d) {
  return (d - fabs(d)) / 2;
}

/**
 * Clips a number between a particular range.
 *
 * @param value the number to clip
 * @param range_min the first of the range
 * @param range_max the last of the range
 * @return max(range_min, min(range_max, d))
 */
template<typename Precision>
inline Precision ClampRange(Precision value, Precision range_min, Precision range_max) {
  if (unlikely(value <= range_min)) {
    return range_min;
  }
  else if (unlikely(value >= range_max)) {
    return range_max;
  }
  else {
    return value;
  }
}

/**
 * Generates a uniform random number between 0 and 1.
 */
template<typename Precision>
inline Precision Random() {
  boost::uniform_real<Precision> dist(0.0, 1.0);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<Precision> > die(mtn19937_gen, dist);
  return die();
}

/**
 * Generates a uniform random number in the specified range.
 */

inline double Random(double lo, double hi) {
  boost::uniform_real<double> dist(lo, hi);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<double> > die(mtn19937_gen, dist);
  return die();
}

inline float Random(float lo, float hi) {
  boost::uniform_real<float> dist(lo, hi);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > die(mtn19937_gen, dist);
  return die();
}

inline int32 Random(int32 lo, int32 hi) {
  boost::uniform_int<int32> dist(lo, hi);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<int32> > die(mtn19937_gen, dist);
  return die();
}

inline int64 Random(int64 lo, int64 hi) {
  boost::uniform_int<int64> dist(lo, hi);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<int64> > die(mtn19937_gen, dist);
  return die();
}

inline uint32 Random(uint32 lo, uint32 hi) {
  boost::uniform_int<uint32> dist(lo, hi);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<uint32> > die(mtn19937_gen, dist);
  return die();
}

inline uint64 Random(uint64 lo, uint64 hi) {
  boost::uniform_int<uint64> dist(lo, hi);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<uint64> > die(mtn19937_gen, dist);
  return die();
}


inline unsigned char Random(unsigned char lo, unsigned char hi) {
  boost::uniform_int<unsigned char> dist(lo, hi);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<unsigned char> > die(mtn19937_gen, dist);
  return die();
}

inline signed char Random(signed char lo, signed char hi) {
  boost::uniform_int<signed char> dist(lo, hi);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<signed char> > die(mtn19937_gen, dist);
  return die();
}

inline double RandomNormal() {
  boost::normal_distribution<double> dist(0, 1);
  boost::variate_generator<boost::mt19937&, 
    boost::normal_distribution<double> > die(mtn19937_gen, dist);
  return die();

}

inline double RandomNormal(double mean, double variance) {
  boost::normal_distribution<double> dist(mean, variance);
  boost::variate_generator<boost::mt19937&, 
    boost::normal_distribution<double> > die(mtn19937_gen, dist);
  return die();

}

///**
// * Generates a uniform random integer.
// */
//inline int Random(int hi_exclusive) {
//  return rand() % hi_exclusive;
//}
///**
// * Generates a uniform random integer.
// */
//inline int Random(int lo, int hi_exclusive) {
//  return (rand() % (hi_exclusive - lo)) + lo;
//}
} // namespace math
} // namespace fl

namespace math__private {
template < typename Precision, int t_numerator, int t_denominator = 1 >
struct ZPowImpl {
  static Precision Calculate(Precision d) {
    return pow(d, t_numerator * 1.0 / t_denominator);
  }
};

template<typename Precision, int t_equal>
struct ZPowImpl<Precision, t_equal, t_equal> {
  static Precision Calculate(Precision d) {
    return d;
  }
};

template<typename Precision>
struct ZPowImpl<Precision, 1, 1> {
  static Precision Calculate(Precision d) {
    return d;
  }
};

template<typename Precision>
struct ZPowImpl<Precision, 1, 2> {
  static Precision Calculate(Precision d) {
    return sqrt(d);
  }
};

template<typename Precision>
struct ZPowImpl<Precision, 1, 3> {
  static Precision Calculate(Precision d) {
    return cbrt(d);
  }
};

template<typename Precision, int t_denominator>
struct ZPowImpl<Precision, 0, t_denominator> {
  static Precision Calculate(Precision d) {
    return 1;
  }
};

template<typename Precision, int t_numerator>
struct ZPowImpl<Precision, t_numerator, 1> {
  static Precision Calculate(Precision d) {
    return ZPowImpl < Precision, t_numerator - 1, 1 >::Calculate(d) * d;
  }
};

// absolute-value-power: have special implementations for even powers
// TODO: do this for all even integer powers

template<typename Precision, int t_numerator, int t_denominator, bool is_even>
struct ZPowAbsImpl;

template<typename Precision, int t_numerator, int t_denominator>
struct ZPowAbsImpl<Precision, t_numerator, t_denominator, false> {
  static Precision Calculate(Precision d) {
    return ZPowImpl<Precision, t_numerator, t_denominator>::
           Calculate(fabs(d));
  }
};

template<typename Precision, int t_numerator, int t_denominator>
struct ZPowAbsImpl<Precision, t_numerator, t_denominator, true> {
  static Precision Calculate(Precision d) {
    return ZPowImpl<Precision, t_numerator, t_denominator>::Calculate(d);
  }
};
}

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
inline Precision Pow(Precision d) {
  return (Precision)math__private::ZPowImpl<Precision, t_numerator, t_denominator>::
         Calculate(d);
}

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
inline Precision PowAbs(Precision d) {
  // we specify whether it's an even function -- if so, we can sometimes
  // avoid the absolute value sign
  return (Precision)math__private::ZPowAbsImpl < Precision, t_numerator, t_denominator,
         (t_numerator % t_denominator == 0) &&
         ((t_numerator / t_denominator) % 2 == 0) >::Calculate(fabs(d));
}


/**
 * A value which is the min or max of multiple other values.
 *
 * Comes with a highly optimized version of x = max(x, y).
 *
 * The template argument should be something like double, with greater-than,
 * less-than, and equals operators.
 */
template<typename TValue>
class MinMaxVal {
  public:
    typedef TValue Value;
  public:
    /** The underlying value. */
    Value val;

  public:
    /**
     * Converts implicitly to the value.
     */
    operator Value() const {
      return val;
    }

    /**
     * Sets the value.
     */
    const Value& operator = (Value val_in) {
      return (val = val_in);
    }

    /**
     * Efficiently performs this->val = min(this->val, incoming_val).
     *
     * The expectation is that it is higly unlikely for the incoming
     * value to be the new minimum.
     */
    void MinWith(Value incoming_val) {
      if (unlikely(incoming_val < val)) {
        val = incoming_val;
      }
    }

    /**
     * Efficiently performs this->val = min(this->val, incoming_val).
     *
     * The expectation is that it is higly unlikely for the incoming
     * value to be the new maximum.
     */
    void MaxWith(Value incoming_val) {
      if (unlikely(incoming_val > val)) {
        val = incoming_val;
      }
    }
};
/**
  * Computes the hyper-volume of a hyper-sphere of dimension d.
  *
  * @param r the radius of the hyper-sphere
  * @param d the number of dimensions
  */
template<typename Precision>
Precision SphereVolume(Precision r, int d) {
  int n = d / 2;
  Precision val;

  BOOST_ASSERT(d >= 0);

  if (d % 2 == 0) {
    val = pow(r * sqrt(fl::math::Const<Precision>::PI), d) / Factorial<Precision>(n);
  }
  else {
    val = pow(2 * r, d) * pow(fl::math::Const<Precision>::PI, n) * Factorial<Precision>(n) /
          Factorial<Precision>(d);
  }

  return val;
}

template < typename Precision, bool USE_NORMALIZATION = true >
class GaussianKernel {
  private:
    Precision neg_inv_bandwidth_2sq_;
    Precision bandwidth_sq_;

  public:
    static const bool HAS_CUTOFF = false;

  public:
    Precision bandwidth_sq() const {
      return bandwidth_sq_;
    }

    void Init(Precision bandwidth_in, index_t dims) {
      Init(bandwidth_in);
    }

    /**
     * Initializes to a specific bandwidth.
     *
     * @param bandwidth_in the standard deviation sigma
     */
    void Init(Precision bandwidth_in) {
      bandwidth_sq_ = bandwidth_in * bandwidth_in;
      neg_inv_bandwidth_2sq_ = -1.0 / (2.0 * bandwidth_sq_);
    }

    /**
     * Evaluates an unnormalized density, given the distance between
     * the kernel's mean and a query point.
     */
    Precision EvalUnnorm(Precision dist) const {
      return EvalUnnormOnSq(dist * dist);
    }

    /**
     * Evaluates an unnormalized density, given the square of the
     * distance.
     */
    Precision EvalUnnormOnSq(Precision sqdist) const {
      Precision d = exp(sqdist * neg_inv_bandwidth_2sq_);
      return d;
    }

    /** Unnormalized range on a range of squared distances. */
    GenRange<Precision> RangeUnnormOnSq(const GenRange<Precision>& range) const {
      return GenRange<Precision>(EvalUnnormOnSq(range.hi), EvalUnnormOnSq(range.lo));
    }

    /**
     * Gets the maximum unnormalized value.
     */
    Precision MaxUnnormValue() {
      return 1;
    }

    /**
     * Divide by this constant when you're done.
     */
    Precision CalcNormConstant(index_t dims) const {
      // Changed because * faster than / and 2 * fl::math::PI opt out.  RR
      //return pow((-fl::math::PI/neg_inv_bandwidth_2sq_), dims/2.0);
      if (USE_NORMALIZATION == true) {
        return pow(2 * fl::math::Const<Precision>::PI * bandwidth_sq_, dims / 2.0);
      }
      else {
        return 1;
      }
    }
};

/**
 * Standard multivariate Gaussian kernel.
 *
 */
template < typename Precision, bool USE_NORMALIZATION = true >
class GaussianStarKernel {
  private:
    Precision neg_inv_bandwidth_2sq_;
    Precision factor_;
    Precision bandwidth_sq_;
    Precision critical_point_sq_;
    Precision critical_point_value_;

  public:
    static const bool HAS_CUTOFF = false;

  public:
    Precision bandwidth_sq() const {
      return bandwidth_sq_;
    }

    /**
     * Initializes to a specific bandwidth.
     *
     * @param bandwidth_in the standard deviation sigma
     */
    void Init(Precision bandwidth_in, index_t dims) {
      bandwidth_sq_ = bandwidth_in * bandwidth_in;
      neg_inv_bandwidth_2sq_ = -1.0 / (2.0 * bandwidth_sq_);
      factor_ = pow(2.0, -dims / 2.0 - 1);
      critical_point_sq_ = 4 * bandwidth_sq_ * (dims / 2.0 + 2) *
                           fl::math::Const<Precision>::LN_2;
      critical_point_value_ = EvalUnnormOnSq(critical_point_sq_);
    }

    /**
     * Evaluates an unnormalized density, given the distance between
     * the kernel's mean and a query point.
     */
    Precision EvalUnnorm(Precision dist) const {
      return EvalUnnormOnSq(dist * dist);
    }

    /**
     * Evaluates an unnormalized density, given the square of the
     * distance.
     */
    Precision EvalUnnormOnSq(Precision sqdist) const {
      Precision d =
        factor_ * exp(sqdist * neg_inv_bandwidth_2sq_ * 0.5)
        - exp(sqdist * neg_inv_bandwidth_2sq_);
      return d;
    }

    /** Unnormalized range on a range of squared distances. */
    GenRange<Precision> RangeUnnormOnSq(const GenRange<Precision>& range) const {
      Precision eval_lo = EvalUnnormOnSq(range.lo);
      Precision eval_hi = EvalUnnormOnSq(range.hi);
      if (range.lo < critical_point_sq_) {
        if (range.hi < critical_point_sq_) {
          // Strictly under critical point.
          return GenRange<Precision>(eval_lo, eval_hi);
        }
        else {
          // Critical point is included
          return GenRange<Precision>(std::min(eval_lo, eval_hi), critical_point_value_);
        }
      }
      else {
        return GenRange<Precision>(eval_hi, eval_lo);
      }
    }

    /**
     * Divide by this constant when you're done.
     *
     * @deprecated -- this function is very confusing
     */
    Precision CalcNormConstant(index_t dims) const {
      if (USE_NORMALIZATION) {
        return 
	pow(2.0 * fl::math::Const<Precision>::PI * bandwidth_sq_, 
	    ((Precision) dims) / 2.0 ) / 2.0;
      }
      else {
        return 1;
      }
    }

    /**
     * Multiply densities by this value.
     */
    Precision CalcMultiplicativeNormConstant(index_t dims) const {
      return 1.0 / CalcNormConstant(dims);
    }
};

/**
 * Multivariate Epanechnikov kernel.
 *
 * To use, first get an unnormalized density, and divide by the
 * normalizeation factor.
 */
template<typename Precision>
class EpanKernel {
  private:
    Precision inv_bandwidth_sq_;
    Precision bandwidth_sq_;

  public:
    static const bool HAS_CUTOFF = true;

  public:
    void Init(Precision bandwidth_in, index_t dims) {
      Init(bandwidth_in);
    }

    /**
     * Initializes to a specific bandwidth.
     */
    void Init(Precision bandwidth_in) {
      bandwidth_sq_ = (bandwidth_in * bandwidth_in);
      inv_bandwidth_sq_ = 1.0 / bandwidth_sq_;
    }

    /**
     * Evaluates an unnormalized density, given the distance between
     * the kernel's mean and a query point.
     */
    Precision EvalUnnorm(Precision dist) const {
      return EvalUnnormOnSq(dist * dist);
    }

    /**
     * Evaluates an unnormalized density, given the square of the
     * distance.
     */
    Precision EvalUnnormOnSq(Precision sqdist) const {
      // TODO: Try the fabs non-branching version.
      if (sqdist < bandwidth_sq_) {
        return 1 - sqdist * inv_bandwidth_sq_;
      }
      else {
        return 0;
      }
    }

    /** Unnormalized range on a range of squared distances. */
    GenRange<Precision> RangeUnnormOnSq(const GenRange<Precision>& range) const {
      return GenRange<Precision>(EvalUnnormOnSq(range.hi), EvalUnnormOnSq(range.lo));
    }

    /**
     * Gets the maximum unnormalized value.
     */
    Precision MaxUnnormValue() {
      return 1.0;
    }

    /**
     * Divide by this constant when you're done.
     */
    Precision CalcNormConstant(index_t dims) const {
      return 2.0 * fl::math::SphereVolume(sqrt(bandwidth_sq_), dims)
             / (dims + 2.0);
    }

    /**
     * Gets the squared bandwidth.
     */
    Precision bandwidth_sq() const {
      return bandwidth_sq_;
    }

    /**
    * Gets the reciproccal of the squared bandwidth.
     */
    Precision inv_bandwidth_sq() const {
      return inv_bandwidth_sq_;
    }
};

/**
 * Computes the factorial of an integer.
 */
template<typename Precision>
Precision Factorial(int d) {
  Precision v = 1;

  DEBUG_ASSERT(d >= 0);

  for (int i = 2; i <= d; i++) {
    v *= i;
  }

  return v;
}

/**
 * Computes the binomial coefficient, n choose k for nonnegative integers
 * n and k
 *
 * @param n the first nonnegative integer argument
 * @param k the second nonnegative integer argument
 * @return the binomial coefficient n choose k
 */

template<typename Precision>
Precision BinomialCoefficient(int n, int k) {
  int n_k = n - k;
  Precision nchsk = 1;
  int i;

  if (k > n || k < 0) {
    return 0;
  }

  if (k < n_k) {
    k = n_k;
    n_k = n - k;
  }

  for (i = 1; i <= n_k; i++) {
    nchsk *= (++k);
    nchsk /= i;
  }
  return nchsk;
}

/**
 * Creates an identity permutation where the element i equals i.
 *
 * For instance, result[0] == 0, result[1] == 1, result[2] == 2, etc.
 *
 * @param size the size to initialize the result to
 * @param result will be initialized to the identity permutation
 */
inline void MakeIdentityPermutation(
  index_t size, std::vector<index_t> *result) {
  result->resize(size);
  MakeIdentityPermutation(size, &(*result)[0]);
}

/**
 * Creates a random permutation over integers 0 throush size - 1.
 *
 * @param size the number of elements
 * @param result will be initialized to a permutation array
 */
inline void MakeRandomPermutation(
  index_t size, std::vector<index_t> *result) {
  result->resize(size);
  MakeRandomPermutation(size, &(*result)[0]);
}

/**
 * Inverts or transposes an existing permutation.
 */
inline void MakeInversePermutation(
  const std::vector<index_t>& original, std::vector<index_t> *reverse) {
  reverse->resize(original.size());
  MakeInversePermutation(original.size(), &original[0], &(*reverse)[0]);
}

template<typename TAnyIntegerType>
inline bool IsPowerTwo(TAnyIntegerType i) {
  return (i & (i - 1)) == 0;
}

/**
 * Finds the log base 2 of an integer.
 *
 * This integer must absolutely be a power of 2.
 */
inline unsigned IntLog2(unsigned i) {
  unsigned l;
  for (l = 0; (unsigned(1) << l) != i; l++) {
    DEBUG_ASSERT_MSG(l < 1024, "Taking IntLog2 of a non-power-of-2.");
  }
  return l;
}

} // namespace math
} // namespace fl



