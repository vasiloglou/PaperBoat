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
// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file math.h
 *
 * Includes all basic FASTlib non-vector math utilities.
 */

#ifndef GEN_RANGE_H
#define GEN_RANGE_H

#include "fastlib/base/base.h"
#include <limits>
#include <math.h>
#include "boost/serialization/nvp.hpp"


/**
 * Simple real-valued range.
 *
 * @experimental
 */
template<typename T>
class GenRange {
  public:
    /**
     * The lower bound.
     */
    T lo;
    /**
     * The upper bound.
     */
    T hi;

  public:

    GenRange() {
    }

    ~GenRange() {
    }

    /** Initializes to specified values. */
    GenRange(T lo_in, T hi_in)
        : lo(lo_in), hi(hi_in) {}
    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & boost::serialization::make_nvp("lo", lo);
      ar & boost::serialization::make_nvp("hi", hi);
    }
    /** Initialize to an empty set, where lo > hi. */
    void InitEmptySet() {
      lo = std::numeric_limits<T>::max();
      hi = -lo;
    }

    /** Initializes to -infinity to infinity. */
    void InitUniversalSet() {
      lo = -std::numeric_limits<T>::max();
      hi =  std::numeric_limits<T>::max();
    }

    /** Initializes to a range of values. */
    void Init(T lo_in, T hi_in) {
      lo = lo_in;
      hi = hi_in;
    }

    /**
     * Resets to a range of values.
     *
     * Since there is no dynamic memory this is the same as Init, but calling
     * Reset instead of Init probably looks more similar to surrounding code.
     */
    void Reset(T lo_in, T hi_in) {
      lo = lo_in;
      hi = hi_in;
    }

    /**
     * Gets the span of the range, hi - lo.
     */
    const T width() const {
      return hi - lo;
    }


    /**
     * Gets the midpoint of this range.
     */
    const T mid() const {
      return (hi + lo) / 2.0;
    }

    /**
     * Interpolates (factor) * hi + (1 - factor) * lo.
     */
    const T &interpolate(T factor) const {
      return factor * width() + lo;
    }

    /**
     * Simulate an union by growing the range if necessary.
     */
    const GenRange& operator |= (T d) {
      if (d < lo) {
        lo = d;
      }
      if (d > hi) {
        hi = d;
      }
      return *this;
    }

    /**
     * Sets this range to include only the specified value, or
     * becomes an empty set if the range does not contain the number.
     */
    const GenRange& operator &= (T d) {
      if (likely(d > lo)) {
        lo = d;
      }
      if (likely(d < hi)) {
        hi = d;
      }
      return *this;
    }

    /**
     * Expands range to include the other range.
     */
    const GenRange& operator |= (const GenRange& other) {
      if (unlikely(other.lo < lo)) {
        lo = other.lo;
      }
      if (unlikely(other.hi > hi)) {
        hi = other.hi;
      }
      return *this;
    }

    /**
     * Shrinks range to be the overlap with another range, becoming an empty
     * set if there is no overlap.
     */
    const GenRange& operator &= (const GenRange& other) {
      if (unlikely(other.lo > lo)) {
        lo = other.lo;
      }
      if (unlikely(other.hi < hi)) {
        hi = other.hi;
      }
      return *this;
    }

    /** Scales upper and lower bounds. */
    friend GenRange operator - (const GenRange& r) {
      return GenRange(-r.hi, -r.lo);
    }

    /** Scales upper and lower bounds. */
    const GenRange& operator *= (T d) {
      DEBUG_ASSERT_MSG
      (d >= 0, "don't multiply DRanges by negatives, explicitly negate");
      lo *= d;
      hi *= d;
      return *this;
    }

    /** Scales upper and lower bounds. */
    friend GenRange<T> operator * (const GenRange<T>& r, T d) {
      DEBUG_ASSERT_MSG
      (d >= 0, "don't multiply DRanges by negatives, explicitly negate");
      return GenRange(r.lo * d, r.hi * d);
    }

    /** Scales upper and lower bounds. */
    friend GenRange operator * (T d, const GenRange& r) {
      DEBUG_ASSERT_MSG
      (d >= 0, "don't multiply DRanges by negatives, explicitly negate");
      return GenRange(r.lo * d, r.hi * d);
    }

    /** Sums the upper and lower independently. */
    const GenRange& operator += (const GenRange& other) {
      lo += other.lo;
      hi += other.hi;
      return *this;
    }

    /** Subtracts from the upper and lower.
     * THIS SWAPS THE ORDER OF HI AND LO, assuming a worst case result.
     * This is NOT an undo of the + operator.
     */
    const GenRange& operator -= (const GenRange& other) {
      lo -= other.hi;
      hi -= other.lo;
      return *this;
    }

    /** Adds to the upper and lower independently. */
    const GenRange& operator += (T d) {
      lo += d;
      hi += d;
      return *this;
    }

    /** Subtracts from the upper and lower independently. */
    const GenRange& operator -= (T d) {
      lo -= d;
      hi -= d;
      return *this;
    }

    friend GenRange operator + (const GenRange& a, const GenRange& b) {
      GenRange result;
      result.lo = a.lo + b.lo;
      result.hi = a.hi + b.hi;
      return result;
    }

    friend GenRange operator - (const GenRange& a, const GenRange& b) {
      GenRange result;
      result.lo = a.lo - b.hi;
      result.hi = a.hi - b.lo;
      return result;
    }

    friend GenRange operator + (const GenRange& a, T b) {
      GenRange result;
      result.lo = a.lo + b;
      result.hi = a.hi + b;
      return result;
    }

    friend GenRange operator - (const GenRange& a, T b) {
      GenRange result;
      result.lo = a.lo - b;
      result.hi = a.hi - b;
      return result;
    }

    /**
     * Takes the maximum of upper and lower bounds independently.
     */
    void MaxWith(const GenRange& range) {
      if (unlikely(range.lo > lo)) {
        lo = range.lo;
      }
      if (unlikely(range.hi > hi)) {
        hi = range.hi;
      }
    }

    /**
     * Takes the minimum of upper and lower bounds independently.
     */
    void MinWith(const GenRange& range) {
      if (unlikely(range.lo < lo)) {
        lo = range.lo;
      }
      if (unlikely(range.hi < hi)) {
        hi = range.hi;
      }
    }

    /**
     * Takes the maximum of upper and lower bounds independently.
     */
    void MaxWith(T v) {
      if (unlikely(v > lo)) {
        lo = v;
        if (unlikely(v > hi)) {
          hi = v;
        }
      }
    }

    /**
     * Takes the minimum of upper and lower bounds independently.
     */
    void MinWith(T v) {
      if (unlikely(v < hi)) {
        hi = v;
        if (unlikely(v < lo)) {
          lo = v;
        }
      }
    }

    /**
     * Compares if this is STRICTLY less than another range.
     */
    friend bool operator < (const GenRange& a, const GenRange& b) {
      return a.hi < b.lo;
    }
    friend bool operator>(GenRange const &b, GenRange const &a) {
      return a < b;
    }
    friend bool operator<=(GenRange const &b, GenRange const &a) {
      return !(a < b);
    }
    friend bool operator>=(GenRange const &a, GenRange const &b) {
      return !(a < b);
    }

    /**
     * Compares if this is STRICTLY equal to another range.
     */
    friend bool operator == (const GenRange& a, const GenRange& b) {
      return a.lo == b.lo && a.hi == b.hi;
    }
    friend bool operator != (const GenRange& a, const GenRange& b) {
      return !(a == b);
    }

    /**
     * Compares if this is STRICTLY less than a value.
     */
    friend bool operator < (const GenRange& a, T b) {
      return a.hi < b;
    }
    friend bool operator>(T const &b, GenRange const &a) {
      return a < b;
    }
    friend bool operator<=(T const &b, GenRange const &a) {
      return !(a < b);
    }
    friend bool operator>=(GenRange const &a, T const &b) {
      return !(a < b);
    }

    /**
     * Compares if a value is STRICTLY less than this range.
     */
    friend bool operator < (T a, const GenRange& b) {
      return a < b.lo;
    }
    friend bool operator>(GenRange const &b, T const &a) {
      return a < b;
    }
    friend bool operator<=(GenRange const &b, T const &a) {
      return !(a < b);
    }
    friend bool operator>=(T const &a, GenRange const &b) {
      return !(a < b);
    }

    /**
     * Determines if a point is contained within the range.
     */
    bool Contains(T d) const {
      return d >= lo && d <= hi;
    }
};



typedef GenRange<double> DRange;

#endif
