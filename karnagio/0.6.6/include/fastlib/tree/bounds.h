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
 * @file tree/bounds.h
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * TODO: Come up with a better design so you can do plug-and-play distance
 * metrics.
 *
 * @experimental
 */

#ifndef FL_LITE_FASTLIB_TREE_BOUNDS_H
#define FL_LITE_FASTLIB_TREE_BOUNDS_H

#include <limits>
#include <string>
#include <vector>
#include "fastlib/dense/matrix.h"
#include "fastlib/math/fl_math.h"
#include "fastlib/math/gen_range.h"

namespace fl {
namespace tree {
/**
 * Hyper-rectangle bound for an L-metric.
 *
 * Template parameter t_pow is the metric to use; use 2 for
 * Euclidean (L2).
 */
template < typename StoragePrecision = double, typename CalcPrecision = double,
int t_pow = 2 >
class GenHrectBound {
  public:
    typedef StoragePrecision Precision_t;
    typedef CalcPrecision CalcPrecision_t;
    static const int PREFERRED_POWER = t_pow;
    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & boost::serialization::make_nvp("bounds", bounds_);
    }

    template<typename StreamType>
    void Print(StreamType &stream, const std::string &delim) const {
      stream << "*RectBound*" << "\n";
      stream << "hi:";
      for (index_t i = 0; i < bounds_.size(); i++) {
        stream << bounds_[i].hi << delim;
      }
      stream << "\n";
      stream << "lo:";
      for (index_t i = 0; i < bounds_.size(); i++) {
        stream << bounds_[i].lo << delim;
      }
      stream << "\n";
    }

  private:

    typedef GenRange<StoragePrecision> BoundsType;
    typedef std::vector<BoundsType> BoundsVector;
    BoundsVector bounds_;

  public:

    GenHrectBound() {
    }

    ~GenHrectBound() {
    }

    /**
     * Initializes to specified dimensionality with each dimension the empty
     * set.
     */
    template<typename TableType>
    void Init(TableType &table) {
      //DEBUG_ASSERT_MSG(bounds_.size() == BIG_BAD_NUMBER, "Already initialized");
      bounds_.resize(table.n_attributes());
      Reset();
    }

    /**
     * Resets all dimensions to the empty set.
     */
    void Reset() {
      for (index_t i = 0; i < bounds_.size(); i++) {
        bounds_[i].InitEmptySet();
      }
    }

    /**
     * Determines if a point is within this bound.
     */
    template<typename PointType>
    bool Contains(const PointType &point) const {
      for (index_t i = 0; i < point.length(); i++) {
        if (!bounds_[i].Contains(point[i])) {
          return false;
        }
      }

      return true;
    }

    /** Gets the dimensionality */
    index_t dim() const {
      return bounds_.size();
    }

    /**
     * Gets the range for a particular dimension.
     */
    const GenRange<StoragePrecision> & get(index_t i) const {
      DEBUG_BOUNDS(i, bounds_.size());
      return bounds_[i];
    }

    CalcPrecision MaxDistanceWithinBound() const {
      CalcPrecision max_dist_within = 0;
      for (index_t i = 0; i < bounds_.size(); i++) {
        max_dist_within += math::Pow<CalcPrecision, t_pow, 1>(bounds_[i].hi - bounds_[i].lo);
      }
      return math::Pow<CalcPrecision, 1, t_pow>(max_dist_within);
    }

    /** Calculates the midpoint of the range, cetroid must be initialized */
    template<typename PointType>
    void CalculateMidpoint(PointType *centroid) const {
      fl::logger->Die() << "Don't use this";
    }

    /** Calculates the midpoint of the range */
    template<typename PointType>
    void CalculateMidpointOverwrite(PointType *centroid) const {
      for (index_t i = 0; i < bounds_.size(); i++) {
        (*centroid)[i] = bounds_[i].mid();
      }
    }


    /**
     * Calculates minimum bound-to-point squared distance.
     */
    template<typename MetricType, typename PointType>
    CalcPrecision MinDistanceSq(const MetricType &metric, const PointType &point)
    const {
      DEBUG_SAME_SIZE(static_cast<index_t>(point.size()), bounds_.size());
      CalcPrecision sum = 0;
      for (index_t i = 0; i < bounds_.size(); i++) {
        StoragePrecision v =  point[i];
        StoragePrecision v1 = bounds_[i].lo - v;
        StoragePrecision v2 = v - bounds_[i].hi;
        v = (v1 + fabs(v1)) + (v2 + fabs(v2));
        sum += fl::math::Pow<CalcPrecision, t_pow, 1>(v); // v is non-negative
      }
      return fl::math::Pow<CalcPrecision, 2, t_pow>(sum) / 4;
    }

    /**
     * Calculates minimum bound-to-bound squared distance.
     *
     * Example: bound1.MinDistanceSq(other) for minimum squared distance.
     */
    template<typename MetricType>
    CalcPrecision MinDistanceSq(const MetricType &metric,
                                const GenHrectBound &other) const {
      CalcPrecision sum = 0;

      DEBUG_SAME_SIZE(bounds_.size(), other.bounds_.size());

      typename std::vector<GenRange<StoragePrecision> >::const_iterator a = bounds_.begin();
      typename std::vector<GenRange<StoragePrecision> >::const_iterator b = other.bounds_.begin();
      for (; a != bounds_.end(); ++a, ++b) {
        CalcPrecision v1 = b->lo - a->hi;
        CalcPrecision v2 = a->lo - b->hi;
        // We invoke the following:
        //   x + fabs(x) = max(x * 2, 0)
        //   (x * 2)^2 / 4 = x^2
        CalcPrecision v = (v1 + fabs(v1)) + (v2 + fabs(v2));

        sum += fl::math::Pow<CalcPrecision, t_pow, 1>(v); // v is non-negative
      }

      return fl::math::Pow<CalcPrecision, 2, t_pow>(sum) / 4;
    }


    /**
     * Calculates maximum bound-to-point squared distance.
     */
    template<typename MetricType, typename PointType>
    CalcPrecision MaxDistanceSq(const MetricType &metric, const PointType &point)
    const {

      CalcPrecision sum = 0;

      DEBUG_SAME_SIZE((index_t)point.size(), (index_t)bounds_.size());

      for (index_t d = 0; d < bounds_.size(); d++) {
        CalcPrecision v = std::max(point[d] - bounds_[d].lo,
                                   bounds_[d].hi - point[d]);
        sum += fl::math::Pow<CalcPrecision, t_pow, 1>(v); // v is non-negative
      }

      return fl::math::Pow<CalcPrecision, 2, t_pow>(sum);
    }

    /**
     * Calculates maximum bound-to-point squared distance.
     */
    template<typename MetricType>
    CalcPrecision MaxDistanceSq(const MetricType &metric,
                                const StoragePrecision *point) const {
      CalcPrecision sum = 0;

      for (index_t d = 0; d < bounds_.size(); d++) {
        CalcPrecision v = std::max(point[d] - bounds_[d].lo,
                                   bounds_[d].hi - point[d]);
        sum += fl::math::Pow<CalcPrecision, t_pow, 1>(v); // v is non-negative
      }

      return fl::math::Pow<CalcPrecision, 2, t_pow>(sum);
    }

    /**
     * Computes maximum distance.
     */
    template<typename MetricType>
    CalcPrecision MaxDistanceSq(const MetricType &metric, const GenHrectBound& other) const {
      CalcPrecision sum = 0;

      DEBUG_SAME_SIZE(bounds_.size(), other.bounds_.size());

      typename std::vector<GenRange<StoragePrecision> >::const_iterator a = bounds_.begin();
      typename std::vector<GenRange<StoragePrecision> >::const_iterator b = other.bounds_.begin();
      for (; a != bounds_.end(); ++a, ++b) {
        CalcPrecision v = std::max(b->hi - a->lo, a->hi - b->lo);

        // v is non-negative
        sum += fl::math::PowAbs<CalcPrecision, t_pow, 1>(v);
      }

      return fl::math::Pow<CalcPrecision, 2, t_pow>(sum);
    }

    /**
     * Calculates minimum and maximum bound-to-bound squared distance.
     */
    template<typename MetricType>
    GenRange<CalcPrecision> RangeDistanceSq(const MetricType &metric,
                                            const GenHrectBound& other) const {

      CalcPrecision sum_lo = 0;
      CalcPrecision sum_hi = 0;

      DEBUG_SAME_SIZE(bounds_.size(), other.bounds_.size());

      typename std::vector<GenRange<CalcPrecision> >::const_iterator a = bounds_.begin();
      typename std::vector<GenRange<CalcPrecision> >::const_iterator b = other.bounds_.begin();
      for (; a != bounds_.end(); ++a, ++b) {
        CalcPrecision v1 = b->lo - a->hi;
        CalcPrecision v2 = a->lo - b->hi;
        // We invoke the following:
        //   x + fabs(x) = max(x * 2, 0)
        //   (x * 2)^2 / 4 = x^2
        CalcPrecision v_lo = (v1 + fabs(v1)) + (v2 + fabs(v2));
        CalcPrecision v_hi = -std::min(v1, v2);

        // v_lo is non-negative.
        sum_lo += fl::math::Pow<CalcPrecision, t_pow, 1>(v_lo);
        // v_hi is non-negative.
        sum_hi += fl::math::Pow<CalcPrecision, t_pow, 1>(v_hi);
      }

      return GenRange<CalcPrecision>(
               fl::math::Pow<CalcPrecision, 2, t_pow>(sum_lo) / 4,
               fl::math::Pow<CalcPrecision, 2, t_pow>(sum_hi));
    }
    /**
    * Calculates minimum and maximum bound-to-point squared distance.
    */
    template<typename MetricType, typename PointType>
    GenRange<CalcPrecision> RangeDistanceSq(const MetricType &metric, const PointType &point) const {

      CalcPrecision sum_lo = 0;
      CalcPrecision sum_hi = 0;

      DEBUG_SAME_SIZE(point.size(), bounds_.size());

      for (int i = 0; i < bounds_.size(); ++i) {
        CalcPrecision v = point[i];
        CalcPrecision v1 = bounds_[i].lo - v;
        CalcPrecision v2 = v - bounds_[i].hi;
        sum_lo += fl::math::Pow<CalcPrecision, t_pow, 1>(
                    (v1 + fabs(v1)) + (v2 + fabs(v2)));
        sum_hi += fl::math::Pow<CalcPrecision, t_pow, 1>(-std::min(v1, v2));
      }
      return GenRange<CalcPrecision>(
               fl::math::Pow<CalcPrecision, 2, t_pow>(sum_lo) / 4,
               fl::math::Pow<CalcPrecision, 2, t_pow>(sum_hi));
    }


    /**
     * Calculates closest-to-their-midpoint bounding box distance,
     * i.e. calculates their midpoint and finds the minimum box-to-point
     * distance.
     *
     * Equivalent to:
     * <code>
     * other.CalcMidpoint(&other_midpoint)
     * return MinDistanceSqToPoint(other_midpoint)
     * </code>
     */
    template<typename MetricType>
    CalcPrecision MinToMidSq(const GenHrectBound& other) const {
      CalcPrecision sum = 0;

      DEBUG_SAME_SIZE(bounds_.size(), other.bounds_.size());

      typename std::vector<GenRange<StoragePrecision> >::const_iterator a = bounds_->begin();
      typename std::vector<GenRange<StoragePrecision> >::const_iterator b = other.bounds_->begin();
      for (; a != bounds_.end(); ++a, ++b) {
        CalcPrecision v = b->mid();
        CalcPrecision v1 = a->lo - v;
        CalcPrecision v2 = v - a->hi;

        v = (v1 + fabs(v1)) + (v2 + fabs(v2));

        sum += fl::math::Pow<CalcPrecision, t_pow, 1>(v); // v is non-negative
      }

      return fl::math::Pow<CalcPrecision, 2, t_pow>(sum) / 4;
    }

    /**
     * Computes minimax distance, where the other node is trying to avoid me.
     */
    template<typename MetricType>
    CalcPrecision MinimaxDistanceSq(const MetricType &metric, const GenHrectBound& other) const {
      CalcPrecision sum = 0;
      index_t mdim = bounds_.size();

      DEBUG_SAME_SIZE(bounds_.size(), other.bounds_.size());

      typename std::vector<GenRange<StoragePrecision> >::const_iterator a = bounds_.begin();
      typename std::vector<GenRange<StoragePrecision> >::const_iterator b = other.bounds_.begin();
      for (; a != bounds_.end(); ++a, ++b) {
        CalcPrecision v1 = b->hi - a->hi;
        CalcPrecision v2 = a->lo - b->lo;
        CalcPrecision v = std::max(v1, v2);
        v = (v + fabs(v)); /* truncate negatives to zero */
        sum += fl::math::Pow<CalcPrecision, t_pow, 1>(v); // v is non-negative
      }

      return fl::math::Pow<CalcPrecision, 2, t_pow>(sum) / 4;
    }

    /**
     * Calculates midpoint-to-midpoint bounding box distance.
     */
    template<typename MetricType>
    CalcPrecision MidDistanceSq(const MetricType &metric, const GenHrectBound& other) const {
      CalcPrecision sum = 0;

      DEBUG_SAME_SIZE(bounds_.size(), other.bounds_.size());

      typename std::vector<GenRange<StoragePrecision> >::const_iterator a = bounds_.begin();
      typename std::vector<GenRange<StoragePrecision> >::const_iterator b = other.bounds_.begin();
      for (; a != bounds_.end(); ++a, ++b) {
        sum += fl::math::PowAbs<CalcPrecision, t_pow, 1>(
                 a->hi + a->lo - b->hi - b->lo);
      }

      return fl::math::Pow<CalcPrecision, 2, t_pow>(sum) / 4;
    }

    /**
     * Expands this region to include a new point.
     */
    template<typename PointType>
    GenHrectBound<StoragePrecision, CalcPrecision, t_pow> &
    operator |= (const PointType& vector) {

      DEBUG_SAME_SIZE(static_cast<index_t>(vector.length()), bounds_.size());

      for (index_t i = 0; i < bounds_.size(); i++) {
        bounds_[i] |= vector[i];
      }

      return *this;
    }

    /**
     * Expands this region to encompass another bound.
     */
    GenHrectBound<StoragePrecision, CalcPrecision, t_pow> &operator |=
    (const GenHrectBound& other) {
      DEBUG_SAME_SIZE(other.bounds_.size(), bounds_.size());

      for (index_t i = 0; i < bounds_.size(); i++) {
        bounds_[i] |= other.bounds_[i];
      }

      return *this;
    }
};

/**
 * Ball bound that works in arbitrary metric spaces.
 *
 * See LMetric for an example metric template parameter.
 *
 * To initialize this, set the radius with @c set_radius
 * and set the point by initializing @c point() directly.
 */
template<typename PointType>
class BallBound {
  public:
    typedef typename PointType::CalcPrecision_t CalcPrecision_t;
    typedef PointType Point_t;

  protected:
    CalcPrecision_t radius_;
    Point_t center_;

  public:
    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & boost::serialization::make_nvp("radius", radius_);
      ar & boost::serialization::make_nvp("center", center_);
    }
    CalcPrecision_t radius() const {
      return radius_;
    }

    template<typename StreamType>
    void Print(StreamType &stream, const std::string &delim) const {
      stream << "*BallBound*\n";
      stream << "centroid:";
      center_.Print(stream, delim);
      stream << "\n";
      stream << "radius:" << radius_ << "\n";
    }

    void set_radius(CalcPrecision_t d) {
      radius_ = d;
    }

    const Point_t& center() const {
      return center_;
    }

    Point_t& center() {
      return center_;
    }
    
    template<typename PostProcessorType>
    void PostProcessCenter(PostProcessorType &p) {
    
    }

    GenRange<CalcPrecision_t> get(index_t d) const {
      return GenRange<CalcPrecision_t>(center_[d] - radius_, center_[d] + radius_);
    }
    /**
     * Determines if a point is within this bound.
     */
    template<typename MetricType>
    bool Contains(const MetricType &metric, const Point_t& point) const {
      return MidDistance(metric, point) <= radius_;
    }

    /**
     * Gets the center.
     *
     * Don't really use this directly.  This is only here for consistency
     * with DHrectBound, so it can plug in more directly if a "centroid"
     * is needed.
     */
    void CalculateMidpoint(Point_t *centroid) const {
      centroid->Copy(center_);
    }

    /**
     *  @brief It computes the maximum possible distance between two points
     *  in a ball. This is two times the radius
     */
    typename Point_t::CalcPrecision_t MaxDistanceWithinBound() const {
      return 2*radius_;
    }


    /** @brief Initializes to specified dimensionality.
     */
    template<typename TableType>
    void Init(TableType &table) {
      //DEBUG_ASSERT_MSG(dim_ == BIG_BAD_NUMBER, "Already initialized");
      //center_.Init(dimension);
      radius_ = 0;
    }

    /**
     * Calculates minimum bound-to-point squared distance.
     */
    template<typename MetricType, typename OtherPointType>
    CalcPrecision_t MinDistance(const MetricType &metric, const OtherPointType& point) const {
      return fl::math::ClampNonNegative(MidDistance(metric, point) - radius_);
    }

    template<typename MetricType, typename OtherPointType>
    CalcPrecision_t MinDistanceSq(const MetricType &metric, const OtherPointType& point) const {
      return fl::math::Pow<CalcPrecision_t, 2, 1>(MinDistance(metric, point));
    }

    /**
     * Calculates minimum bound-to-bound squared distance.
     */
    template<typename MetricType>
    CalcPrecision_t MinDistance(const MetricType &metric, const BallBound& other) const {
      CalcPrecision_t delta = MidDistance(metric, other.center_) - radius_ -
                              other.radius_;
      return fl::math::ClampNonNegative(delta);
    }

    template<typename MetricType>
    CalcPrecision_t MinDistanceSq(const MetricType &metric, const BallBound& other) const {
      return fl::math::Pow<CalcPrecision_t, 2, 1>(MinDistance(metric, other));
    }

    /**
     * Computes maximum distance.
     */
    template<typename MetricType, typename OtherPointType>
    CalcPrecision_t MaxDistance(const MetricType &metric, const OtherPointType& point) const {
      return MidDistance(metric, point) + radius_;
    }

    template<typename MetricType, typename OtherPointType>
    CalcPrecision_t MaxDistanceSq(const MetricType &metric, const OtherPointType& point) const {
      return fl::math::Pow<CalcPrecision_t, 2, 1>(MaxDistance(metric, point));
    }

    /**
     * Computes maximum distance.
     */
    template<typename MetricType>
    CalcPrecision_t MaxDistance(const MetricType &metric, const BallBound& other) const {
      return MidDistance(metric, other.center_) + radius_ + other.radius_;
    }

    template<typename MetricType>
    CalcPrecision_t MaxDistanceSq(const MetricType &metric, const BallBound& other) const {
      return fl::math::Pow<CalcPrecision_t, 2, 1>(MaxDistance(metric, other));
    }

    /**
     * Calculates minimum and maximum bound-to-bound squared distance.
     *
     * Example: bound1.MinDistanceSq(other) for minimum squared distance.
     */
    template<typename MetricType>
    GenRange<CalcPrecision_t> RangeDistance(const MetricType &metric, const BallBound& other) const {
      CalcPrecision_t delta = MidDistance(metric, other.center_);
      CalcPrecision_t sumradius = radius_ + other.radius_;
      return GenRange<CalcPrecision_t>
             (fl::math::ClampNonNegative(delta - sumradius), delta + sumradius);
    }

    template<typename MetricType, typename OtherPointType>
    GenRange<CalcPrecision_t> RangeDistance(const MetricType &metric, const OtherPointType& other) const {
      CalcPrecision_t delta = MidDistance(metric, other);
      CalcPrecision_t sumradius = radius_;
      return GenRange<CalcPrecision_t>
             (fl::math::ClampNonNegative(delta - sumradius), delta + sumradius);
    }

    template<typename MetricType>
    GenRange<CalcPrecision_t> RangeDistanceSq(const MetricType &metric, const BallBound& other) const {
      CalcPrecision_t delta = MidDistance(metric, other.center_);
      CalcPrecision_t sumradius = radius_ + other.radius_;
      return GenRange<CalcPrecision_t>(
               fl::math::Pow<CalcPrecision_t, 2, 1>
               (fl::math::ClampNonNegative(delta - sumradius)),
               fl::math::Pow<CalcPrecision_t, 2, 1>(delta + sumradius));
    }

    template<typename MetricType, typename OtherPointType>
    GenRange<CalcPrecision_t> RangeDistanceSq(const MetricType &metric,
        const OtherPointType& other) const {
      CalcPrecision_t delta = MidDistance(metric, other);
      CalcPrecision_t sumradius = radius_;
      return GenRange<CalcPrecision_t>(
               fl::math::Pow<CalcPrecision_t, 2, 1>
               (fl::math::ClampNonNegative(delta - sumradius)),
               fl::math::Pow<CalcPrecision_t, 2, 1>(delta + sumradius));
    }

    /**
     * Calculates closest-to-their-midpoint bounding box distance,
     * i.e. calculates their midpoint and finds the minimum box-to-point
     * distance.
     *
     * Equivalent to:
     * <code>
     * other.CalcMidpoint(&other_midpoint)
     * return MinDistanceSqToPoint(other_midpoint)
     * </code>
     */
    template<typename MetricType>
    CalcPrecision_t MinToMid(const MetricType &metric, const BallBound& other) const {
      CalcPrecision_t delta = MidDistance(metric, other.center_) - radius_;
      return fl::math::ClampNonNegative(delta);
    }

    template<typename MetricType>
    CalcPrecision_t MinToMidSq(const MetricType &metric, const BallBound& other) const {
      return fl::math::Pow<CalcPrecision_t, 2, 1>(MinToMid(metric, other));
    }

    /**
     * Computes minimax distance, where the other node is trying to
     * avoid me.
     */
    template<typename MetricType>
    CalcPrecision_t MinimaxDistance(const MetricType &metric, const BallBound& other) const {
      CalcPrecision_t delta = MidDistance(metric, other.center_) + other.radius_ -
                              radius_;
      return fl::math::ClampNonNegative(delta);
    }

    template<typename MetricType>
    CalcPrecision_t MinimaxDistanceSq(const MetricType &metric, const BallBound& other) const {
      return fl::math::Pow<CalcPrecision_t, 2, 1>(MinimaxDistance(metric, other));
    }

    /**
     * Calculates midpoint-to-midpoint bounding box distance.
     */
    template<typename MetricType>
    CalcPrecision_t MidDistance(const MetricType &metric, const BallBound& other) const {
      return MidDistance(metric, other.center_);
    }

    template<typename MetricType>
    CalcPrecision_t MidDistanceSq(const MetricType &metric, const BallBound& other) const {
      return fl::math::Pow<CalcPrecision_t, 2, 1>(MidDistance(metric, other));
    }
    template<typename MetricType, typename OtherPointType>
    CalcPrecision_t MidDistance(const MetricType &metric,
                                const OtherPointType& point) const {
      return metric.Distance(center_, point);
    }
};

class EmptyBound {

  public:
    void Init(index_t dimension_in) {
    }
    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {

    }
};
} // tree namespace
} // fl namespace

#endif
