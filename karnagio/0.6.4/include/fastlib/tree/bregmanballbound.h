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

#ifndef FL_LITE_INCLUDE_FASTLIB_TREE_BREGMAL_BALLBOUND_H_
#define FL_LITE_INCLUDE_FASTLIB_TREE_BREGMAL_BALLBOUND_H_
#include "bounds.h"

namespace fl {
namespace tree {
template<typename PointType>
class BregmanBallBound  : public BallBound<PointType> {
  public:
    typedef typename PointType::CalcPrecision_t CalcPrecision_t;
    typedef PointType Point_t;

    BregmanBallBound() {
      epsilon_ = 0.01;
    }

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & boost::serialization::make_nvp("radius", this->radius_);
      ar & boost::serialization::make_nvp("radius_sq", this->radius_sq_);
      ar & boost::serialization::make_nvp("center", this->center_);
      ar & boost::serialization::make_nvp("center_gradient", this->center_gradient_);

    }
    template<typename StreamType>
    void Print(StreamType &stream, const std::string &delim) const {
      stream << "*BregmanBallBound*\n";
      stream << "centroid:";
      this->center_.Print(stream, delim);
      stream << "\n";
      stream << "radius:" << this->radius_ << "\n";
    }

    void set_radius(CalcPrecision_t d) {
      this->radius_ = d;
      this->radius_sq_ = d * d;
    }


    const Point_t& center() const {
      return this->center_;
    }

    Point_t& center() {
      return this->center_;
    }

    template<typename DivergenceType>
    void PostProcessCenter(DivergenceType &div) {
      this->center_gradient_.Copy(this->center_);
      div.Gradient(this->center_, &this->center_gradient_);
    }

    GenRange<CalcPrecision_t> get(index_t d) const {
      fl::logger->Die() << "Function \"GenRange<CalcPrecision_t> get(index_t d)\" "
      " not defined for bregman balls";
    }
    /**
     * Determines if a point is within this bound.
     */
    template<typename DivergenceType, typename OtherPointType>
    bool Contains(const DivergenceType &divergence, const OtherPointType& point) const {
      return  divergence.DistanceSq(point, this->center_) < this->radius_sq_;
    }


    /**
     *  @brief It computes the maximum possible distance between two points
     *  in a ball.
     */
    typename Point_t::CalcPrecision_t MaxDistanceWithinBound() const {
      fl::logger->Die() << "Function \"MaxDistanceWithinBound())\" "
      " not defined for bregman balls";
      return 0;
    }


    /** @brief Initializes to specified dimensionality.
     */
    template<typename TableType>
    void Init(TableType &table) {
      this->radius_ = 0;
    }

    /**
     * Calculates minimum bound-to-point squared distance.
     */
    template<typename DivergenceType, typename OtherPointType>
    CalcPrecision_t MinDistance(const DivergenceType &divergence, const OtherPointType& point) const {
      return fl::math::Pow<double, 1, 2>(MinDistanceSq(divergence, point));
    }

    template<typename DivergenceType, typename OtherPointType>
    CalcPrecision_t MinDistanceSq(const DivergenceType &divergence, const OtherPointType& point) const {
      // Get the gradient at the query point.
      OtherPointType point_gradient;
      point_gradient.Copy(point);
      divergence.Gradient(point, &point_gradient);
      double lo = 0;
      double hi = 1;
      PointType trial_point_gradient;
      PointType trial_point;
      PointType succ_trial_point;
      trial_point.Copy(point);
      CalcPrecision_t succ_distance_sq = -1;
      // Interpolate between the gradient of the divergence at the
      // query and the reference centroids.
      do {
        trial_point_gradient.Copy(this->center_gradient_);
        double theta = 0.5 * (lo + hi);
        fl::la::AddExpert(
          (1 - theta) / theta, point_gradient, &trial_point_gradient);
        fl::la::SelfScale(theta, &trial_point_gradient);

        // Use the convex conjugate function to apply the inverse mapping.
        divergence.GradientConvexConjugate(trial_point_gradient, &trial_point);
        CalcPrecision_t distance_sq = divergence.DistanceSq(
                                        trial_point, this->center_);
        if (distance_sq > this->radius_sq_) {
          succ_distance_sq = distance_sq;
          succ_trial_point.Copy(trial_point);
          lo = theta;
        }
        else {
          hi = theta;
        }

        // If the interval becomes too narrow, then break.
        if (fabs(lo - hi) <= 1e-6) {
          break;
        }
      }
      while (
        succ_distance_sq < this->radius_sq_ ||
        fabs(succ_distance_sq - this->radius_sq_) / this->radius_sq_ >=
        this->epsilon_);

      // If the bisection search returned a projection that is still
      // inside the reference ball, then consider the search a failure
      // and return 0 since it is not a valid outer projection.
      if (succ_distance_sq < this->radius_sq_) {
        return 0;
      }

      // Return the divergence between the trial point which is inside
      // the reference ball and the query point.
      double divergence_between_query_and_projection =
        divergence.DistanceSq(succ_trial_point, point);

      return divergence_between_query_and_projection;
    }

    /**
     * Calculates minimum bound-to-bound squared distance.
     */
    template<typename DivergenceType>
    CalcPrecision_t MinDistance(const DivergenceType &divergence,
                                const BregmanBallBound& other) const {
      return fl::math::Pow<double, 1, 2>(MinDistSq(divergence, other));
    }

    template<typename DivergenceType>
    CalcPrecision_t MinDistanceSq(const DivergenceType &divergence,
                                  const BregmanBallBound& other) const {
      // Get the gradient at the query point.
      double other_radius_sq = other.radius_sq_;
      const PointType &other_center = other.center_;
      const PointType &other_center_gradient = other.center_gradient_;

      PointType trial_point1;
      PointType succ_trial_point1;
      PointType trial_point_gradient1;
      trial_point1.Copy(this->center_);
      succ_trial_point1.Copy(this->center_);

      // Do a quick test. If the center of the other ball is within
      // this ball then the balls intersect.
      if (divergence.DistanceSq(
            other_center, this->center_) < this->radius_sq_) {
        return 0;
      }
      double lo = 0;
      double hi = 1;

      // Project the reference centroid on the surface of the query
      // ball.
      CalcPrecision_t succ_dist_sq = -1;
      do {
        double theta = 0.5 * (lo + hi);
        trial_point_gradient1.Copy(this->center_gradient_);
        fl::la::AddExpert(
          (1 - theta) / theta, other_center_gradient, &trial_point_gradient1);
        fl::la::SelfScale(theta, &trial_point_gradient1);

        // Use the convex conjugate function to apply the inverse mapping.
        divergence.GradientConvexConjugate(
          trial_point_gradient1, &trial_point1);
        CalcPrecision_t dist_sq = divergence.DistanceSq(
                                    trial_point1, this->center_);

        if (dist_sq > radius_sq_) {
          succ_dist_sq = dist_sq;
          succ_trial_point1.CopyValues(trial_point1);
          lo = theta;
        }
        else {
          hi = theta;
        }

        if (fabs(lo - hi) <= 1e-6) {
          break;
        }
      }
      while (
        succ_dist_sq < this->radius_sq_ ||
        fabs(succ_dist_sq - this->radius_sq_) / this->radius_sq_ >=
        this->epsilon_);

      // Check whether the reference centroid projection on the query
      // ball is a valid outer projection. If not, return failure
      // (zero).
      if (succ_dist_sq < this->radius_sq_) {
        return 0;
      }

      // Now project the query centroid on the shell of the reference
      // node bound.
      PointType trial_point2;
      PointType succ_trial_point2;
      PointType trial_point_gradient2;
      trial_point2.Copy(other_center);
      succ_trial_point2.Copy(other_center);
      succ_dist_sq = -1;
      lo = 0;
      hi = 1;
      do {
        double theta = 0.5 * (lo + hi);
        trial_point_gradient2.Copy(other_center_gradient);
        fl::la::AddExpert(
          (1 - theta) / theta, this->center_gradient_, &trial_point_gradient2);
        fl::la::SelfScale(theta, &trial_point_gradient2);

        // Use the convex conjugate function to apply the inverse mapping.
        divergence.GradientConvexConjugate(
          trial_point_gradient2, &trial_point2);
        CalcPrecision_t dist_sq = divergence.DistanceSq(
                                    trial_point2, other_center);

        if (dist_sq > other_radius_sq) {
          succ_dist_sq = dist_sq;
          succ_trial_point2.CopyValues(trial_point2);
          lo = theta;
        }
        else {
          hi = theta;
        }

        if (fabs(lo - hi) <= 1e-6) {
          break;
        }
      }
      while (
        succ_dist_sq < other_radius_sq ||
        fabs(succ_dist_sq - other_radius_sq) / other_radius_sq >=
        this->epsilon_);

      // Check whether the query centroid projection on the reference
      // ball is a valid outer projection. If not, return failure
      // (zero).
      if (succ_dist_sq < other_radius_sq) {
        return 0;
      }

      // Given the projection of the query centroid on the reference
      // ball and the projection of the reference centroid on the
      // query ball: test wheter the qproj is within the query ball or
      // the rproj is within the reference ball, in which case we
      // return min distance of zero.
      if (divergence.DistanceSq(
            succ_trial_point2, this->center_) < radius_sq_ ||
          divergence.DistanceSq(
            succ_trial_point1, other_center) < other_radius_sq) {
        return 0;
      }

      double return_value =
        divergence.DistanceSq(succ_trial_point2, succ_trial_point1);
      return return_value;
    }

    /**
     * Computes maximum distance.
     */
    template<typename DivergenceType, typename OtherPointType>
    CalcPrecision_t MaxDistance(const DivergenceType &divergence, const OtherPointType& point) const {
      fl::logger->Die() << "Function \"MaxDistance\" not defined for bregman balls";
      return 0;
    }

    template<typename DivergenceType, typename OtherPointType>
    CalcPrecision_t MaxDistanceSq(const DivergenceType &divergence, const OtherPointType& point) const {
      fl::logger->Die() << "Function \"MaxDistanceSq\" not defined for bregman balls";
      return 0;
    }

    /**
     * Computes maximum distance.
     */
    template<typename DivergenceType>
    CalcPrecision_t MaxDistance(const DivergenceType &divergence,
                                const BregmanBallBound& other) const {
      fl::logger->Die() << "Function \"MaxDistance\" not defined for bregman balls";
    }

    template<typename DivergenceType>
    CalcPrecision_t MaxDistanceSq(const DivergenceType &divergence,
                                  const BregmanBallBound& other) const {
      fl::logger->Die() << "Function \"MaxDistance\" not defined for bregman balls";
      return 0;
    }

    /**
     * Calculates minimum and maximum bound-to-bound squared distance.
     *
     * Example: bound1.MinDistanceSq(other) for minimum squared distance.
     */
    template<typename DivergenceType>
    GenRange<CalcPrecision_t> RangeDistance(const DivergenceType &divergence,
                                            const BregmanBallBound& other) const {
      fl::logger->Die() << "Function \"RangeDistance\" not defined for bregman balls";
    }

    template<typename DivergenceType, typename OtherPointType>
    GenRange<CalcPrecision_t> RangeDistance(const DivergenceType &divergence, const OtherPointType& other) const {
      fl::logger->Die() << "Function \"RangeDistance\" not defined for bregman balls";
    }

    template<typename DivergenceType>
    GenRange<CalcPrecision_t> RangeDistanceSq(const DivergenceType &divergence,
        const BregmanBallBound& other) const {
      fl::logger->Die() << "Function \"RangeDistanceSq\" not defined for bregman balls";
    }

    template<typename DivergenceType, typename OtherPointType>
    GenRange<CalcPrecision_t> RangeDistanceSq(const DivergenceType &divergence,
        const OtherPointType& other) const {
      fl::logger->Die() << "Function \"RangeDistanceSq\" not defined for bregman balls";
    }
  private:
    PointType center_gradient_;
    CalcPrecision_t epsilon_;
    CalcPrecision_t radius_sq_;
};

}
}

#endif
