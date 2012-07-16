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
#ifndef FL_LITE_FASTLIB_TREE_CLASSIFICATION_DECISION_TREE_PRIVATE_H
#define FL_LITE_FASTLIB_TREE_CLASSIFICATION_DECISION_TREE_PRIVATE_H

#include "fastlib/base/base.h"
#include "fastlib/tree/spacetree.h"
#include "fastlib/tree/bounds.h"
#include "fastlib/tree/cart_impurity.h"
#include "fastlib/tree/classification_decision_tree_partition.h"
#include <vector>
#include <map>
#include <deque>
#include <algorithm>
#include "boost/type_traits/is_same.hpp"

namespace fl {
namespace tree {

class CommonUpdate {
  public:
    template<typename SplitValuesType, typename CalcPrecision_t>
    CommonUpdate(SplitValuesType &split_values,
                 CalcPrecision_t best_gain,
                 std::string dimension_type) {
      *(split_values.best_improvement()) = best_gain;
      *(split_values.best_dimension_type()) = dimension_type;
      *(split_values.best_dimension()) = *(split_values.current_dimension());
    }
};

template<typename T>
class UpdateSplitValues {
  public:
    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      T split_val,
                      CalcPrecision_t best_gain);

    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      const std::string &split_val,
                      CalcPrecision_t best_gain);
};

template<>
class UpdateSplitValues<float> {
  public:
    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      float split_val,
                      CalcPrecision_t best_gain) {
      if (best_gain > *(split_values.best_improvement())) {

        // Update the global best.
        CommonUpdate(split_values, best_gain, std::string("float"));

        // Store the split value.
        split_values.float_dimension_split_values()->first =
          *(split_values.current_dimension());
        split_values.float_dimension_split_values()->second = split_val;
      }
    }

    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      const std::string &split_val,
                      CalcPrecision_t best_gain) {
    }
};

template<>
class UpdateSplitValues<double> {
  public:
    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      double split_val,
                      CalcPrecision_t best_gain) {
      if (best_gain > *(split_values.best_improvement())) {

        // Update the global best.
        CommonUpdate(split_values, best_gain, std::string("double"));

        // Store the split value.
        split_values.double_dimension_split_values()->first =
          *(split_values.current_dimension());
        split_values.double_dimension_split_values()->second = split_val;
      }
    }

    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      const std::string &split_val,
                      CalcPrecision_t best_gain) {
    }
};

template<>
class UpdateSplitValues<int> {
  public:
    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      int split_val,
                      CalcPrecision_t best_gain) {
    }

    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      std::string &split_val,
                      CalcPrecision_t best_gain) {
      if (best_gain > *(split_values.best_improvement())) {

        // Update the global best.
        CommonUpdate(split_values, best_gain, std::string("int"));

        // Store the split value.
        split_values.int_dimension_split_values()->first =
          *(split_values.current_dimension());
        split_values.int_dimension_split_values()->second = split_val;
      }
    }
};

template<>
class UpdateSplitValues<short unsigned int> {
  public:
    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      short unsigned int split_val,
                      CalcPrecision_t best_gain) {
    }

    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      std::string &split_val,
                      CalcPrecision_t best_gain) {
      if (best_gain > *(split_values.best_improvement())) {

        // Update the global best.
        CommonUpdate(split_values, best_gain, std::string("short unsigned int"));

        // Store the split value.
        split_values.int_dimension_split_values()->first =
          *(split_values.current_dimension());
        split_values.int_dimension_split_values()->second = split_val;
      }
    }
};

template<>
class UpdateSplitValues<unsigned char> {
  public:
    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      char split_val,
                      CalcPrecision_t best_gain) {
    }

    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      const std::string &split_val,
                      CalcPrecision_t best_gain) {
      if (best_gain > *(split_values.best_improvement())) {

        // Update the global best.
        CommonUpdate(split_values, best_gain, std::string("char"));

        // Store the split value.
        split_values.char_dimension_split_values()->first =
          *(split_values.current_dimension());
        split_values.char_dimension_split_values()->second = split_val;
      }
    }
};

template<>
class UpdateSplitValues<bool> {
  public:
    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      bool split_val,
                      CalcPrecision_t best_gain) {
    }

    template<typename SplitValuesType, typename CalcPrecision_t>
    UpdateSplitValues(SplitValuesType &split_values,
                      const std::string &split_val,
                      CalcPrecision_t best_gain) {
      if (best_gain > *(split_values.best_improvement())) {

        // Update the global best.
        CommonUpdate(split_values, best_gain, std::string("bool"));

        // Store the split value.
        split_values.bool_dimension_split_values()->first =
          *(split_values.current_dimension());
        split_values.bool_dimension_split_values()->second = split_val;
      }
    }
};

template<typename T>
class PartitionDimension {
};

template<>
class PartitionDimension<float> {
  public:
    PartitionDimension() {
    }

    template<typename MapSplitValuesType>
    void PartitionDimensionBranch(MapSplitValuesType &split_values) {
      float split_val;
      typedef typename MapSplitValuesType::TreeT::Point_t::CalcPrecision_t
      CalcPrecision_t;
      CalcPrecision_t best_gain = - std::numeric_limits<CalcPrecision_t>::max();
      PartitionNumericDimension(split_values, &split_val, &best_gain);

      // Update the best dimension.
      UpdateSplitValues<float>(split_values, split_val, best_gain);
    }
};

template<>
class PartitionDimension<double> {
  public:
    PartitionDimension() {
    }

    template<typename MapSplitValuesType>
    void PartitionDimensionBranch(MapSplitValuesType &split_values) {
      double split_val;
      typedef typename MapSplitValuesType::TreeT::Point_t::CalcPrecision_t
      CalcPrecision_t;
      CalcPrecision_t best_gain = - std::numeric_limits<CalcPrecision_t>::max();
      PartitionNumericDimension(split_values, &split_val, &best_gain);

      // Update the best dimension.
      UpdateSplitValues<double>(split_values, split_val, best_gain);
    }
};

template<>
class PartitionDimension<short unsigned int> {
  public:
    PartitionDimension() {
    }

    template<typename MapSplitValuesType>
    void PartitionDimensionBranch(MapSplitValuesType &split_values) {
      std::string split_val;
      typedef typename MapSplitValuesType::TreeT::Point_t::CalcPrecision_t
      CalcPrecision_t;
      CalcPrecision_t best_gain = - std::numeric_limits<CalcPrecision_t>::max();
      PartitionNominalDimension<short unsigned int>(split_values, &split_val, &best_gain);

      // Update the best dimension.
      UpdateSplitValues<int>(split_values, split_val, best_gain);
    }
};

template<>
class PartitionDimension<int32> {
  public:
    PartitionDimension() {
    }

    template<typename MapSplitValuesType>
    void PartitionDimensionBranch(MapSplitValuesType &split_values) {
      std::string split_val;
      typedef typename MapSplitValuesType::TreeT::Point_t::CalcPrecision_t
      CalcPrecision_t;
      CalcPrecision_t best_gain = - std::numeric_limits<CalcPrecision_t>::max();
      PartitionNominalDimension<int32>(split_values, &split_val, &best_gain);

      // Update the best dimension.
      UpdateSplitValues<int>(split_values, split_val, best_gain);
    }
};

template<>
class PartitionDimension<int64> {
  public:
    PartitionDimension() {
    }

    template<typename MapSplitValuesType>
    void PartitionDimensionBranch(MapSplitValuesType &split_values) {
      std::string split_val;
      typedef typename MapSplitValuesType::TreeT::Point_t::CalcPrecision_t
      CalcPrecision_t;
      CalcPrecision_t best_gain = - std::numeric_limits<CalcPrecision_t>::max();
      PartitionNominalDimension<int64>(split_values, &split_val, &best_gain);

      // Update the best dimension.
      UpdateSplitValues<int>(split_values, split_val, best_gain);
    }
};
template<>
class PartitionDimension<unsigned char> {
  public:
    PartitionDimension() {
    }

    template<typename MapSplitValuesType>
    void PartitionDimensionBranch(MapSplitValuesType &split_values) {
      std::string split_val;
      typedef typename MapSplitValuesType::TreeT::Point_t::CalcPrecision_t
      CalcPrecision_t;
      CalcPrecision_t best_gain = - std::numeric_limits<CalcPrecision_t>::max();
      PartitionNominalDimension<char>(split_values, &split_val, &best_gain);

      // Update the best dimension.
      UpdateSplitValues<unsigned char>(split_values, split_val, best_gain);
    }
};

template<>
class PartitionDimension<bool> {
  public:
    PartitionDimension() {
    }

    template<typename MapSplitValuesType>
    void PartitionDimensionBranch(MapSplitValuesType &split_values) {
      std::string split_val;
      typedef typename MapSplitValuesType::TreeT::Point_t::CalcPrecision_t
      CalcPrecision_t;
      CalcPrecision_t best_gain = - std::numeric_limits<CalcPrecision_t>::max();
      PartitionNominalDimension<bool>(split_values, &split_val, &best_gain);

      // Update the best dimension.
      UpdateSplitValues<bool>(split_values, split_val, best_gain);
    }
};

template<bool is_dense, typename T>
class TypeLength {
};

template<typename T>
class TypeLength<true, T> {
  public:
    template<typename PointType>
    static int Compute(const PointType &point) {
      return point.template dense_point<T>().length();
    }
};

template<typename T>
class TypeLength<false, T> {
  public:
    template<typename PointType>
    static int Compute(const PointType &point) {
      return point.template sparse_point<T>().length();
    }
};

template < typename ImpurityType, typename TreeIteratorType, typename TreeType,
bool is_dense >
class PartitionEachType {

  public:

    class MapSplitValues {

      public:

        typedef TreeIteratorType TreeIteratorT;

        typedef TreeType TreeT;

      private:

        ImpurityType *impurity_;

        TreeIteratorType *iterator_;

        TreeType *node_;

        std::pair<int, float> *float_dimension_split_values_;

        std::pair<int, double> *double_dimension_split_values_;

        std::pair<int, std::string> *bool_dimension_split_values_;

        std::pair<int, std::string> *char_dimension_split_values_;

        std::pair<int, std::string> *int_dimension_split_values_;

        int *current_dimension_;

        int *best_dimension_;

        std::string *best_dimension_type_;

        double *best_improvement_;

      public:

        const ImpurityType *impurity() const {
          return impurity_;
        }

        ImpurityType *impurity() {
          return impurity_;
        }

        TreeIteratorType *iterator() {
          return iterator_;
        }

        TreeType *node() {
          return node_;
        }

        std::pair<int, float> *float_dimension_split_values() {
          return float_dimension_split_values_;
        }

        std::pair<int, double> *double_dimension_split_values() {
          return double_dimension_split_values_;
        }

        std::pair<int, std::string> *bool_dimension_split_values() {
          return bool_dimension_split_values_;
        }

        std::pair<int, std::string> *char_dimension_split_values() {
          return char_dimension_split_values_;
        }

        std::pair<int, std::string> *int_dimension_split_values() {
          return int_dimension_split_values_;
        }

        int *current_dimension() {
          return current_dimension_;
        }

        int *best_dimension() {
          return best_dimension_;
        }

        std::string *best_dimension_type() {
          return best_dimension_type_;
        }

        double *best_improvement() {
          return best_improvement_;
        }

        void Init(ImpurityType *impurity_in,
                  TreeIteratorType *it_in,
                  TreeType *node_in,
                  std::pair<int, float> *float_dimension_split_values_in,
                  std::pair<int, double> *double_dimension_split_values_in,
                  std::pair<int, std::string> *bool_dimension_split_values_in,
                  std::pair<int, std::string> *char_dimension_split_values_in,
                  std::pair<int, std::string> *int_dimension_split_values_in,
                  int *current_dimension_in,
                  int *best_dimension_in,
                  std::string &best_dimension_type_in,
                  double *best_improvement_in) {

          impurity_ = impurity_in;
          iterator_ = it_in;
          node_ = node_in;
          float_dimension_split_values_ = float_dimension_split_values_in;
          double_dimension_split_values_ = double_dimension_split_values_in;
          bool_dimension_split_values_ = bool_dimension_split_values_in;
          char_dimension_split_values_ = char_dimension_split_values_in;
          int_dimension_split_values_ = int_dimension_split_values_in;
          current_dimension_ = current_dimension_in;
          best_dimension_ = best_dimension_in;
          best_dimension_type_ = &best_dimension_type_in;
          best_improvement_ = best_improvement_in;
        }
    };

  private:
    MapSplitValues split_values_;

  public:

    PartitionEachType(
      ImpurityType &impurity_in,
      TreeIteratorType &it,
      TreeType *node,
      std::pair<int, float> &float_dimension_split_values_in,
      std::pair<int, double> &double_dimension_split_values_in,
      std::pair<int, std::string> &bool_dimension_split_values_in,
      std::pair<int, std::string> &char_dimension_split_values_in,
      std::pair<int, std::string> &int_dimension_split_values_in,
      int *current_dimension_in,
      int *best_dimension_in,
      std::string &best_dimension_type,
      double *best_improvement_in) {

      split_values_.Init(&impurity_in,
                         &it,
                         node,
                         &float_dimension_split_values_in,
                         &double_dimension_split_values_in,
                         &bool_dimension_split_values_in,
                         &char_dimension_split_values_in,
                         &int_dimension_split_values_in,
                         current_dimension_in,
                         best_dimension_in,
                         best_dimension_type,
                         best_improvement_in);
    }

    template<typename T>
    void operator()(T) {
      PartitionDimension<T> partition_dimension;

      // Figure out how many dimensions this point has for this type.
      typename TreeIteratorType::Point_t dummy_point;
      split_values_.iterator()->table().get(0, &dummy_point);
      int length_of_current_type =
        TypeLength<is_dense, T>::Compute(dummy_point);

      for (int i = 0; i < length_of_current_type; i++) {
        partition_dimension.PartitionDimensionBranch(split_values_);
        (*(split_values_.current_dimension()))++;
      }
    }
};
};
};

#endif
