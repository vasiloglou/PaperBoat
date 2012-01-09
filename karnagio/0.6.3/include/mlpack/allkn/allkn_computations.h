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
#ifndef FL_LITE_MLPACK_ALLKN_ALLKN_COMPUTATIONS_H_
#define FL_LITE_MLPACK_ALLKN_ALLKN_COMPUTATIONS_H_

#include <vector>
#include <deque>
#include "boost/static_assert.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/mpl/at.hpp"
#include "fastlib/base/base.h"
#include "fastlib/tree/abstract_statistic.h"
#include "fastlib/base/mpl.h"
#include "allkn_modes.h"

namespace fl {
/**
 * @brief namespace ml, contains all machine learning algorithms based on fastlib
 *
 *
 */
namespace ml {
template<typename TemplateMap>
class AllKN;


template<typename>
class SingleThreadAllKNTraversal;

template<template<typename> class>
struct DynamicArguments;

template<>
struct DynamicArguments<SingleThreadAllKNTraversal> {
public:
  DynamicArguments() {
    num_of_prunes_ = 0;
  }
  index_t &num_of_prunes() {
    return num_of_prunes_;
  }
private:
  index_t num_of_prunes_;
};


// we need this specialization because sometimes we wrap templates in mpl containers
// with the fl::WrapTemplate1 trick
template<>
struct DynamicArguments<fl::WrapTemplate1<SingleThreadAllKNTraversal>::type > :
      public  DynamicArguments<SingleThreadAllKNTraversal> {
};

/**
 * Traits are like properties. For example in the monochromatic case,
 * the base case needs to check that the query point is not the same
 * as the reference point. This is different for the bichromatic case
 * and this is an example of a trait. Other traits include things like
 * give an container to write results to, does the called function need
 * to allocate memory for the result container or simply write to it.
 */
template < typename TemplateMap,
bool IsMonochromatic,
bool Is1N,
AllKNMode mode >
class SingleThreadDualTreeAllKNBaseTrait;

struct ALLKNMPL {
  struct UpdateOperator1 {
    struct type {
      template < typename NeighborContainerType,
      typename DistanceContainerType,
      typename IndexContainerType >
      static inline void InitTempNeighbors(NeighborContainerType *neighbors,
                                           index_t ind,
                                           index_t offset,
                                           DistanceContainerType &distances,
                                           IndexContainerType &indices);
      template < typename Container1Type,
      typename Container2Type,
      typename CalcPrecisionType
      >
      static inline void Insert(Container1Type *indices,
                                Container2Type *distances,
                                index_t neighbor_position,
                                index_t neighbor,
                                index_t neighbor_rank,
                                CalcPrecisionType distance);
    };
  };

  struct UpdateOperator2 {
    struct type {
      template < typename NeighborContainerType,
      typename DistanceContainerType,
      typename IndexContainerType >
      static inline void InitTempNeighbors(NeighborContainerType *neighbors,
                                           index_t ind,
                                           index_t offset,
                                           DistanceContainerType &distances,
                                           IndexContainerType &indices);
      template < typename Container1Type,
      typename Container2Type,
      typename CalcPrecisionType
      >
      static inline void Insert(Container1Type *indices,
                                Container2Type *distances,
                                index_t p1,
                                index_t p2,
                                index_t neighbor_rank,
                                CalcPrecisionType distance);
    };
  };
};

/**
 * This is a necessary trait for
 * distinguishing all nn for binary
 * and non-binary trees
 */

template<typename TypeVector>
class SingleThreadAllKNTraversal {

  public:
    typedef SingleThreadAllKNTraversal<TypeVector> type;
    typedef typename boost::mpl::at_c<TypeVector, 0>::type AllKNArgs;
    static const bool IsQueryBinary = boost::mpl::at_c<TypeVector, 1>::type::value;
    static const bool IsReferenceBinary = boost::mpl::at_c<TypeVector, 2>::type::value;
    static const bool Is1N = boost::mpl::at_c<TypeVector, 3>::type::value;
    static const bool IsMonochromatic = boost::mpl::at_c<TypeVector, 4>::type::value;
    static const AllKNMode mode = static_cast<AllKNMode>(
                                    boost::mpl::at_c<TypeVector, 5>::type::value);
    typedef AllKN<AllKNArgs> AllKN_t;
    typedef typename AllKN_t::QueryTree_t QueryTree_t;
    typedef typename QueryTree_t::Point_t QueryPoint_t;
    typedef typename AllKN_t::ReferenceTree_t ReferenceTree_t;
    typedef typename AllKN_t::QueryTable_t QueryTable_t;
    typedef typename ReferenceTree_t::Point_t ReferencePoint_t;
    typedef typename AllKN_t::ReferenceTable_t ReferenceTable_t;
    typedef typename AllKN_t::CalcPrecision_t CalcPrecision_t;
    typedef  SingleThreadDualTreeAllKNBaseTrait<AllKNArgs, true, Is1N, mode>
    SingleThreadDualTreeAllKNBaseTraitMono_t;
    typedef SingleThreadDualTreeAllKNBaseTrait<AllKNArgs, false, Is1N, mode>
    SingleThreadDualTreeAllKNBaseTraitBi_t;
    struct QueryBinaryReferenceBinary;
    // This implementation is only for Binary trees
    BOOST_STATIC_ASSERT(IsQueryBinary && IsReferenceBinary);

    typedef typename boost::mpl::if_c < IsQueryBinary && IsReferenceBinary,
    QueryBinaryReferenceBinary, boost::mpl::void_ >::type Engine;

    template<typename ArgsType>
    static inline void ComputeDualNeighborsRecursion(ArgsType *args) {
      Engine::ComputeDualNeighborsRecursion(args);
    }
    template<typename ArgsType>
    static inline void ComputeNaive(ArgsType *args) {
      Engine::ComputeNaive(args);
    }
    template<typename ArgsType>
    static inline void ComputeSingleNeighborsRecursion(ArgsType *args) {
      Engine::ComputeSingleNeighborsRecursion(args);
    }
    template<typename ArgsType>
    static inline void ComputeAllSingleNeighborsRecursion(ArgsType *args) {
      Engine::ComputeAllSingleNeighborsRecursion(args);
    }
    template<typename ArgsType>
    static inline void ComputeDualNeighborsProgressive(std::deque<ArgsType> *trace,
        bool *done,
        index_t *points_finished) {
      Engine::ComputeDualNeighborsProgressive(trace, done, points_finished);
    }

    template<typename ArgsType>
    static inline void InitSingleNeighborsProgressive(std::vector<ArgsType> *trace) {
      Engine::InitSingleNeighborsProgressive(trace);
    }

    template<typename ArgsType>
    static inline void ComputeAllSingleNeighborsProgressive(
      std::vector<std::vector<ArgsType> > *trace,
      bool *done,
      index_t *points_finished) {
      Engine::ComputeAllSingleNeighborsProgressive(trace, done, points_finished);
    }

    template<typename ArgsType>
    static inline void ComputeSingleNeighborsProgressive(std::vector<ArgsType> *trace) {
      Engine::ComputeSingleNeighborsProgressive(trace);
    }

    struct QueryBinaryReferenceBinary {
      /**
       * @brief Routines for exact full blown  neighbors
       */
      template<typename ArgsType>
      static inline void ComputeDualNeighborsRecursion(ArgsType *args);

      template<typename ArgsType>
      static inline void ComputeNaive(ArgsType *args);

      template<typename ArgsType>
      static inline void ComputeAllSingleNeighborsRecursion(ArgsType *args);

      template<typename ArgsType>
      static inline void ComputeSingleNeighborsRecursion(ArgsType *args);

      /**
       *  @brief Routines for progressive computations for neighbors
       */
      template<typename ArgsType>
      static inline void ComputeDualNeighborsProgressive(std::deque<ArgsType> *trace,
          bool *done,
          index_t *points_finished);

      template<typename ArgsType>
      static inline void InitSingleNeighborsProgressive(std::vector<ArgsType> *trace);

      template<typename ArgsType>
      static inline void ComputeAllSingleNeighborsProgressive(
        std::vector<std::vector<ArgsType> > *trace,
        bool *done,
        index_t *points_finished);

      template<typename ArgsType>
      static inline void ComputeSingleNeighborsProgressive(std::vector<ArgsType> *trace);

    }; // struct QueryBinaryReferenceBinary
};


/**
 * This is an optimizer for the case
 * of 1 nearest neighbor
 */
template < typename AllKNArgs,
bool IsMonochromatic,
bool Is1N,
AllKNMode mode >
class SingleThreadDualTreeAllKNBaseTrait {
  public:
    typedef AllKN<AllKNArgs> AllKN_t;
    typedef typename AllKN_t::CalcPrecision_t CalcPrecision_t;
    typedef typename AllKN_t::QueryTable_t QueryTable_t;
    typedef typename QueryTable_t::Point_t QueryPoint_t;
    typedef typename AllKN_t::ReferenceTable_t ReferenceTable_t;
    typedef typename ReferenceTable_t::Point_t ReferencePoint_t;
    typedef typename AllKN_t::QueryTree_t QueryTree_t;
    typedef typename AllKN_t::ReferenceTree_t Reference_t;

    /**
     * @note: although nearest_distances and nearest_indices
     *        are going to be modified inside base
     *        we pass them as references and not as pointers
     *        because they are going to be traversed many times
     *        and this would decrease speed
     */

    template<typename ArgsType>
    static inline void ComputeBaseCase(ArgsType *args);
};

template<typename Precision>
class NeighborStatistic : public fl::tree::AbstractStatistic {
  public:
    typedef Precision CalcPrecision_t;

    NeighborStatistic() {
    }

    ~NeighborStatistic() {
    }

    template<typename TreeIterator>
    void Init(TreeIterator &it) {
    }

    template<typename TreeIterator>
    void Init(TreeIterator &it,
              const NeighborStatistic& left_stat,
              const NeighborStatistic& right_stat) {

    }

    void set_dist_so_far(CalcPrecision_t distance) {
      dist_so_far_ = distance;
    }

    CalcPrecision_t dist_so_far() const {
      return dist_so_far_;
    }

  private:
    CalcPrecision_t dist_so_far_;
};

} // namespace ml
}   // namespace fl


#endif
