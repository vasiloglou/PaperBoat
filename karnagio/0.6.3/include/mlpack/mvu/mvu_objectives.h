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
#ifndef FL_LITE_MLPACK_MVU_OBJECTIVES_H_
#define FL_LITE_MLPACK_MVU_OBJECTIVES_H_

#include "boost/mpl/void.hpp"
#include "boost/mpl/has_key.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/if.hpp"
#include "fastlib/base/base.h"
#include "fastlib/dense/matrix.h"
#include "fastlib/dense/linear_algebra.h"
#include "fastlib/optimization/augmented_lagrangian/optimization_utils.h"
#include "mlpack/allkn/allkn.h"

namespace fl {
namespace ml {
class MVUArgs{
  public:
    typedef boost::mpl::void_ TableType;
    typedef boost::mpl::void_ ResultTableType;
    typedef std::vector<std::pair<index_t, double> >  NNPairsContainerType;
    typedef std::vector<index_t> IndexContainerType;
    typedef std::vector<double> DistanceContainerType;
};

template<typename TemplateOpts>
class MaxVariance {
  public:
    typedef typename TemplateOpts::NNPairsContainerType  NNPairsContainer_t;

    typedef typename TemplateOpts::DistanceContainerType DistanceContainer_t;

    typedef typename TemplateOpts::IndexContainerType IndexContainer_t;

    typedef typename TemplateOpts::ResultTableType ResultTable_t;

    typedef typename TemplateOpts::TableType Table_t;

    typedef typename TemplateOpts::TableType::Point_t Point_t;

    typedef typename ResultTable_t::CalcPrecision_t CalcPrecision_t;
    
    struct MVUOpts {
      index_t knns;
      index_t new_dimension;
      index_t num_of_points;
      bool auto_tune;
      CalcPrecision_t desired_feasibility_error;
      CalcPrecision_t infeasibility_tolerance;
      CalcPrecision_t grad_tolerance;
      IndexContainer_t *from_tree_neighbors;
      DistanceContainer_t *from_tree_distances;
    };

    void Init(MVUOpts &opts);
    void Destruct();
    /*inline*/
    void ComputeGradient(ResultTable_t &coordinates,
                         ResultTable_t *gradient);
    /*inline*/
    void ComputeObjective(ResultTable_t &coordinates,
                          CalcPrecision_t *objective);
    /*inline*/
    void ComputeFeasibilityError(ResultTable_t &coordinates,
                                 CalcPrecision_t *error);
    /*inline*/
    CalcPrecision_t ComputeLagrangian(ResultTable_t &coordinates);
    /*inline*/
    void UpdateLagrangeMult(ResultTable_t &coordinates);
    /*inline*/
    void Project(ResultTable_t *coordinates);
    /*inline*/
    void set_sigma(CalcPrecision_t sigma);
    /*inline*/
    bool IsDiverging(CalcPrecision_t objective);
    /*inline*/
    bool IsOptimizationOver(ResultTable_t &coordinates,
                            ResultTable_t &gradient, CalcPrecision_t step);
    /*inline*/
    bool IsIntermediateStepOver(ResultTable_t &coordinates,
                                ResultTable_t &gradient, CalcPrecision_t step);
    void GiveInitMatrix(ResultTable_t *init_data);
    /*inline*/
    index_t num_of_points();

  protected:
    index_t knns_;
    NNPairsContainer_t nearest_neighbor_pairs_;
    DistanceContainer_t nearest_distances_;
    fl::dense::Matrix<CalcPrecision_t, true> eq_lagrange_mult_;
    index_t num_of_nearest_pairs_;
    CalcPrecision_t sigma_;
    CalcPrecision_t sum_of_furthest_distances_;
    index_t num_of_points_;
    index_t new_dimension_;
    CalcPrecision_t infeasibility1_;
    CalcPrecision_t previous_infeasibility1_;
    CalcPrecision_t desired_feasibility_error_;
    CalcPrecision_t infeasibility_tolerance_;
    CalcPrecision_t sum_of_nearest_distances_;
    CalcPrecision_t grad_tolerance_;
};


template<typename TemplateOpts>
class MaxFurthestNeighbors  : public MaxVariance<TemplateOpts> {
  public:
    typedef typename MaxVariance<TemplateOpts>::NNPairsContainer_t NNPairsContainer_t;
    typedef typename MaxVariance<TemplateOpts>::DistanceContainer_t DistanceContainer_t;
    typedef typename MaxVariance<TemplateOpts>::IndexContainer_t IndexContainer_t;
    typedef typename MaxVariance<TemplateOpts>::ResultTable_t ResultTable_t;
    typedef typename MaxVariance<TemplateOpts>::CalcPrecision_t CalcPrecision_t;
    struct MFNOpts : public  MaxVariance<TemplateOpts>::MVUOpts {
      IndexContainer_t *from_tree_furhest_neighbors;
      DistanceContainer_t *from_tree_furthest_distances;
    };
    void Init(MFNOpts &opts);
    void Destruct();
    /*inline*/
    void ComputeGradient(ResultTable_t &coordinates,
                         ResultTable_t *gradient);
    /*inline*/
    void ComputeObjective(ResultTable_t &coordinates,
                          CalcPrecision_t *objective);
    /*inline*/
    CalcPrecision_t ComputeLagrangian(ResultTable_t &coordinates);
    /*inline*/
    bool IsDiverging(CalcPrecision_t objective);
    typedef MaxVariance<TemplateOpts> Parent_t;
  protected:
    index_t num_of_furthest_pairs_;
    NNPairsContainer_t furthest_neighbor_pairs_;
    DistanceContainer_t furthest_distances_;
};

class MaxVarianceUtils {
  public:
    template < typename IndexContainerType,
    typename DistanceContainerType,
    typename IndexIndexContainerType >
    static void ConsolidateNeighbors(
      IndexContainerType &from_tree_ind,
      DistanceContainerType  &from_tree_dist,
      index_t num_of_neighbors,
      index_t chosen_neighbors,
      IndexIndexContainerType *neighbor_pairs,
      DistanceContainerType *distances,
      index_t *num_of_pairs);
    template < typename IndexContainerType,
    typename DistanceContainerType >
    static void EstimateKnns(IndexContainerType &neares_neighbors,
                             DistanceContainerType &nearest_distances,
                             index_t maximum_knns,
                             index_t num_of_points,
                             index_t dimension,
                             index_t *optimum_knns);
};

template<typename T>
class DimensionalityReduction;

template<>
class DimensionalityReduction<boost::mpl::void_> {
  public:
    template<typename TableType1>
    struct Core {
      struct DefaultAllKNNMap : public AllKNArgs {
        typedef TableType1 QueryTableType;
        typedef TableType1 ReferenceTableType;
        typedef boost::mpl::int_<0>::type  KNmode;
      };
      struct DefaultAllKFNMap : public AllKNArgs  {
        typedef TableType1 QueryTableType;
        typedef TableType1 ReferenceTableType;
        typedef boost::mpl::int_<1>::type  KNmode;
      };
      typedef AllKN<DefaultAllKNNMap> DefaultAllKNN;
      typedef AllKN<DefaultAllKFNMap> DefaultAllKFN;
      
      class MVUArgs {
        public:
          typedef TableType1 TableType;
          typedef fl::dense::Matrix<double, false> ResultTableType;
          typedef std::vector<std::pair<index_t, double> >  NNPairsContainerType;
          typedef std::vector<index_t> IndexContainerType;
          typedef std::vector<double> DistanceContainerType;
      };
    
      typedef fl::ml::MaxVariance<MVUArgs> MVUObjective;
      typedef fl::ml::MaxFurthestNeighbors<MVUArgs> MFNUObjective;

      struct LbfgsTypeOptsMVU {
        typedef MVUObjective OptimizedFunctionType;
      };

      struct LbfgsTypeOptsMFNU {
        typedef MFNUObjective OptimizedFunctionType;
      };

      template<typename DataAccessType>
      static int Main(DataAccessType *data,
                      boost::program_options::variables_map &vm);
    };
    template<typename DataAccessType, typename BranchType>
    static int Main(DataAccessType *data,
                    const std::vector<std::string> &args);

    template<typename DataAccessType>
    static void Run(DataAccessType *data,
        const std::vector<std::string> &args);


};

}
} //namespace fl::ml


#endif //MVU_OBJECTIVES_H_
