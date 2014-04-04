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
 * @file allknn.h
 *
 * Defines AllNN class to perform all-nearest-neighbors on two specified
 * data sets. This class contains several nearest neighbor algorithms such as k-nearest
 * neighbors, range neighbors, furthest neighbors, approximate neighbors. 
 */

#ifndef FL_LITE_MLPACK_ALLKN_ALLKN_H
#define FL_LITE_MLPACK_ALLKN_ALLKN_H

#include <vector>
#include <string>
#include "fastlib/base/mpl.h"
#include "fastlib/base/base.h"
#include "fastlib/table/table.h"
#include "boost/mpl/if.hpp"
#include "boost/mpl/int.hpp"
#include "boost/mpl/insert.hpp"
#include "boost/mpl/assert.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/type_traits/is_integral.hpp"
#include "allkn_mpl_defs.h"
#include "allkn_modes.h"
#include "allkn_computations.h"
#include "boost/shared_ptr.hpp"
#include "boost/program_options.hpp"

class TestAllkN;
namespace fl {
/**
 * @brief namespace ml, contains all machine learning algorithms
 */
namespace ml {

  template<template <typename> class>
  struct DynamicArguments;
  /**
   * @brief struct AllKArgs contains all the type definitions that are necessary
   *                         for instantiating an AllKN engine
   *
   */
  struct AllKNArgs {
    /** The Table you are going to use for query data */
    class QueryTableType;
    /** The Table you are going to use for reference data */
    class ReferenceTableType;
    /**
     * @brief
     * @verbatim
        KNMode is a boost::mpl::int_<> 
        constant that defines the mode of  AllKN, 
        it can be:
          boost::mpl::int_<0> 
          for computing nearest neighbors
          boost::mpl::int_<1> 
          for computing furthest neighbors
       @endverbatim   
     */
    typedef boost::mpl::int_<0>::type KNmode;
    /**
     * @brief 
     * AllKNTraversal is a templated class that defines functions necessary for 
     * traversing the tree for nearest neighbors. We have implemented 
     * SingleThreadAllKNTraversal and MultiThreadAllKNTraversal for multithreaded
     * environments
     */
    typedef fl::WrapTemplate1<SingleThreadAllKNTraversal>  AllKNTraversal;
  };
  
  
  /**
  * @brief 
  * @code
  *   template <typename ArgMap> class AllKN 
  * @endcode
  *   This is the core object for running many different types of nearest neighbor
  */
  template<typename ArgMap>
  class AllKN {
    public:
      /** Typedefs for practical reasons */
      typedef  AllKN<ArgMap> AllKN_t;
      typedef typename ArgMap::QueryTableType QueryTable_t;
  
      typedef typename  ArgMap::ReferenceTableType ReferenceTable_t;
  
      typedef typename QueryTable_t::Point_t QueryPoint_t;
      typedef typename ReferenceTable_t::Point_t ReferencePoint_t;
      typedef ReferencePoint_t Point_t;
  
      typedef typename Point_t::CalcPrecision_t CalcPrecision_t;
      typedef typename QueryTable_t::Tree_t QueryTree_t;
      typedef typename ReferenceTable_t::Tree_t ReferenceTree_t;
  
      static const AllKNMode KNmode = (AllKNMode)ArgMap::KNmode::value;
  
  
      typedef  typename ArgMap::AllKNTraversal AllKNTraversal;
  
      /**
       * @code
       *   template<typename> struct DynamicArguments;
       * @endcode
       * @brief 
       *        is a utility struct for passing information in recursice calls of nearest
       *        neighbor. This struct contains information that gets updated as nearest 
       *        neighbor is executed. It is a part of a biger class TreeArguments
       */
      typedef DynamicArguments<ArgMap::AllKNTraversal::template type> DynamicArguments_t;
 
      /**
       * @code
       *  template<typename MetricType,
       *           typename NeighborMetricType,
       *           typename ContainerDistType,
       *           typename ContainerIndType> 
       *   struct TreeArguments;
       * @endcode
       * @brief 
       *  This is a utility class holding all the information nearest neighbor needs 
       *  in intermediate calls
       *
       */
      template<typename MetricType,
        typename NeighborMethodType,
        typename ContainerDistType,
        typename ContainerIndType>
      struct TreeArguments {
        public:
          static const bool IS_RANGE_NEIGHBORS =
            !boost::is_integral<NeighborMethodType>::value;
    
          TreeArguments();
    
          TreeArguments(QueryTree_t* query_node1,
                        QueryTable_t* query_table1,
                        ReferenceTree_t* reference_node1,
                        ReferenceTable_t* reference_table1,
                        ContainerDistType* neighbor_distances1,
                        ContainerIndType*  neighbor_indices1,
                        std::vector<NeighborStatistic<CalcPrecision_t> > *stat,
                        const NeighborMethodType kns,
                        const CalcPrecision_t lower_bound_distance1,
                        const MetricType *metric,
                        DynamicArguments_t *info);
    
          TreeArguments(QueryPoint_t &query_point1,
                        index_t query_point_id1,
                        ReferenceTree_t* reference_node1,
                        const ReferenceTable_t* reference_table1,
                        ContainerDistType* neighbor_distances1,
                        ContainerIndType*  neighbor_indices1,
                        const NeighborMethodType kns1,
                        const CalcPrecision_t min_dist_so_far,
                        const MetricType *metric,
                        DynamicArguments_t *info
                       );
    
    
          TreeArguments(const TreeArguments &other) ;
    
    
	   inline typename AllKN<ArgMap>::template TreeArguments<MetricType,
			NeighborMethodType,
			ContainerDistType,
			ContainerIndType> &operator=(const TreeArguments &other);

          inline void CopyNonSharedOnly(const TreeArguments &other);
    
          inline QueryTree_t* &query_node();
    
          inline QueryPoint_t* &query_point();
    
          inline const QueryTable_t* &query_table();
    
          inline index_t &query_point_id();
    
          inline ReferenceTree_t* &reference_node();
    
          inline const ReferenceTable_t* &reference_table();
    
          inline ContainerDistType* &neighbor_distances();
    
          inline ContainerIndType*  &neighbor_indices();

          inline std::vector<NeighborStatistic<CalcPrecision_t> > *&stat();
    
          inline NeighborMethodType &kns();
    
          inline CalcPrecision_t &lower_bound_distance();
    
          inline CalcPrecision_t &bound_distance();
    
          inline CalcPrecision_t &min_dist_so_far();
    
          inline CalcPrecision_t &dist_so_far();
    
          inline const MetricType* &metric();
    
          inline DynamicArguments_t* &info();
    
        private:
          union {
            QueryTree_t* query_node_;
            QueryPoint_t *query_point_;
          };
          const QueryTable_t *query_table_;
          index_t query_point_id_;
    
          ReferenceTree_t* reference_node_;
          const ReferenceTable_t *reference_table_;
          ContainerDistType* neighbor_distances_;
          ContainerIndType*  neighbor_indices_;
          std::vector<NeighborStatistic<CalcPrecision_t> > *stat_;
          NeighborMethodType  kns_;
          boost::shared_ptr<CalcPrecision_t> bound_distance_;
          boost::shared_ptr<CalcPrecision_t> dist_so_far_;
          const MetricType *metric_;
          DynamicArguments_t *info_;
      };
      friend class TestAllkN;
  
  
    public:
 
      /**
       * @brief
       * @code
       *   template<typename MetricType,
       *            typename NeighborMethodType,
       *            typename ContainerDistType,
       *            typename ContainerIndType>
       *    class iterator;
       * @endcode
       *  This is a very powerful class that gives the opportunity to run 
       *  nearest neighbors progressively. It behaves like an "STL iterator"
       *  on an algorithm. So every operator++() call will improve the accuracy of
       *  the nearest neighbors.
       *  @param MetricType  the metric you want to use for nearest neighbors
       *  @param NeighborMethodType  boost::mpl::int_<0> is for nearest neighbors
       *                              boost::mpl::int_<1> is for furthest neighbors
       *  @param ContainerDistType  the container to save distances (typically std::vector<double>)
       *  @param ContainerIndType   the container to save indices of neighbors (typically std::vector<int>)
       */ 
      template<typename MetricType,
               typename NeighborMethodType,
               typename ContainerDistType,
               typename ContainerIndType>
      class iterator {
        public:
          typedef  TreeArguments<MetricType,
            NeighborMethodType,
            ContainerDistType,
            ContainerIndType > Args_t;
          /**  Default Constructor */ 
          iterator();
          /** Copy Constructor */
          iterator(const iterator &other);
          
	   /** Assignment operator */
          typename AllKN<ArgMap>::template iterator< MetricType, NeighborMethodType,
				ContainerDistType, ContainerIndType > 
          &operator=(const iterator &other);
          
 	   /** Destructor */
          ~iterator();
          /**
           *  @brief
           *  @code 
           *    void Init(AllKN<ArgMap> *allkn,
           *              const std::string &algorithm,
           *              const MetricType &metric,
           *              NeighborMethodType kns,
           *              ContainerDistType *dist,
           *              ContainerIndType *ind);
           *  @endcode            
           *  @param allkn is a pointer to an AllKN<ArgMap> engine
           *  @param algorithm A string with the mode of the algorithm, 
           *              it can be "single", "dual"               
           *  @param metric is a class that implements distance functions
           *  @param kns if it is an integer then it finds the k nearest neighbors
           *         if it is a double then it computes range neighbors 
           *  @param dist is a pointer to a container that contains the distances to the points
           *  @param ind is a pointer to a container that contains the indices to the nearest points
           */
          void Init(AllKN<ArgMap> *allkn,
                    const std::string &algorithm,
                    const MetricType &metric,
                    NeighborMethodType kns,
                    ContainerDistType *dist,
                    ContainerIndType *ind);
          
          bool operator==(const iterator &other) const;

          bool operator!=(const iterator &other) const;
          typename ContainerDistType::value_type operator*();
          
	   typename AllKN<ArgMap>::template iterator< MetricType, NeighborMethodType,
				ContainerDistType, ContainerIndType > 
          &operator++();
				
 
        private:
          AllKN *allkn_;
          union {
            std::vector<std::vector<Args_t> > *single_tree_stacks_;
            std::deque<Args_t> *dual_tree_stack_;
          };
          bool is_dual_tree_;
          NeighborMethodType kns_;
          bool done_;
          index_t points_finished_;
          ContainerDistType *dist_;
          ContainerIndType *ind_;
          const MetricType *metric_;
          index_t stage_;
      };
      template<typename T>
      struct IsSTLPairTrait : public boost::false_type {
  
      };
  
      template<typename T, typename U>
      struct IsSTLPairTrait<std::pair<T, U> > : public boost::true_type {
      };
  
      struct InitContainerWithPairs {
        struct type {
          template<typename ContainerType, typename Type>
          static inline void Init(ContainerType *c, Type size) {
  
          }
        };
      };
  
      struct InitContainerWithoutPairs {
        struct type {
          template<typename ContainerType>
          static inline void Init(ContainerType &c, index_t size) {
            c->resize(size);
          }
        };
      };
  
  
  
      /**
       * @brief
       * Constructors are generally very simple, most of the work is done by Init().  This is only
       * responsible for ensuring that the object is ready to be destroyed safely.
       */
      AllKN();
      ~AllKN();
  
      /**
       * @brief
       * Computes the minimum squared distance between the bounding boxes
       * of two nodes
       */
      template<typename MetricType>
      inline CalcPrecision_t MinNodeDistSq(const MetricType &metric, QueryTree_t *query_node,
                                           ReferenceTree_t* reference_node);
      /**
       * @brief
       * Computes the maximum squared distance between the bounding boxes
       * of two nodes
       */
      template<typename MetricType>
      inline CalcPrecision_t MaxNodeDistSq(const MetricType &metric, QueryTree_t *query_node,
                                           ReferenceTree_t* reference_node);
      /**
       * @brief
       * Computes the minimum squared distances between a point and a
       * node's bounding box
       */
      template<typename MetricType>
      inline CalcPrecision_t MinPointNodeDistSq(const MetricType &metric,
          const QueryPoint_t& query_point,
          ReferenceTree_t* reference_node);
  
      /**
       * @brief
       * Computes the maximum squared distances between a point and a
       * node's bounding box
       */
      template<typename MetricType>
      inline CalcPrecision_t MaxPointNodeDistSq(const MetricType &metric,
          const QueryPoint_t& query_point,
          ReferenceTree_t* reference_node);
  
  
      void Init(ReferenceTable_t* const reference_table, QueryTable_t* const query_table);
      /**
       * @brief Resets the distance so far
       */
      template<typename PrecisionType>
      void ResetStatistics(PrecisionType value);
  
      /** @brief
        * Computes the nearest neighbors and stores them in *results
        * if NeighborType is integral then it computes k-nearest/furthest neighbors
        * if NeighborType is double or float then it computes the range neighbors
        */
      // Containers must be properly initialized
      template < typename MetricType,
      typename NeighborMethodType,
      typename ContainerDistType,
      typename ContainerIndType >
      void ComputeNeighbors(const std::string &traversal_mode,
                            const MetricType &metric,
                            NeighborMethodType kns,
                            ContainerDistType* neighbor_distances,
                            ContainerIndType* neighbor_indices);
  
      /**
       * @brief
       * Computes the nearest neighbors and stores them in *results
       * In this version you get the chance to pass more parameters online
       * if NeighborType is integral then it computes k-nearest/furthest neighbors
       * if NeighborType is double or float then it computes the range neighbors
       */
      // Containers must be properly initialized
      template < typename MetricType,
      typename NeighborMethodType,
      typename ContainerDistType,
      typename ContainerIndType >
      void ComputeNeighbors(const std::string &traversal_mode,
                            const MetricType &metric,
                            NeighborMethodType kns,
                            DynamicArguments_t *info,
                            ContainerDistType* neighbor_distances,
                            ContainerIndType* neighbor_indices);
      /**
       * @brief
       * We also offer this single query functionality, it returns
       * the distances and the indices of the neighbors
       * if NeighborType is integral then it computes k-nearest/furthest neighbors
       * if NeighborType is double or float then it computes the range neighbors
       */
      template < typename MetricType,
      typename NeighborMethodType,
      typename ContainerDistType,
      typename ContainerIndType >
      void ComputeNeighbors(typename QueryTable_t::Point_t &point,
                            const MetricType &metric,
                            NeighborMethodType kns,
                            ContainerDistType* neighbor_distances,
                            ContainerIndType* neighbor_indices);
      /**
       * @brief
       * We also offer this single query functionality, same as before
       * but it actually returns the points too.
       * if NeighborType is integral then it computes k-nearest/furthest neighbors
       * if NeighborType is double or float then it computes the range neighbors
       */
      template < typename MetricType,
      typename PointContainerType,
      typename NeighborMethodType, typename ContainerDistType >
      void ComputeNeighbors(typename QueryTable_t::Point_t &point,
                            const MetricType &metric,
                            NeighborMethodType kns,
                            ContainerDistType* neighbor_distances,
                            PointContainerType* neighbor_points);
  
 
      index_t num_of_prunes() const {
        return number_of_prunes_;
      }
  
      /**
       * @brief returns a pointer to the query_table
       */
      QueryTable_t * query_table() const  {
        return query_table_;
      }
  
      /**
       * @brief returns a pointer to the refernece_table
       */
      ReferenceTable_t * reference_table() const  {
        return reference_table_;
      }
  
    private:
      // These will store our data sets.
      QueryTable_t *query_table_;
      ReferenceTable_t *reference_table_;
      //AllKNTraversal allkn_traversal_;
      bool monochromatic_flag_;
      // The total number of prunes.
      index_t number_of_prunes_;
      std::vector<NeighborStatistic<CalcPrecision_t> > stat_; 
      template<typename TreeType, typename TableType, typename PrecisionType>
      void ResetStatisticsRecursion_(TreeType *node, TableType *table,
                                     PrecisionType value);
  
  }; //class AllkNN
  
  // specialization that only contains the Main functions
  template<>
  class AllKN<boost::mpl::void_> {
    public:
      template<typename TableType>
      struct Core {
        struct MySqrt {
          MySqrt() {
            sum=0;
          }
          template<typename T>
          void operator()(T *t) {
            sum+=*t;
            *t=sqrt((typename TableType::CalcPrecision_t)*t);  
          }
          double sum;
        };
        struct Norm {
          Norm(double s) {
            s_=s;
          }
          template<typename T>
          void operator()(T *t) {
            *t/=s_;  
          }
          double s_;
        };

        struct DefaultAllKNNMap : public AllKNArgs {
          typedef TableType QueryTableType;
          typedef TableType ReferenceTableType;
          typedef boost::mpl::int_<0>::type  KNmode;
        };
        struct DefaultAllKFNMap : public AllKNArgs  {
          typedef TableType QueryTableType;
          typedef TableType ReferenceTableType;
          typedef boost::mpl::int_<1>::type  KNmode;
        };
        typedef AllKN<DefaultAllKNNMap> DefaultAllKNN;
        typedef AllKN<DefaultAllKFNMap> DefaultAllKFN;
        template<typename DataAccessType>
        static int Main(DataAccessType *data,
                        boost::program_options::variables_map &vm);
      };
      /**
       * @brief This is the main driver function that the user has to
       *        call.
       */
      template<typename DataAccessType, typename BranchType> 
      static int Main(DataAccessType *data,
                      const std::vector<std::string> &args);

      template<typename DataAccessType>
      static void Run(DataAccessType *data,
        const std::vector<std::string> &args);

  };

}; // namespace ml
}; // namespace fl

#endif
// end inclusion guards
