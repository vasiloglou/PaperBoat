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
#ifndef FL_LITE_MLPACK_CLUSTERING_KMEANS_H
#define FL_LITE_MLPACK_CLUSTERING_KMEANS_H

#include "fastlib/base/base.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/math/fl_math.h"
#include "fastlib/math/gen_range.h"
#include <list>
#include "boost/program_options.hpp"
#include "boost/mpl/if.hpp"

class TestKMeans;

namespace fl {
namespace ml {
/**
 * @brief This is the tree based implementation of Dan Pelleg
 * and Andrew Moore's paper titled "Accelerating Exact
 * K-Means Algorithms with Geometric Reasoning".
 *
 * @author: Abhimanyu Aditya (abhimanyu@analytics1305.com)
 */


struct KMeansArgs {
  typedef boost::mpl::void_ TableType;
  typedef boost::mpl::void_ MetricType;
  typedef boost::mpl::void_ CentroidTableType;
};

template <typename KMeansMap>
class KMeans {

  public:
    typedef typename KMeansMap::TableType Table_t;
    typedef typename Table_t::Point_t Point_t;
    typedef typename Table_t::Tree_t Tree_t;
    typedef typename Tree_t::Bound_t Bound_t;
    typedef typename Point_t::CalcPrecision_t CalcPrecision_t;
    typedef typename KMeansMap::MetricType Metric_t;
    typedef typename KMeansMap::CentroidTableType CentroidTable_t;
    typedef typename KMeansMap::CentroidTableType::Point_t CentroidPoint_t;

    KMeans();
    ~KMeans();


    /**
    * If centroids are passed in they must already have been initialized.
    */
    void Init(const int k_in, Table_t* table_in,
              const CentroidPoint_t* centroids_in, const Metric_t* metric_in) ;

    void Init(const int k_in, Table_t* table_in, const Metric_t* metric_in);

    /**
    * If centroids are passed in they must already have been initialized.
    */
    void Init(const int k_in, Table_t* table_in, const CentroidPoint_t* centroids_in);

    void Init(const int k_in, Table_t* table_in) ;
    
    const Metric_t &metric() {
      return *metric_;
    }
    /**
    * This function keep's calling the assignment/update function
    * till the centroids do not change. Note: if by chance the randomly
    * initialized centroids happen to be the final centroids (extremely rare)
    * then instead of finishing in one iteration, it will take 2 iterations.
    * This is because 'point_cluster_assignments' is initially set to all 0.
    * And any change in this is used to indicated if 'something_changed_'. So
    * in the first iteration it will change even though the centroids got the
    * same value. In the next iteration it will terminate.
    */
    index_t RunKMeans(const std::string traversal_mode);

    /**
     * Copies the centroids into the passed table.
     */
    void GetCentroids(CentroidTable_t* centroids) ;
    /**
    * Copies memberships into the passed in vector.
    */
    template<typename ContainerType>
    void GetMemberships(ContainerType* memberships_out);
   
    /**
    * Copies memberships into the passed in vector.
    */
    template<typename ContainerType>
    void GetMembershipCounts(ContainerType* memberships_out);

	CalcPrecision_t GetDistortion();
    static void AssignInitialCentroids(const int k, CentroidPoint_t* points, Table_t* table);
    static void AssignInitialCentroids(const int k, CentroidTable_t* centroid_table, Table_t &table);
  
    template<typename WorkSpaceType, typename TableType>
    static void KMeansPlusPlus(const int k,
        double probability, 
        Metric_t &metric,
        WorkSpaceType *ws,
        const std::vector<std::string> &table_names,
        CentroidTable_t* centroid_table);

    CentroidPoint_t *curr_centroids() {
      return curr_centroids_;
    }
   
    void set_max_iterations(index_t max_iterations) {
      max_iterations_=max_iterations;
    }

    void set_min_cluster_movement_threshold(CalcPrecision_t threshold) {
      minimum_cluster_movement_threshold_ = threshold;
    }
    // returns the index of the closest centroid
    // if there is a tie returns the one with the lower index
    int GetClosestCentroid(const Point_t *point, CalcPrecision_t& distance_square_out);


  private:
    friend class TestKMeans;

    bool BreakOnMinimumClusterMovement();

    index_t NaiveKMeans();

    index_t TreeBasedKMeans();

    /**
    * Initializes the centroids to coincide with random points in
    * the dataset. Points passed are initilized in the function.
    */


    int GetClosestCentroid(const typename KMeans<KMeansMap>::Point_t *point, std::list<index_t>& blacklisted);

    /**
    * Returns true if 1 and only 1 centroid is closest to the node.
    * It puts the index of the first of the closest centroids into
    * centroid_idx i.e. even if the function returns false indicating
    * that there are 2 centroids which share smallest minimum distances
    * with the node, centroid_idx will contain one of these centroids
    * and particularly the one with the lower index;
    */
    bool GetClosestCentroid(Tree_t* node, int& centroid_idx, std::list<index_t>& blacklisted);

    /**
    * This function returns true if centroid_idx dominates all other
    * non-blacklisted centroids with repspect to this node. It also
    * blacklists all centroids which were previously not blacklisted
    * but are dominated by this centroid (centroid_idx).
    */
    bool Dominates(const int centroid_idx, Tree_t* node, std::list<index_t> &blacklisted);
    /**
     * The following structs are necessary for Dominates
     * They do some sort of metaprogramming so that we can differentiate
     * between ball tree and kdtree
     */
    struct BallKdNullaryMetaFunction1 {
      // We need to templatize over TableType, so that we can get lazy evaluation
      template<typename TableType>
      static bool Do(TableType *table,
                     Tree_t *node,
                     const Metric_t *metric,
                     CentroidPoint_t *curr_centroids,
                     index_t k,
                     const int centroid_idx,
                     std::list<index_t> &blacklisted);
    };

    struct BallKdNullaryMetaFunction2 {
      // We need to templatize over TableType, so that we can get lazy evaluation
      template<typename TableType>
      static bool Do(TableType *table,
                     Tree_t *node,
                     const Metric_t *metric,
                     CentroidPoint_t *curr_centroids,
                     index_t k,
                     const int centroid_idx,
                     std::list<index_t> &blacklisted);
    };

    void KMeansBaseCase(Tree_t* node, std::list<index_t> &blacklisted);

    void AssignPoints(typename Table_t::TreeIterator &point_it, std::list<index_t> &blacklisted);
    /**
    * Assigns a point to a centroid.
    */
    inline void AssignPointToCentroid(const Point_t* point,
                                      const int point_id, const int centroid_idx);

    void AssignAllPointsToCentroid(Tree_t* node, const int centroid_idx);

    void AssignUpdateStepRecursive(Tree_t* node, std::list<index_t>& blacklisted);

    /**
    * Avoids issues if Init(...) is called more than once
    * on the same object.
    */
    void Destroy();

    void PrivateInit(const int k_in, Table_t* table_in,
                     CentroidPoint_t* centroids_in, const Metric_t* metric_in);

    int k_;     // the number of clusters
    int max_iterations_;
    CalcPrecision_t minimum_cluster_movement_threshold_;
	  CalcPrecision_t final_distortion_;
    CentroidPoint_t* curr_centroids_;
    CentroidPoint_t* new_centroids_;
    std::vector<int> centroid_point_counts_;
    std::vector<int> point_centroid_assignments_;
    bool something_changed_;
    const Metric_t* metric_;
    Table_t *table_;
};

template<>
class KMeans<boost::mpl::void_> {
  public:
    template<typename TableType1>
    struct Core {
      template<typename MetricType1, typename CentroidTable>
      struct KMeansArgs {
        typedef TableType1 TableType;
        typedef MetricType1 MetricType;
        typedef CentroidTable CentroidTableType;
      };
      template<typename DataAccessType>
      static int Main(DataAccessType *data,
                      boost::program_options::variables_map &vm);

	  template<typename DataAccessType, typename KMeansType>
      static void AttachResults(DataAccessType *data, KMeansType* kmeans, TableType1& table,
		  std::string centroids_out, std::string memberships_out, std::string distortions_out, index_t k);
    template<typename DataAccessType, typename KMeansType>
      static void AttachResults( DataAccessType *data, KMeansType* kmeans, std::vector<std::string>& table_names,
        std::string centroids_out, std::vector<std::string> &memberships_out, 
        std::string &distortions_out, index_t k);
    };
    template <typename DataAccessType, typename BranchType>
    static int Main(DataAccessType *data, const std::vector<std::string> &args);
    
    template<typename DataAccessType>
    static void Run(DataAccessType *data,
        const std::vector<std::string> &args);


};
}
}


#endif
