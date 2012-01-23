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
#ifndef FL_LITE_FASTLIB_TREE_TABLE_H_
#define FL_LITE_FASTLIB_TREE_TABLE_H_

#include <string>
#include <sstream>
#include <vector>
#include <typeinfo>
#include "fastlib/base/base.h"
//#include "fastlib/tree/spacetree.h"
#include "boost/mpl/map.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/has_key.hpp"
#include "boost/mpl/bool.hpp"
#include "boost/mpl/front.hpp"
#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"
#include "boost/serialization/string.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/scoped_ptr.hpp"
#include "boost/serialization/split_member.hpp"
#include "boost/serialization/nvp.hpp"

/**
 *  @brief Forward declaration of tree
 */
 namespace fl {
   namespace tree {
     template<typename>
     class Tree;
   }
 }

 namespace table_set {
   template<typename>
   class General;

   template<typename>
   class Special;
 }

 namespace table_get {
   template<typename>
   class General;

   template<typename>
   class Special;
 }


class TreeTest;
/**
 *  @brief The first fastlib database like table
 *  Currently it supports one index, but it can easily
 *  be extended to multiple indices. It can sort the points
 *  or keep its original ordering. Currently it doesn't support
 *  dynamic updates. Table stands as a basic structure for all
 *  mlpack algorithms
 *
 */

namespace fl {
/**
 * @brief namespace table, contains table structures that look like database tables
 */
namespace table {
class TableArgs {
  public:
    typedef boost::mpl::void_  DatasetType ;
    typedef boost::mpl::bool_<true> SortPoints ;
};

template<typename TemplateMap>
class Table : boost::noncopyable {
  public:
    friend class ::TreeTest;
    typedef typename TemplateMap::TableArgs TableArgs_t;
    typedef Table<TemplateMap> Table_t;
    typedef typename TableArgs_t::DatasetType  Dataset_t;
    typedef typename Dataset_t::IsNativeMatrix_t IsNativeMatrix_t;
    typedef typename Dataset_t::IsMatrixOnly_t IsMatrixOnly_t;
    typedef typename Dataset_t::DenseBasicStorageType_t DenseBasicStorageType_t;
    typedef typename Dataset_t::ExportedPointCollection_t ExportedPointCollection_t;
    typedef typename Dataset_t::Point_t Point_t;
    typedef typename Dataset_t::CalcPrecision_t CalcPrecision_t;
    static const bool sort_points = TableArgs_t::SortPoints::value;


    typedef fl::tree::Tree<TemplateMap> Tree_t;
    // this must be an mpl::vector
    typedef typename Tree_t::Statistic_t Statistic_t;
    friend class fl::tree::Tree<TemplateMap>;
    // boost static assertions to make sure we don't have type mismatch
    BOOST_MPL_ASSERT_MSG(
      (boost::is_same < boost::mpl::bool_<sort_points>,
       boost::mpl::bool_<Tree_t::sort_points> >::value),
      Tree_and_table_have_different_sorting_flag,
      (Tree_t, Table_t));

    /**
     * @brief This struct should be passed to the function that builds
     *        the tree. It provides the following arguments:
     *        leaf_size: If this is non-negative then it will build a tree 
     *                   recursively and make sure that each leaf does not
     *                   have more than leaf_size number of points. The 
     *                   recursion will stop once the node size is less
     *                   than leaf_size
     *        level: If this is non-negative then it will build the tree
     *               recursively and make sure that the maximum tree level
     *               is no more than level. The recursion will stop once
     *               the current level is equal to level
     *        diameter: if this is non-negative then it will build a 
     *                    tree where each leaf will fit a vector
     *                    with with L2 norm  at most max_length. The 
     *                    recursion will stop once the maximum length
     *                    inside the node is less or equal to max_length
     *        Only one of the above attributes can be non-negative
     */
    template<typename MetricType>
    struct IndexArgs {
      public:
        IndexArgs() : leaf_size(-1), level(-1), diameter(-1) {
        }
        MetricType metric;
        index_t leaf_size;
        index_t level;
        CalcPrecision_t diameter;
        
        template<typename Archive>
        void serialize(Archive &ar, const unsigned int version);
    };

    class TreeIterator {
      public:
        typedef typename Dataset_t::Point_t Point_t;
        typedef typename Dataset_t::CalcPrecision_t CalcPrecision_t;
        TreeIterator(const Table_t &table, const Tree_t *node);
        TreeIterator(const Table_t &table, index_t begin, index_t count);
         bool HasNext() const ;
         void Next(Point_t *entry, index_t *point_id);
         void get(index_t i,  Point_t *entry);
         void get_id(index_t i, index_t *id);
         void RandomPick(Point_t *entry);
         void RandomPick(Point_t *entry, index_t *point_id);
         void Reset();
         index_t count() const;
         const Table_t& table() const ;
         index_t start() const;
         index_t end() const;

      private:
        const Table_t &table_;
        index_t start_;
        index_t end_;
        index_t ind_;
    };

    Table();
    Table(Dataset_t *data);
    ~Table();
    void Init(const std::string &file, const char* mode);
    template<typename DenseContainer>
    void Init(const DenseContainer &cont);
    template<typename ContainerType>
    void Init(ContainerType dense_dimensions,
              ContainerType sparse_dimensions,
              const index_t num_of_points);
    template<typename ContainerType> 
    void Init(const std::string &name,
              ContainerType dense_dimensions,
              ContainerType sparse_dimensions,
              const index_t num_of_points);
    void Destruct();
    void Save();
    // Carefull
    // CloneData will make a point by point copy
    // it will not copy index
    void CloneDataOnly(Table_t *table);
    /**
     * @brief Appends a table of the same type to the current table
     */
    void Append(Table_t &table);
    template<typename IndexArgsType>
    void SplitNode(IndexArgsType &args, Tree_t *node);

    template<typename IndexArgsType>
    void IndexData(IndexArgsType &args);
    void DeleteIndex();
    template<typename ContainerType>
    void ComputeNodesPerLevel(ContainerType *cont);
    template<typename ContainerType1, typename ContainerType2>
    void ComputeStatsPerLevel(ContainerType1 *nodes, ContainerType2 *diameters);
    /**
     * This version counts nodes separately from leafs. The reason why we call it UpTo
     * instead of Per  is because you can compute the number of leafs you would have 
     * if every node up to a specific level was considered a leaf
     */
    template<typename ContainerType1, typename ContainerType2>
    void ComputeStatsUpToLevel(ContainerType1 *nodes, ContainerType1 *leafs, 
        ContainerType2 *node_diameters, ContainerType2 *leaf_diameters);
    void RestrictTableToCentroidsUpToLevel(index_t up_to_level, Table_t *table);
    void RestrictTableToCentroidsUpToDiameter(CalcPrecision_t diameter, Table_t *table);
    void RestrictTableToSamplesUpToLevel(index_t level,
                                index_t num_of_samples_per_node,
                                Table_t *table);
    void RestrictTableToSamplesUpToDiameter(CalcPrecision_t diameter,
                                index_t num_of_samples_per_node, Table_t *table);
    std::vector<boost::shared_ptr<Table_t> > Split(int32 n_tables, const std::string &args);

    void LogTreeStats();
    bool is_indexed() const ;
    void PrintTree();
    Tree_t *get_tree() const;
    ExportedPointCollection_t get_point_collection() const;
    Dataset_t *data() ;
    const Dataset_t *data() const ;
    const std::string &filename() const;
    std::string &filename();
    index_t num_of_nodes() const;
    const std::string get_tree_metric();
    void get(index_t point_id, Point_t *entry) const;
    double get(index_t i, index_t j=0);
    void get(index_t i, std::vector<std::pair<index_t,double> > *point);
    void get(index_t i, signed char *meta1, double *meta2, int *meta3);
    std::pair<index_t, Point_t*> &get_cached_point();
    void set(index_t i, index_t j, double v);
    void SetAll(double value);
    void UpdatePlus(index_t i, index_t j, double value);
    void UpdateMul(index_t i, index_t j, double value);
    void push_back(std::string &point);
    void push_back(Point_t &point);
    index_t n_attributes() const;
    const std::vector<index_t> dense_sizes() const;
    const std::vector<index_t> sparse_sizes() const;
    index_t n_entries() const;
    index_t get_node_begin(Tree_t *node) const;
    index_t get_node_end(Tree_t *node) const ;
    index_t get_node_count(Tree_t *node) const;
    const typename Table<TemplateMap>::Tree_t::Bound_t &get_node_bound(const Tree_t *node) const;
    typename Table<TemplateMap>::Tree_t::Bound_t &get_node_bound(Tree_t *node);
    Tree_t *get_node_child(Tree_t *node, index_t i);
    index_t get_node_id(Tree_t *node);
    const index_t get_node_id(Tree_t *node) const;
    // Statistic_t& get_node_stat(Tree_t *node) const;
    // Statistic_t*& get_node_stat_ptr(Tree_t *node) const;
    bool node_is_leaf(Tree_t *node) const;
    Tree_t *get_node_left_child(Tree_t *node) const;
    Tree_t *get_node_right_child(Tree_t *node) const;
    TreeIterator get_node_iterator(Tree_t *node) const;
    TreeIterator get_node_iterator(index_t begin, index_t count) const;
    template<typename Archive>
    void save(Archive &ar, const unsigned int version) const ;
    template<typename Archive>
    void load(Archive &ar, const unsigned int version);

    BOOST_SERIALIZATION_SPLIT_MEMBER()

  private:
    Tree_t    *tree_;
    index_t num_of_nodes_;
    index_t leaf_size_;
    std::string metric_type_id_;
    boost::scoped_ptr<Dataset_t> data_;
    std::string filename_;
    std::vector<index_t> real_to_shuffled_;
    std::vector<index_t> shuffled_to_real_;
    std::vector<std::pair<Tree_t*, Point_t> > sampled_data_;
    std::pair<index_t, Point_t*> cached_point_;

    template<typename ContainerType>
    void ComputeNodesPerLevelRecusrion_(Tree_t *node,
                                        index_t level,
                                        ContainerType *cont);
    void RestrictTableToCentroidsUpToLevelRecursion_(Tree_t *node,
                                            index_t up_to_level, index_t current_level, 
                                            Table_t *table);

    template<typename ContainerType1, typename ContainerType2>
    void ComputeStatsPerLevelRecursion_(Tree_t *node,
                                        index_t level,
                                        ContainerType1 *nodes,
                                        ContainerType2 *diameters);
    template<typename ContainerType1, typename ContainerType2>
    void ComputeStatsPerLevelRecursion_(Tree_t *node,
                                        index_t level,
                                        ContainerType1 *nodes,
                                        ContainerType1 *leafs,
                                        ContainerType2 *node_diameters,
                                        ContainerType2 *leaf_diameters);
    template<typename ContainerType1, typename ContainerType2>
    void ComputeStatsUpToLevelRecursion_(Tree_t *node,
                                         index_t level,
                                         ContainerType1 *nodes,
                                         ContainerType1 *leafs,
                                         ContainerType2 *node_diameters,
                                         ContainerType2 *leaf_diameters);
    void RestrictTableToCentroidsUpToDiameterRecursion_(Tree_t *node,
                                            CalcPrecision_t diameter, 
                                            Table_t *table);
    void RestrictTableToSamplesUpToLevelRecursion_(Tree_t *node,
                                 index_t up_to_level, 
                                 index_t current_level, 
                                 index_t num_of_samples,
                                 Table_t *table);
    void RestrictTableToSamplesUpToDiameterRecursion_(Tree_t *node, 
                                 CalcPrecision_t diameter,
                                 index_t num_of_points, 
                                 Table_t *table);

  protected:
    void direct_get_(index_t point_id, Point_t *entry) const;
    index_t  direct_get_id_(index_t point_id) const;
  // friend help structure
  friend struct table_set::General<Table_t>;
  friend struct table_set::Special<Table_t>;
  friend struct table_get::General<Table_t>;
  friend struct table_get::Special<Table_t>;
};
}; // namesapce tree
}; // namespace fl


#endif
