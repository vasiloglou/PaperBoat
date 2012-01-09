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
#ifndef FL_LITE_FASTLIB_TREE_SPACETREE_H
#define FL_LITE_FASTLIB_TREE_SPACETREE_H

//#include <omp.h>
#include <limits>
#include <deque>
#include <iostream>
#include "fastlib/base/base.h"
#include "abstract_statistic.h"
#include "fastlib/metric_kernel/lmetric.h"
#include "spacetree_private.h"
#include "boost/mpl/if.hpp"
#include "boost/mpl/bool.hpp"
#include "boost/mpl/has_key.hpp"
#include "boost/mpl/assert.hpp"
#include "boost/type_traits.hpp"
#include "boost/serialization/nvp.hpp"


namespace fl {
namespace table {
template<typename>
class Table;
};
/**
 * @brief namespace tree, contains all the structures for building multidimensional trees
 *
 */
namespace tree {
class TreeArgs {
  public:
    typedef boost::mpl::void_ TreeSpecType;
    typedef AbstractStatistic StatisticType;
    typedef boost::mpl::bool_<true> SortPoints;
    typedef boost::mpl::bool_<true> StoreLevel;
    typedef boost::mpl::void_ BoundType;
};
/**
 *  @brief class Tree provides the core datastructure for building trees
 *         It is strongly connected with class Table, since it provides
 *         indexing for multidimensional data. It gets a boost::mpl::map
 *         as a template argument, that contains all the specs necessary
 *         you can find more information about the design inside dokuwiki
 */
template<typename TemplateMap>
class Tree  {
  public:
    typedef Tree<TemplateMap>  Tree_t;

    typedef typename TemplateMap::TreeArgs TreeMap_t;;

    typedef typename TreeMap_t::TreeSpecType TreeSpec_t;

    static const bool sort_points = TreeMap_t::SortPoints::value;

    static const bool store_level = TreeMap_t::StoreLevel::value;

    static const bool IsBinary = TreeSpec_t::is_binary;

    typedef class fl::table::Table<TemplateMap> Table_t;

    typedef typename Table_t::Point_t Point_t;

    typedef typename TreeMap_t::StatisticType  Statistic_t;

    typedef typename TreeMap_t::BoundType Bound_t;

    friend class fl::table::Table<TemplateMap>;
    friend class fl_private::LeftRightAccessors<IsBinary>;
    //friend class fl_private::ComputeStatisticHelper<IsBinary>;
    template<typename MetricType, typename TreeIteratorType, typename TreeType>
    friend bool Partition(MetricType &metric,
                          TreeIteratorType &it,
                          TreeType *node,
                          std::deque<bool> *membership);
    /**
     * @brief A simple constructor
     */
    Tree();
    /**
     * @brief deleted recursively the node and the corresponding subtree
     */
    ~Tree();
    /** @brief Build the tree given a table and the limit on the
     *         leaf size.
     */
    template<int mode, typename IndexArgsType>
    static Tree_t *BuildTree(Table_t &table,
                             IndexArgsType &args,
                             index_t *num_of_nodes);

    template<typename MetricType>
    static inline index_t MatrixPartition(MetricType &metric,
                                          Table_t &table,
                                          index_t first,
                                          index_t count,
                                          Tree_t *new_node,
                                          std::deque<bool> &left_membership);

    /**
     *  @brief SplitTree splits the nodes of the tree.
     *         mode: if it is 0 it stops splitting according to
     *               the leaf_size
     *               if it is 1 it stops splitting according to
     *               the level
     *               if it is 2 it stops splitting according to
     *               the diameter of the node
     */
    template<int mode, typename IndexArgsType>
    static void SplitTree(IndexArgsType &args,
                          Table_t &table,
                          Tree_t *node,
                          index_t level,
                          index_t *node_counter);
    /**
     * @brief inherited from StoreLevelTrait
     */

    inline void set_level(index_t level_in);
    /**
     * @brief returns the level
     */
    inline void level(index_t *level_out) const ;
    /**
     *  @brief prints the current level
     */
    inline void PrintLevel(const Tree_t *root_node) const;

    /**
     *  @brief serializes a tree
     */
    template<typename Archive>
    void serialize(Archive &ar,
                   const unsigned int version);

  private:
    /**
     * @brief In some trees it is necessary to store the level
     *
     */
    fl_private::StoreLevelTrait<store_level> store_level_;
    /**
     * @brief Every node must have a bounding box, also known as Bound_t
     */
    Bound_t bound_;
    /**
     * @brief A unique node id
     */
    index_t node_id_;
    /**
     * @brief The beginning index of the tree. To-Do: Put this is a
     *         union.
     */
    index_t begin_;
    /**
     * @brief The number of points in this tree.
     */
    index_t count_;
    /**
     * @brief The computed statistic for the node.
     * We don't use that any more since the statistic is stored on the 
     * algorithm so that we can share data between threads/algorithms
     */
    // Statistic_t *stat_;
    /**
     * @brief The container that holds the child nodes. Must support
     *         the bracket operator[].
     */
    typedef typename fl_private::ChildTrait<IsBinary, Tree_t>::Container
    ChildContainer_t;
    ChildContainer_t children_;

    /** Member functions */
    /**
     * @brief
     */
    inline const Tree_t *FindByBeginCountHelper(const ChildContainer_t &children_in,
        index_t begin_q) const;
    /**
     * @brief
     */
    inline Tree_t *FindByBeginCountHelper_(const ChildContainer_t &children_in,
                                           index_t begin_q);
    /**
     * @brief prints the tree recursively
     */
    inline void PrintHelper_(const Tree_t *root_node) const;

    inline void Init(index_t begin_in, index_t count_in, index_t node_id);

    inline void Init(index_t begin_in, index_t count_in);

    /** @brief Find a node in this tree by its begin and count.
     *
     * Every node is uniquely identified by these two numbers.
     * This is useful for communicating position over the network,
     * when pointers would be invalid.
     *
     * @param begin_q the begin() of the node to find
     * @param count_q the count() of the node to find
     * @return the found node, or NULL
     */
    inline const Tree_t* FindByBeginCount(index_t begin_q, index_t count_q) const;

    /** @brief Find a node in this tree by its begin and count (const).
     *
     * Every node is uniquely identified by these two numbers.
     * This is useful for communicating position over the network,
     * when pointers would be invalid.
     *
     * @param begin_q the begin() of the node to find
     * @param count_q the count() of the node to find
     * @return the found node, or NULL
     */
    inline Tree_t* FindByBeginCount(index_t begin_q, index_t count_q);

    inline ChildContainer_t &children();

    inline const ChildContainer_t &children() const;

    void set_children(Tree_t *new_child_in, index_t new_child_index);

    inline void compute_statistic(Table_t &table);

    inline Bound_t& bound();

    inline const Bound_t& bound() const;

    inline index_t& node_id();

    inline const index_t& node_id() const ; 
    // inline Statistic_t& stat();

    // inline const Statistic_t& stat() const;

    // inline Statistic_t* &stat_ptr();

    // inline const Statistic_t* &stat_ptr() const;

    inline bool is_leaf() const ;

    /** @brief Returns the left child of the node. This function is
     *         defined for only binary trees.
     */
    inline Tree_t *left();

    /** @brief Returns the right child of the node. This function is
     *         defined for only binary trees.
     */
    inline Tree_t *right();

    /** @brief Gets the index of the begin point of this subset.
     */
    inline index_t begin() const;

    /** @brief Gets the index one beyond the last index in the series.
     */
    inline index_t end() const;

    /** @brief Gets the number of points in this subset.
     */
    inline index_t count() const ;

    inline void Print() const;
};

/**
 * @brief A simple constructor
 */
template<typename TemplateMap>
Tree<TemplateMap>::Tree() {
  // stat_ = NULL;
  node_id_=-1;
}
/**
 * @brief deleted recursively the node and the corresponding subtree
 */

template<typename TemplateMap>
Tree<TemplateMap>::~Tree() {
 // delete stat_;
  if (!is_leaf()) {
    fl_private::ChildTrait<IsBinary, Tree_t>::Destruct(children_);
  }
}

/** @brief Build the tree given a table and the limit on
 *         args.leaf_size if mode==0
 *         args.level     if mode==1
 *         args.diameter  if mode==2
 *
 */
template<typename TemplateMap>
template<int mode, typename IndexArgsType>
typename Tree<TemplateMap>::Tree_t *Tree<TemplateMap>::BuildTree(
  typename Tree<TemplateMap>::Table_t &table,
  IndexArgsType &args,
  index_t *num_of_nodes) {
  index_t node_counter=*num_of_nodes;
  Tree_t *node=NULL;
  try {
    node = new Tree_t();
  } 
  catch(const std::bad_alloc &e) {
    fl::logger->Die() << "Problems while allocating memory in tree building. "
      << "It might be that your dataset is too big to fit in RAM or "
      << "you are using a 32bit platform which limits the process address space "
      << "to 4GB";
  }
  DEBUG_ASSERT(node_counter>=0);
  node->Init(0, table.n_entries(), node_counter);
  DEBUG_ASSERT(&(table.shuffled_to_real_[0]) != NULL);
  DEBUG_ASSERT(&(table.real_to_shuffled_[0]) != NULL);
  // Initialize the old from new mapping to the identity
  // mapping.
  for (index_t i = 0; i < table.n_entries(); i++) {
    table.shuffled_to_real_[i] = i;
    table.real_to_shuffled_[i] = i;
  }
  // Find the bounding box on the entire data and its level.
  node->bound().Init(table);
  typename Table_t::TreeIterator it = table.get_node_iterator(node);
  TreeSpec_t::SplitRule::FindBoundFromMatrix(args.metric,
      it, &node->bound());
  index_t initial_level = -1;
  if (store_level) {
    TreeSpec_t::ComputeLevel(table, node, (Tree_t *) NULL,
                             &initial_level);
  }
  node->set_level(initial_level);
  // Starting splitting.
  node_counter++;
  Tree_t::template SplitTree<mode>(args, table, node, initial_level, &node_counter);
  *num_of_nodes=node_counter;
  return node;
}

template<typename TemplateMap>
template<typename MetricType>
index_t Tree<TemplateMap>::MatrixPartition(MetricType &metric,
    typename Tree<TemplateMap>::Table_t &table,
    index_t first,
    index_t count,
    Tree_t *new_node,
    std::deque<bool> &left_membership) {

  index_t left = first;
  index_t right = first + count - 1;

  // At any point:
  //
  // everything < left is correct
  // everything > right is correct
  for (;;) {
    while (left_membership[left - first] && likely(left <= right)) {
      left++;
    }
    while (!left_membership[right - first] && likely(left <= right)) {
      right--;
    }
    if (unlikely(left > right)) {
      // left == right + 1
      break;
    }

    // Swap the left vector with the right vector.
    if (sort_points) {
      Point_t left_point, right_point;
      table.direct_get_(left, &left_point);
      table.direct_get_(right, &right_point);
      left_point.SwapValues(&right_point);
    }

    // Swap the membership boolean vectors and the old from new
    // vectors.
    bool t = left_membership[left - first];
    left_membership[left - first] = left_membership[right - first];
    left_membership[right - first] = t;
    std::swap(table.shuffled_to_real_[left],
              table.shuffled_to_real_[right]);
    table.real_to_shuffled_[table.shuffled_to_real_[left]] = left;
    table.real_to_shuffled_[table.shuffled_to_real_[right]] = right;
    DEBUG_ASSERT(left <= right);
    right--;
  }


  DEBUG_ASSERT(left == right + 1);

  // Create the new node and find its bound.
  new_node->Init(first, left - first);
  typename Table_t::TreeIterator it = table.get_node_iterator(new_node);
  TreeSpec_t::SplitRule::FindBoundFromMatrix(metric,
      it, &(new_node->bound()));
  return left;
}

template<typename TemplateMap>
template<int mode, typename IndexArgsType>
void Tree<TemplateMap>::SplitTree(IndexArgsType &args,
                                  typename Tree<TemplateMap>::Table_t &table,
                                  Tree_t *node,
                                  index_t level,
                                  index_t *node_counter) {

  if ((mode == 0 && node->count() > args.leaf_size)
      || (mode == 1 && level < args.level)
      || (mode == 2 && node->bound().MaxDistanceWithinBound() > args.diameter)) {
    index_t num_child_created = 0;
    index_t child_limit = (TreeSpec_t::is_binary) ? 1 :
                          std::numeric_limits<index_t>::max();
    index_t remaining_count = node->count();
    index_t starting_index = node->begin();

    while (num_child_created < child_limit
           && ((mode == 0 && remaining_count > args.leaf_size)
               || ((mode == 1 || mode == 2) && remaining_count > 0))) {
      std::deque<bool> membership;
      // Create an iterator that starts from the beginning of
      // the remaining set of points.
      typename Table_t::TreeIterator it =
        table.get_node_iterator(starting_index, remaining_count);
      bool can_cut = TreeSpec_t::SplitRule::Partition(args.metric,
                     it, node, &membership);
      if (!can_cut) {
        break;
      }
      // Create the new child and add to the child list of the
      // current node and increment the child number.

      Tree_t *new_child =NULL;
      try { 
        new_child=new Tree_t();
      }
      catch(const std::bad_alloc &e) {
        fl::logger->Die() << "Problems while allocating memory in tree building. "
          << "It might be that your dataset is too big to fit in RAM or "
          << "you are using a 32bit platform which limits the process address space "
          << "to 4GB";
      }
      new_child->bound().Init(table);
      node->set_children(new_child, num_child_created);
      num_child_created++;
      // Reorder the matrix so that the points under the new node
      // are sequentially ordered in DFS from left to right.
      starting_index = MatrixPartition(args.metric,
                                       table, starting_index, remaining_count,
                                       new_child, membership);
      DEBUG_ASSERT(*node_counter>=0);
      new_child->node_id()=*node_counter;
      (*node_counter)++;

      remaining_count -= new_child->count();

      // Compute the level of the new child node.
      index_t new_child_level = level;
      if (store_level) {
        TreeSpec_t::ComputeLevel(table, new_child,
                                 node, &new_child_level);
        new_child->set_level(new_child_level);
      }
      // Recurse on the new child branch.
      SplitTree<mode>(args, table,
                      new_child, new_child_level, node_counter);
    }
    // Create the final branch if at least one cutting was
    // successful and recurse.
    if (num_child_created > 0) {
      Tree_t *final_child = NULL;
      try {
        final_child = new Tree_t();
      }
      catch(const std::bad_alloc &e) {
        fl::logger->Die() << "Problems while allocating memory in tree building. "
          << "It might be that your dataset is too big to fit in RAM or "
          << "you are using a 32bit platform which limits the process address space "
          << "to 4GB";
      }

      node->set_children(final_child, num_child_created);
      final_child->bound().Init(table);
      DEBUG_ASSERT(*node_counter>=0);
      final_child->Init(starting_index, remaining_count, *node_counter);
      (*node_counter)++;
      typename Table_t::TreeIterator it =
        table.get_node_iterator(final_child);
      TreeSpec_t::SplitRule::FindBoundFromMatrix(
        args.metric, it, &(final_child->bound()));
      index_t final_child_level = level;
      if (store_level) {
        TreeSpec_t::ComputeLevel(table, final_child, node,
                                 &final_child_level);
        final_child->set_level(final_child_level);
      }
      SplitTree<mode>(args, table, final_child, final_child_level, node_counter);
    }
  } // end of the recursion case.
  // Compute the statistics based on the created children.
  // node->compute_statistic(table);
}

/**
 * @brief inherited from StoreLevelTrait
 */
template<typename TemplateMap>
void Tree<TemplateMap>::set_level(index_t level_in) {
  store_level_.set_level(level_in);
}
/**
 * @brief returns the level
 */
template<typename TemplateMap>
void Tree<TemplateMap>::level(index_t *level_out) const {
  store_level_.level(level_out);
}
/**
 *  prints the current level
 */
template<typename TemplateMap>
void Tree<TemplateMap>::PrintLevel(const
                                   typename Tree<TemplateMap>::Tree_t *root_node) const {
  store_level_.PrintLevel(root_node);
}

template<typename TemplateMap>
template<typename Archive>
void Tree<TemplateMap>::serialize(Archive &ar,
                                  const unsigned int version) {
  ar & boost::serialization::make_nvp("store_level", store_level_);
  ar & boost::serialization::make_nvp("bound", bound_);
  ar & boost::serialization::make_nvp("begin", begin_);
  ar & boost::serialization::make_nvp("count", count_);
  //ar & boost::serialization::make_nvp("stat", stat_);
  ar & boost::serialization::make_nvp("node_id", node_id_);
  bool isleaf=is_leaf();
  ar & boost::serialization::make_nvp("isleaf", isleaf);
  if (isleaf) {
    ar & boost::serialization::make_nvp("children", children_);
  }
}


template<typename TemplateMap>
const typename Tree<TemplateMap>::Tree_t *Tree<TemplateMap>::FindByBeginCountHelper(
  const typename Tree<TemplateMap>::ChildContainer_t &children_in,
  index_t begin_q) const {
  Tree_t *returned_node = NULL;
  for (index_t i = 0; i < children_.size(); i++) {
    if (children_in[i]->begin() >= begin_q) {
      returned_node = children_in[i - 1];
    }
  }
  return returned_node;
}
/**
 * @brief
 */
template<typename TemplateMap>
typename Tree<TemplateMap>::Tree_t *Tree<TemplateMap>::FindByBeginCountHelper_(
  const typename Tree<TemplateMap>::ChildContainer_t &children_in,
  index_t begin_q) {
  Tree_t *returned_node = NULL;
  for (index_t i = 0; i < children_.size(); i++) {
    if (children_in[i]->begin() >= begin_q) {
      returned_node = children_in[i - 1];
    }
  }
  return returned_node;
}
/**
 * @brief prints the tree recursively
 */
template<typename TemplateMap>
void Tree<TemplateMap>::PrintHelper_(
  const typename Tree<TemplateMap>::Tree_t *root_node) const {
  PrintLevel(root_node);
  std::cout<<": node: "
      << begin_ 
      <<"to " << begin_+count_-1 
      <<":" << count_ 
      <<" points total"<<std::endl;
  
  bound_.Print(std::cout, std::string(","));
  if (!is_leaf()) {
    printf(": Has %d children\n", (int) children_.size());
    for (std::size_t i = 0; i < children_.size(); i++) {
      children_[i]->PrintHelper_(root_node);
    }
  }
  else {
    printf(": is a leaf node.\n");
  }
}

template<typename TemplateMap>
void Tree<TemplateMap>::Init(index_t begin_in, index_t count_in, index_t node_id) {
  begin_ = begin_in;
  count_ = count_in;
  // stat_ = NULL;
  node_id_=node_id;
}

template<typename TemplateMap>
void Tree<TemplateMap>::Init(index_t begin_in, index_t count_in) {
  begin_ = begin_in;
  count_ = count_in;
}
/** @brief Find a node in this tree by its begin and count.
 *
 * Every node is uniquely identified by these two numbers.
 * This is useful for communicating position over the network,
 * when pointers would be invalid.
 *
 * @param begin_q the begin() of the node to find
 * @param count_q the count() of the node to find
 * @return the found node, or NULL
 */
template<typename TemplateMap>
const typename Tree<TemplateMap>::Tree_t* Tree<TemplateMap>::FindByBeginCount(
  index_t begin_q, index_t count_q) const {
  DEBUG_ASSERT(begin_q >= begin_);
  DEBUG_ASSERT(count_q <= count_);
  if (begin_ == begin_q && count_ == count_q) {
    return this;
  }
  else {
    if (unlikely(is_leaf())) {
      return NULL;
    }
  }
  const Tree_t *node = FindByBeginCountHelper_(children_, begin_q,
                       count_q);
  return node->FindByBeginCount(begin_q, count_q);
}

/** @brief Find a node in this tree by its begin and count (const).
 *
 * Every node is uniquely identified by these two numbers.
 * This is useful for communicating position over the network,
 * when pointers would be invalid.
 *
 * @param begin_q the begin() of the node to find
 * @param count_q the count() of the node to find
 * @return the found node, or NULL
 */
template<typename TemplateMap>
typename Tree<TemplateMap>::Tree_t* Tree<TemplateMap>::FindByBeginCount(
  index_t begin_q, index_t count_q) {
  DEBUG_ASSERT(begin_q >= begin_);
  DEBUG_ASSERT(count_q <= count_);
  if (begin_ == begin_q && count_ == count_q) {
    return this;
  }
  else {
    if (unlikely(is_leaf())) {
      return NULL;
    }
  }
  Tree_t *node = FindByBeginCountHelper_(children_, begin_q, count_q);
  return node->FindByBeginCount(begin_q, count_q);
}

template<typename TemplateMap>
typename Tree<TemplateMap>::ChildContainer_t &Tree<TemplateMap>::children() {
  return children_;
}

template<typename TemplateMap>
const typename Tree<TemplateMap>::ChildContainer_t &Tree<TemplateMap>::children() const {
  return children_;
}

template<typename TemplateMap>
void Tree<TemplateMap>::set_children(
  typename Tree<TemplateMap>::Tree_t *new_child_in, index_t new_child_index) {
  fl_private::ChildTrait<IsBinary, Tree_t>::set_children(
    children_, new_child_in, new_child_index);
}

// template<typename TemplateMap>
// void Tree<TemplateMap>::compute_statistic(typename Tree<TemplateMap>::Table_t &table) {
//  fl_private::ComputeStatisticHelper<TreeSpec_t::is_binary>::
//  ComputeStatistic(this, table);
// }

template<typename TemplateMap>
typename Tree<TemplateMap>::Bound_t& Tree<TemplateMap>::bound() {
  return bound_;
}

template<typename TemplateMap>
const typename Tree<TemplateMap>::Bound_t& Tree<TemplateMap>::bound() const {
  return bound_;
}

template<typename TemplateMap>
index_t& Tree<TemplateMap>::node_id() {
  return node_id_;
}

template<typename TemplateMap>
const index_t& Tree<TemplateMap>::node_id() const {
  return node_id_;
}

//template<typename TemplateMap>
//typename Tree<TemplateMap>::Statistic_t& Tree<TemplateMap>::stat() {
//  return *stat_;
//}

//template<typename TemplateMap>
//const typename Tree<TemplateMap>::Statistic_t& Tree<TemplateMap>::stat() const {
//  return *stat_;
//}

//template<typename TemplateMap>
//typename Tree<TemplateMap>::Statistic_t* &Tree<TemplateMap>::stat_ptr() {
//  return stat_;
//}

//template<typename TemplateMap>
//const typename Tree<TemplateMap>::Statistic_t* &Tree<TemplateMap>::stat_ptr() const {
//  return stat_;
//}

template<typename TemplateMap>
bool Tree<TemplateMap>::is_leaf() const {
  return fl_private::ChildTrait<IsBinary, Tree_t>::is_leaf(children_);
}

/** @brief Returns the left child of the node. This function is
 *         defined for only binary trees.
 */
template<typename TemplateMap>
typename Tree<TemplateMap>::Tree_t *Tree<TemplateMap>::left() {
  return fl_private::LeftRightAccessors<TreeSpec_t::is_binary>::
         Left(this);
}

/** @brief Returns the right child of the node. This function is
 *         defined for only binary trees.
 */
template<typename TemplateMap>
typename Tree<TemplateMap>::Tree_t *Tree<TemplateMap>::right() {
  return fl_private::LeftRightAccessors<TreeSpec_t::is_binary>::
         Right(this);
}

/** @brief Gets the index of the begin point of this subset.
 */
template<typename TemplateMap>
index_t Tree<TemplateMap>::begin() const {
  return begin_;
}

/** @brief Gets the index one beyond the last index in the series.
 */
template<typename TemplateMap>
index_t Tree<TemplateMap>::end() const {
  return begin_ + count_;
}

/** @brief Gets the number of points in this subset.
 */
template<typename TemplateMap>
index_t Tree<TemplateMap>::count() const {
  return count_;
}

template<typename TemplateMap>
void Tree<TemplateMap>::Print() const {
  PrintHelper_(this);
}


}; // tree namespace
}; // fl namespace

#endif
