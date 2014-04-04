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
#ifndef FL_LITE_FASTLIB_TABLE_TABLE_DEFS_H_
#define FL_LITE_FASTLIB_TABLE_TABLE_DEFS_H_
#include "table.h"
#include "boost/archive/archive_exception.hpp"

namespace fl {
/**
 * @brief namespace table, contains table structures that look like database tables
 */
namespace table {
  template<typename TemplateMap>
  template<typename DenseContainer>
  void Table<TemplateMap>::Init(const DenseContainer &cont) {
    data_->template Init<typename boost::mpl::front<typename Dataset_t::DenseTypeList_t>::type >(cont);
    SetAll(0.0);
  }
  
  template<typename TemplateMap>
  template<typename ContainerType>
  void Table<TemplateMap>::Init(ContainerType dense_dimensions,
                                ContainerType sparse_dimensions,
                                index_t num_of_points) {
    data_->Init(dense_dimensions,
                sparse_dimensions,
                num_of_points);
    SetAll(0.0);
  }
  
  template<typename TemplateMap>
  template<typename ContainerType>
  void Table<TemplateMap>::Init(const std::string &filename,
                                ContainerType dense_dimensions,
                                ContainerType sparse_dimensions,
                                index_t num_of_points) {
    filename_=filename;
    data_->Init(dense_dimensions,
                sparse_dimensions,
                num_of_points);
    SetAll(0.0);
  }

  template<typename TemplateMap>
  template<typename IndexArgsType>
  void Table<TemplateMap>::SplitNode(IndexArgsType &args, Tree_t *node) {
  
    DEBUG_ASSERT(metric_type_id_ == typeid(args.metric).name());
    if (node == NULL) {
      if (tree_ != NULL) {
        fl::logger->Die()<<"The tree is already initialized!";
      }
      else {
        real_to_shuffled_.resize(n_entries());
        shuffled_to_real_.resize(n_entries());
        args.leaf_size = n_entries();
        num_of_nodes_=0;
        tree_ = Tree_t::template BuildTree<0>(*this, args, &num_of_nodes_);
      }
    }
    else {
      if (!node->is_leaf()) {
        fl::logger->Die()<<"You cannot split a non-leaf node!";
      }
      else {
        Tree_t::template SplitTree<0>(args, *this, node, 0, &num_of_nodes_);
      }
    }
  }
  
  template<typename TemplateMap>
  template<typename IndexArgsType>
  void Table<TemplateMap>::IndexData(IndexArgsType &args) {
    if (tree_ != NULL) {
      fl::logger->Die() << "You are trying to index a table which is already indexed";
    }
    if (data_->n_points()<=0) {
      fl::logger->Die()<<"You are trying to index a table that is empty";
    }
    real_to_shuffled_.resize(data_->n_points());
    shuffled_to_real_.resize(data_->n_points());
    leaf_size_ = args.leaf_size;
    if (leaf_size_==1) {
      fl::logger->Die()<<"You cannot build a tree with leaf_size=1. Some algorithms might crash";
    }
    metric_type_id_ = typeid(args.metric).name();
    num_of_nodes_=0;
    if (args.leaf_size > 0) {
      if (args.level>0 || args.diameter>0) {
        fl::logger->Die() << "Level and Diameter must be negative if leaf_size "
                          << "is positive";
      }
      tree_ = Tree_t::template BuildTree<0>(*this, args, &num_of_nodes_);
    } else {
      if (args.level > 0) {
        if (args.leaf_size>0 || args.diameter>0) {
          fl::logger->Die() << "Leaf size and Diameter must be negative if level "
                          << "is positive";
        }
        tree_ = Tree_t::template BuildTree<1>(*this, args, &num_of_nodes_);
      } else {
        if (args.diameter > 0) {
          if (args.leaf_size>0 || args.level>0) {
            fl::logger->Die() << "Leaf size and Level must be negative if diameter "
                               << "is positive";
          }
          tree_ = Tree_t::template BuildTree<2>(*this, args, &num_of_nodes_);
        }     
      }
    }
  }

  template<typename TemplateMap>
  template<typename ContainerType>
  void Table<TemplateMap>::ComputeNodesPerLevel(ContainerType *cont) {
    if (tree_ == NULL) {
      fl::logger->Die()<<"You are attempting to compute the node distribution of "
            "a tree that doesn't exist";
    }
    index_t level = 0;
    ComputeNodesPerLevelRecursion_(tree_, level, cont);
  }
  
  
  template<typename TemplateMap>
  template<typename ContainerType1, typename ContainerType2>
  void Table<TemplateMap>::ComputeStatsPerLevel(ContainerType1 *nodes, 
      ContainerType2 *diameters) {
    if (tree_ == NULL) {
      fl::logger->Die()<<"You are attempting to compute the node distribution of "
            "a tree that doesn't exist";
    }
    index_t level = 0;
    ComputeStatsPerLevelRecursion_(tree_, level, nodes, diameters);
    DEBUG_ASSERT(nodes->size()==diameters->size());
    for(index_t i=0; i<nodes->size(); i++) {
      DEBUG_ASSERT((*nodes)[i]!=0);
      (*diameters)[i]/=(*nodes)[i];
    }
  }

  template<typename TemplateMap>
  template<typename ContainerType1, typename ContainerType2>
  void Table<TemplateMap>::ComputeStatsUpToLevel(ContainerType1 *nodes, 
      ContainerType1 *leafs,
      ContainerType2 *node_diameters,
      ContainerType2 *leaf_diameters) {
    if (tree_ == NULL) {
      fl::logger->Die()<<"You are attempting to compute the node distribution of "
            "a tree that doesn't exist";
    }
    index_t level = 0;
    ComputeStatsUpToLevelRecursion_(tree_, level, nodes, leafs, 
        node_diameters, leaf_diameters);
    for(index_t i=0; i<nodes->size(); i++) {
      (*node_diameters)[i]/=(*nodes)[i];
    }
    for(index_t i=0; i<leaf_diameters->size(); i++) {
      if ((*leafs)[i]!=0) {
        (*leaf_diameters)[i]/=(*leafs)[i];
      } else {
        (*leaf_diameters)[i]=std::numeric_limits<double>::max();
      }
    }
  }

  template<typename TemplateMap>
  template<typename ContainerType>
  void  Table<TemplateMap>::ComputeNodesPerLevelRecusrion_(
    Tree_t *node,
    index_t level,
    ContainerType *cont) {
    if (level >= cont->size()) {
      cont->push_back(1);
    }
    else  {
      cont->operator[](level)+=1;
    }
    if (!node_is_leaf(node)) {
      level++;
      ComputeNodesPerLevelRecusrion_(get_node_left_child(node),
                                     level,
                                     cont);
      ComputeNodesPerLevelRecusrion_(get_node_right_child(node),
                                     level,
                                     cont);
    }
  }

  template<typename TemplateMap>
  template<typename ContainerType1, typename ContainerType2>
  void  Table<TemplateMap>::ComputeStatsPerLevelRecursion_(
    Tree_t *node,
    index_t level,
    ContainerType1 *nodes,
    ContainerType2 *diameters) {
    if (level >= nodes->size()) {
      nodes->push_back(1);
      diameters->push_back(get_node_bound(node).MaxDistanceWithinBound());
    } else  {
      nodes->operator[](level)+=1;
      diameters->operator[](level)+=get_node_bound(node).MaxDistanceWithinBound();
    }
    if (!node_is_leaf(node)) {
      level++;
      ComputeStatsPerLevelRecursion_(get_node_left_child(node),
                                     level,
                                     nodes,
                                     diameters);
      ComputeStatsPerLevelRecursion_(get_node_right_child(node),
                                     level,
                                     nodes,
                                     diameters);
    }
  }
 
  template<typename TemplateMap>
  template<typename ContainerType1, typename ContainerType2>
  void Table<TemplateMap>::ComputeStatsUpToLevelRecursion_(
    Tree_t *node,
    index_t level,
    ContainerType1 *nodes,
    ContainerType1 *leafs,
    ContainerType2 *node_diameters,
    ContainerType2 *leaf_diameters) {
    if (level >= nodes->size()) {
      nodes->push_back(0);
      leafs->push_back(0);
      node_diameters->push_back(0);
      leaf_diameters->push_back(0);
    }
    if (node_is_leaf(node)) {
      leafs->operator[](level)+=1;  
      leaf_diameters->operator[](level)+=get_node_bound(node).MaxDistanceWithinBound();
    } else {
      nodes->operator[](level)+=1;
      node_diameters->operator[](level)+=get_node_bound(node).MaxDistanceWithinBound();
    }
    if (!node_is_leaf(node)) {
      level++;
      ComputeStatsUpToLevelRecursion_(get_node_left_child(node),
                                     level,
                                     nodes,
                                     leafs,
                                     node_diameters,
                                     leaf_diameters);
      ComputeStatsUpToLevelRecursion_(get_node_right_child(node),
                                     level,
                                     nodes,
                                     leafs,
                                     node_diameters,
                                     leaf_diameters);
    } 
  }
 

}}
#endif
