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
#ifndef FASTLIB_TREE_SPACETREE_PRIVATE_H
#define FASTLIB_TREE_SPACETREE_PRIVATE_H

#include <algorithm>
#include <vector>
#include "boost/mpl/assert.hpp"
#include "fastlib/base/base.h"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/base_object.hpp"

namespace fl_private {

template<bool IsBinary, typename T>
class ChildTrait {
};

template<typename T>
class ChildTrait<true, T> {
  public:
    struct Container {
     public:
      Container() {
        data_[0] = NULL;
        data_[1] = NULL;
      }
      size_t size() const {
        return 2;
      }

      const T* operator[](size_t i) const {
        DEBUG_BOUNDS(i, 2);
        return data_[i];
      }

      T* &operator[](size_t i) {
        DEBUG_BOUNDS(i, 2);
        return data_[i];
      }
      template<typename Archive>
      void serialize(Archive &ar, const unsigned int version) {
        ar & boost::serialization::make_nvp("data", data_);
      }

private:
      T* data_[2];
    };

    static void set_children(Container &cont, T* child, index_t i) {
      DEBUG_BOUNDS(i, 2);
      cont[i] = child;
    }

    static bool is_leaf(const Container &cont) {
      return (cont[0] == NULL);
    }

    static void Destruct(Container &cont) {
      delete cont[0];
      delete cont[1];
    }

};

template<typename T>
class ChildTrait<false, T> {
  public:
    class Container : public std::vector<T*> {
      public:
        Container(size_t size) : std::vector<T*>(size) {
          std::fill(std::vector<T*>::begin(),
                    std::vector<T*>::end(), NULL);
        }
        template<typename Archive>
        void serialize(Archive &ar, const unsigned int version) {
          ar & boost::serialization::make_nvp("vector", boost::serialization::base_object<std::vector<T*> >(*this));
        }
    };

    static void set_children(Container &cont, T* child, index_t i) {
      DEBUG_ASSERT(i >= 0);
      if (i < (index_t)cont.size()) {
        cont[i] = child;
      }
      else {
        cont.push_back(child);
      }
    }

    static bool is_leaf(const Container &cont) {
      return (cont.size() == 0);
    }

    static void Destruct(Container &cont) {
      for (std::size_t i = 0; i < cont.size(); i++) {
        delete cont[i];
      }
    }

};

template<bool StoreLevel>
class StoreLevelTrait {
  public:
    void set_level(index_t level_in);
    void level(index_t *level_out) const;

    template<typename TreeType>
    void PrintLevel(const TreeType *root_node) const;
};

template<>
class StoreLevelTrait<true> {
  private:
    index_t level_;

  public:

    StoreLevelTrait() {
      level_ = 0;
    }

    ~StoreLevelTrait() {
    }

    void set_level(index_t level_in) {
      level_ = level_in;
    }

    void level(index_t *level_out) const {
      *level_out = level_;
    }
    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & boost::serialization::make_nvp("level",level_);
    }

    template<typename TreeType>
    void PrintLevel(const TreeType *root_node) const {
      index_t root_level;
      root_node->level(&root_level);
      index_t level_difference = root_level - level_;
      for (index_t i = 0; i < level_difference; i++) {
        std::cout<<" ";
      }
      std::cout<<"Level "<< level_;
    }
};

template<>
class StoreLevelTrait<false> {
  public:
    void set_level(index_t level_int) {
    }

    void level(index_t *level_out) const {
    }

    template<typename TreeType>
    void PrintLevel(const TreeType *root_node) const {
    }
};

template<bool BinaryTree>
class LeftRightAccessors {
  public:
    BOOST_MPL_ASSERT_MSG(BinaryTree,
                         LeftRightAccessor_is_defined_only_for_binary_trees, ());
    template<typename TreeType>
    static TreeType *Left(TreeType *node);

    template<typename TreeType>
    static TreeType *Right(TreeType *node);
};

template<>
class LeftRightAccessors<true> {

  public:

    template<typename TreeType>
    static TreeType *Left(TreeType *node) {
      return (TreeType *)(node->children())[0];
    }

    template<typename TreeType>
    static TreeType *Right(TreeType *node) {
      return (TreeType *)(node->children())[1];
    }
};


//template<bool BinaryTree>
//class ComputeStatisticHelper {
//  public:
//
//    template<typename TreeType, typename TableType>
//    static void ComputeStatistics(TreeType *node, const TableType &table);
//};
//
//template<>
//class ComputeStatisticHelper<true> {
//  public:
//
//    template<typename TreeType, typename TableType>
//    static void ComputeStatistic(TreeType *node, TableType &table) {
//      typename TableType::TreeIterator it = table.get_node_iterator(node);
//      if (!table.node_is_leaf(node)) {
//        /*
//        typename TreeType::Statistic_t &stat =
//          table.get_node_stat(node);
//        typename TreeType::Statistic_t &stat_left =
//          table.get_node_stat(table.get_node_left_child(node));
//        typename TreeType::Statistic_t &stat_right =
//          table.get_node_stat(table.get_node_right_child(node));
//        */
////****************Warning you must fix this
//        //stat.Init(it, stat_left, stat_right);
//      } else {
//        // node->stat().Init(it);
//      }
//    }
//};
//
//template<>
//class ComputeStatisticHelper<false> {
//
//  public:
//    template<typename TreeType, typename TableType>
//    static void ComputeStatistic(TreeType *node, const TableType &table) {
//      typename TableType::TreeIterator it = table.get_node_iterator(node);
//      if (!node->is_leaf()) {
////************ Warning you must fix this
//        // node->stat().Init(it, node->children());
//      } else {
//        //node->stat().Init(it);
//      }
//    }
//};
};

#endif
