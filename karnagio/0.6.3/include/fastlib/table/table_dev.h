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
#ifndef FL_LITE_FASTLIB_TABLE_TABLE_DEV_H_
#define FL_LITE_FASTLIB_TABLE_TABLE_DEV_H_
#include "fastlib/table/table.h"
#include "fastlib/table/table_defs.h"
#include "fastlib/math/fl_math.h"
#include "boost/lexical_cast.hpp"
#include "boost/mpl/if.hpp"
#include "boost/mpl/not.hpp"
#include "boost/mpl/and.hpp"
#include "boost/mpl/eval_if.hpp"
#include "boost/type_traits/is_same.hpp"
#include "fastlib/util/string_utils.h"

namespace fl {
/**
 * @brief namespace table, contains table structures that look like database tables
 */
namespace table {
  
template<typename TemplateMap>
Table<TemplateMap>::TreeIterator::TreeIterator(const Table_t &table,
    const Tree_t *node) :
    table_(table), start_(node->begin()), end_(node->end()), ind_(start_ -1)  {
  DEBUG_ASSERT(end_ >= start_);
}

template<typename TemplateMap>
Table<TemplateMap>::TreeIterator::TreeIterator(const Table_t &table,
    index_t begin,
    index_t count) :
    table_(table), start_(begin), end_(begin + count), ind_(start_ - 1) {
  DEBUG_ASSERT(begin >= 0 && count > 0);
}


template<typename TemplateMap>
bool Table<TemplateMap>::TreeIterator::HasNext() const {
  if (ind_ < end_ -1) {
    return true;
  }
  else {
    return false;
  }
}

template<typename TemplateMap>
void Table<TemplateMap>::TreeIterator::Next(Point_t *entry, index_t *point_id)  {
  ind_++;
  table_.direct_get_(ind_, entry);
  *point_id = table_.direct_get_id_(ind_);
}

template<typename TemplateMap>
void Table<TemplateMap>::TreeIterator::get(index_t i, Point_t *entry)  {
  table_.direct_get_(start_ + i, entry);
}

template<typename TemplateMap>
void Table<TemplateMap>::TreeIterator::get_id(index_t i, index_t *id)  {
  *id = table_.direct_get_id_(start_ + i);
}

template<typename TemplateMap>
void Table<TemplateMap>::TreeIterator::RandomPick(Point_t *entry) {
  table_.direct_get_(fl::math::Random(start_, end_-1), entry);
}

template<typename TemplateMap>
void Table<TemplateMap>::TreeIterator::RandomPick(Point_t *entry,
    index_t *point_id) {
  *point_id = fl::math::Random(start_, end_-1);
  table_.direct_get_(*point_id, entry);
}

template<typename TemplateMap>
void Table<TemplateMap>::TreeIterator::Reset() {
  ind_ = start_ -1;
}

template<typename TemplateMap>
index_t Table<TemplateMap>::TreeIterator::count() const {
  return end_ - start_;
}

template<typename TemplateMap>
const  Table<TemplateMap> &Table<TemplateMap>::TreeIterator::table() const {
  return table_;
}

template<typename TemplateMap>
index_t Table<TemplateMap>::TreeIterator::start() const {
  return start_;
}

template<typename TemplateMap>
index_t Table<TemplateMap>::TreeIterator::end() const {
  return end_;
}

template<typename TemplateMap>
Table<TemplateMap>::Table(typename Table<TemplateMap>::Dataset_t *data) {
  tree_ = NULL;
  data_.reset(data);
  cached_point_.first=-1;
  cached_point_.second=new Point_t();
}


template<typename TemplateMap>
Table<TemplateMap>::Table() {
  tree_ = NULL;
  num_of_nodes_=0;
  data_.reset(new Dataset_t());
  cached_point_.first=-1;
  cached_point_.second=new Point_t();
}

template<typename TemplateMap>
Table<TemplateMap>::~Table() {
  if (tree_ != NULL) {
    delete tree_;
    tree_ = NULL;
  }
  delete cached_point_.second;
}

template<typename TemplateMap>
void Table<TemplateMap>::Init(const std::string &file, const char* mode) {
  data_->Init(file, mode);
}


template<typename TemplateMap>
void Table<TemplateMap>::Destruct() {
  if (tree_ != NULL) {
    delete tree_;
    tree_ = NULL;
  }
  tree_ = NULL;
  data_.reset(new Dataset_t());

}

template<typename TemplateMap>
void Table<TemplateMap>::Save() {
  data_->Save(filename_,
             true,
             std::vector<std::string>(),
             std::string(","));

}

template<typename TemplateMap>
void Table<TemplateMap>::CloneDataOnly(Table<TemplateMap> *table) {
  table->Init(filename_, 
      this->dense_sizes(), 
      this->sparse_sizes(),
      this->n_entries());
  table->labels()=this->labels();
  for(index_t i=0; i<this->n_entries(); ++i) {
    Point_t p1, p2;
    this->get(i, &p1);
    table->get(i, &p2);
    p2.CopyValues(p1);
  }
}

template<typename TemplateMap>
void Table<TemplateMap>::Append(Table<TemplateMap> &table) {
  Point_t point;
  for(index_t i=0;i<table.n_entries(); ++i) {
    table.get(i, &point);
    this->push_back(point);
  }
}

template<typename TemplateMap>
std::vector<index_t> Table<TemplateMap>::RedundantCategoricals() {
  std::vector<index_t> red_features;
  Point_t point;
  const std::vector<std::string> &labels=this->labels();
  std::map<std::string, std::pair<index_t, index_t> > categorical_ranges;
  for(index_t i=0; i<labels.size(); ++i) {
    if (fl::StringStartsWith(labels[i], "cat:")) {
      std::vector<std::string> tokens=fl::SplitString(labels[i], ":");
      if (tokens.size()<3) {
        fl::logger->Die()<<"Categorical labels/attribute_names must be of the "
          "form cat:var_name:number";
      }
      const std::string var_name=tokens[1];
      if (categorical_ranges.count(var_name)) {
        if (i>=categorical_ranges[var_name].second) {
          categorical_ranges[var_name].second=i;
        }
      } else {
        categorical_ranges[var_name].first=i;
        categorical_ranges[var_name].second=i+1;
      }
    }
  }
  std::list<std::pair<index_t, double> > active_columns;
  this->get(0, &point);
  for(std::map<std::string, std::pair<index_t, index_t> >::iterator it=categorical_ranges.begin();
      it!=categorical_ranges.end(); ++it) {
    for(index_t i=it->second.first; i!=it->second.second; ++i) {
      active_columns.push_back(
          std::make_pair(i, point[i]));
    }
  }
  // remove all the columns that all have the same value
  for(index_t i=0; i<this->n_entries(); ++i) {
    if (active_columns.empty()) {
      break;
    } 
    this->get(i, &point);
    for(std::list<std::pair<index_t, double> >::iterator it=active_columns.begin();
        it!=active_columns.end(); ++it) {
      if (point[it->first]!=it->second) {
        it=active_columns.erase(it);
      } 
    }
  }
  std::set<index_t> deleted_columns;
  for(std::list<std::pair<index_t, double> >::iterator it=active_columns.begin();
        it!=active_columns.end(); ++it) {
    deleted_columns.insert(it->first);
  }

  // check if we have a point with all columns zeros
  // if we don't have one, then we should remove a column
  bool all_zero_col=true;
  std::map<std::string, bool> extra_col;
  for(index_t i=0; i<this->n_entries(); ++i) {
    this->get(i, &point);
    std::map<std::string, std::pair<index_t, index_t> >::iterator it;
    for(it=categorical_ranges.begin();
        it!=categorical_ranges.end(); ++it) {
      for(index_t i=it->second.first; i!=it->second.second; ++i) {   
        if (deleted_columns.count(i)!=0) {
          continue;
        }
        if (point[i]!=0) {
          all_zero_col=false;
          break;
        }
      }
      if (all_zero_col==true) {
        extra_col[it->first]=true;
        break;
      }
    }
  }
  for(std::map<std::string, std::pair<index_t, index_t> >::iterator it=categorical_ranges.begin();
        it!=categorical_ranges.end(); ++it) {
    for(index_t i=it->second.first; i!=it->second.second; ++i) {
      if (deleted_columns.count(i)) {
        red_features.push_back(i);
      }
    }
    if (extra_col[it->first]==false) {
      index_t k=1;
      while(true) {
        if (deleted_columns.count(it->second.second-k)==0) {
          red_features.push_back(it->second.second-k);
          break;
        }
        if (it->second.second-k==it->second.first) {
          break;
        }
        k++;
      }
    }
  }
  return red_features;
}

template<typename TemplateMap>
void Table<TemplateMap>::DeleteIndex() {
  metric_type_id_ = "";
  num_of_nodes_=0;
  if (tree_ != NULL) {
    delete tree_;
    tree_ = NULL;
  }
  if (sort_points == true) {
    for (index_t i = 0; i < static_cast<index_t>(real_to_shuffled_.size()); i++) {
      if (real_to_shuffled_[i] != i) {
        Point_t p1, p2;
        data()->get(real_to_shuffled_[i], &p1);
        data()->get(i, &p2);
        p1.SwapValues(&p2);
        real_to_shuffled_[shuffled_to_real_[i]] = real_to_shuffled_[i];
        shuffled_to_real_[real_to_shuffled_[i]] = shuffled_to_real_[i];
        shuffled_to_real_[i] = i;
        real_to_shuffled_[i] = i;
      }
    }
  }
  real_to_shuffled_.clear();
  shuffled_to_real_.clear();
}


template<typename TemplateMap>
void Table<TemplateMap>::LogTreeStats() {
  std::vector<index_t> nodes;
  std::vector<CalcPrecision_t> diameters;
  ComputeStatsPerLevel(&nodes, &diameters);
  std::ostringstream stream;  
  for(index_t i=0; i<nodes.size(); i++) {
    stream << i << ":" << nodes[i] <<",";
  }
  fl::logger->Message() << "Distribution of nodes per level (level:number_of_nodes): "
    << stream.str();
  stream.str("");
  for(index_t i=0; i<diameters.size(); i++) {
    stream << i << ":" << diameters[i] <<",";
  }
  fl::logger->Message() << "Distribution of diameters per level (level:average_diameter): "
    << stream.str();

}

template<typename TemplateMap>
void Table<TemplateMap>::RestrictTableToCentroidsUpToLevel(index_t up_to_level, 
   Table<TemplateMap> *table) {
  if (tree_ == NULL) {
    fl::logger->Die() << "You attempted to sample on a tree that doesn't exist";
  }
  if (up_to_level < 0) {
    fl::logger->Die() << "The level should be greater or equal to one";
  }
  table->Init(data_->dense_sizes(), data_->sparse_sizes(), 0);
  RestrictTableToCentroidsUpToLevelRecursion_(tree_, 0, up_to_level, table);
}

template<typename TemplateMap>
void Table<TemplateMap>::RestrictTableToCentroidsUpToDiameter(CalcPrecision_t diameter,
    Table<TemplateMap> *table) {
  if (tree_ == NULL) {
    fl::logger->Die()<<"You attempted to sample on a tree that doesn't exist";
  }
  if (diameter < 0) {
    fl::logger->Die()<<"The diameter should be greater than zero";
  }
  table->Init("", data_->dense_sizes(), data_->sparse_sizes(), 0);
  RestrictTableToCentroidsUpToDiameterRecursion_(tree_, diameter, table);
}

template<typename TemplateMap>
void Table<TemplateMap>::RestrictTableToSamplesUpToLevel(index_t up_to_level,
    index_t num_of_samples_per_node,
    Table<TemplateMap> *table) {
  if (tree_ == NULL) {
    fl::logger->Die()<<"You attempted to sample on a tree that doesn't exist";
  }
  if (up_to_level < 0) {
    fl::logger->Die()<<"The level should be greater or equal to one";
  }
  table->Init("", data_->dense_sizes(), data_->sparse_sizes(), 0);
  RestrictTableToSamplesUpToLevelRecursion_(tree_, up_to_level, 0,
                                     num_of_samples_per_node,
                                     table);
}

template<typename TemplateMap>
void Table<TemplateMap>::RestrictTableToSamplesUpToDiameter(
  typename Table<TemplateMap>::CalcPrecision_t diameter,
  index_t num_of_samples_per_node,
  Table<TemplateMap> *table) {
  if (tree_ == NULL) {
    fl::logger->Die()<<"You attempted to sample on a tree that doesn't exist";
  }
  if (diameter < 0) {
    fl::logger->Die()<<"The diameter should be greater than zero";
  }
  table->Init("", data_->dense_sizes(), data_->sparse_sizes(), 0);
  RestrictTableToSamplesUpToDiameterRecursion_(tree_,
                                     diameter,
                                     num_of_samples_per_node, 
                                     table);
}
 
 namespace temp_table_split_function {
   struct MetaFunction1 {
     template<typename PointType>
     static void Assign(PointType *point, index_t i) {
       point->meta_data().template get<2>()=i;
     }
   };
   struct MetaFunction2 {
     template<typename PointType>
     static void Assign(PointType *point, index_t i) {
     }
   };
 }

template<typename TemplateMap>
std::vector<boost::shared_ptr<Table<TemplateMap> > > Table<TemplateMap>::Split(int32 n_tables,
    const std::string &arguments) {
  std::vector<boost::shared_ptr<Table_t> > new_tables(n_tables);
  std::vector<std::string> args=fl::SplitString(arguments, ",");
  const std::string &random_method=args[0];
  const bool assign_labels=boost::lexical_cast<bool>(args[1]);
  for(int32 i=0; i<n_tables; ++i) {
    new_tables[i].reset(new Table_t());
    new_tables[i]->Init("dummy", 
        this->dense_sizes(),
        this->sparse_sizes(), 
        0);
    new_tables[i]->labels()=this->labels();
  }  

  if (random_method=="random_unique") {
    Point_t point;
    for(index_t i=0; i<this->n_entries(); ++i) {
      this->get(i, &point);        
      int32 table_to_be_assigned=fl::math::Random(0, n_tables-1);  
      if (assign_labels==true) {
        boost::mpl::if_<
          typename Dataset_t::HasMetaData_t,
          temp_table_split_function::MetaFunction1,
          temp_table_split_function::MetaFunction2>::type::Assign(&point, i);
      }
      new_tables[table_to_be_assigned]->push_back(point);
    }
  } else {
    fl::logger->Die()<<"This Method ("<<random_method<<") is not supported";
  }
  return new_tables;
}

template<typename TemplateMap>
bool Table<TemplateMap>::is_indexed() const {
  return tree_ != NULL;
}

template<typename TemplateMap>
void Table<TemplateMap>::PrintTree() {
  tree_->Print();
}

template<typename TemplateMap>
typename Table<TemplateMap>::Tree_t * Table<TemplateMap>::get_tree() const {
  return tree_;
}

template<typename TemplateMap>
typename Table<TemplateMap>::ExportedPointCollection_t Table<TemplateMap>::get_point_collection() const {
  return data_->point_collection();
}

template<typename TemplateMap>
typename Table<TemplateMap>::Dataset_t *Table<TemplateMap>::data()  {
  return data_.get();
}

template<typename TemplateMap>
const typename Table<TemplateMap>::Dataset_t *Table<TemplateMap>::data() const  {
  return data_.get();
}

template<typename TemplateMap>
const std::string &Table<TemplateMap>::filename() const  {
  return filename_;
}

template<typename TemplateMap>
const std::vector<std::string> &Table<TemplateMap>::labels() const {
  return data_->labels();
}

template<typename TemplateMap>
std::vector<std::string> &Table<TemplateMap>::labels() {
  return data_->labels();
}

template<typename TemplateMap>
std::string &Table<TemplateMap>::filename()  {
  return filename_;
}

template<typename TemplateMap>
index_t Table<TemplateMap>::num_of_nodes() const  {
  return num_of_nodes_;
}

template<typename TemplateMap>
const std::string Table<TemplateMap>::get_tree_metric()  {
  return metric_type_id_;
}

template<typename TemplateMap>
void Table<TemplateMap>::get(index_t point_id, typename Table<TemplateMap>::Point_t *entry) const {
  if (sort_points == false || !is_indexed()) {
    data_->get(point_id, entry);
  }
  else {
    data_->get(real_to_shuffled_[point_id], entry);
  }
}

namespace table_get {
  template<typename TableType>
  struct General {
    struct type {
      static void get(TableType *table, 
          index_t i,
          index_t j,
          double *val) {
        if (i!=table->get_cached_point().first) {
          table->get(i, table->get_cached_point().second);
          table->get_cached_point().first=i;
        }
        *val=static_cast<double>(table->get_cached_point().second->operator[](j));
      }
    };
  };
  
  template<typename TableType>
  struct Special {
    struct type {
      static void get(TableType *table,
          index_t i,
          index_t j,
          double *val) {
        *val=table->get_point_collection().dense->template get<
          typename TableType::DenseBasicStorageType_t>().get(j,i);
      }
    };
  };
}

template<typename TemplateMap>
double Table<TemplateMap>::get(index_t i, index_t j) {
  double val=0;
  boost::mpl::eval_if<
    IsMatrixOnly_t, 
    table_get::Special<Table_t>,
    table_get::General<Table_t>
  >::type::get(this, i, j, &val);
  return val;  
}

template<typename TemplateMap>
void Table<TemplateMap>::get(index_t i, 
    std::vector<std::pair<index_t, double> > *p1) {
  Point_t p;
  this->get(i, &p);
  std::vector<std::pair<index_t, double> > point;
  for(typename Point_t::iterator it=p.begin(); it!=p.end(); ++it) {
    p1->push_back(std::pair<index_t, double>(
          it.attribute(), static_cast<double>(it.value())));
  } 
}

struct MetaGetter1 {
  struct type {
    template<typename PointType>
    static void Do(PointType &p, 
         signed char *meta1, double *meta2, int* meta3) {
 
      *meta1=p->meta_data().template get<0>();
      *meta2=p->meta_data().template get<1>();
      *meta3=p->meta_data().template get<2>();
    }
  };
};

struct MetaGetter2 {
  struct type {
    template<typename PointType>
    static void Do(PointType &p, 
        signed char *meta1, double *meta2, int* meta3) {
      *meta1=0;
      *meta2=0;
      *meta3=0;
    }
  };
};
template<typename TemplateMap>
void Table<TemplateMap>::get(index_t i, signed char *meta1,
    double *meta2, int *meta3) {
  if (i!=cached_point_.first) {
    this->get(i, cached_point_.second);
    cached_point_.first=i;
  }
  boost::mpl::eval_if<
    boost::mpl::and_<
      typename Dataset_t::HasMetaData_t,
      boost::mpl::not_<
        boost::is_same<
          typename Dataset_t::MetaDataType_t, 
          boost::mpl::void_
        >
      >
    >,
    MetaGetter1,
    MetaGetter2
  >::type::Do(cached_point_.second, meta1, meta2, meta3);

}

template<typename TemplateMap>
std::pair<index_t, typename Table<TemplateMap>::Point_t*> &Table<TemplateMap>::get_cached_point() {
  return cached_point_;
}

namespace table_set {
 template<typename TableType>
 struct General {
   struct type {
     static void set(TableType *table, 
         index_t i, 
         index_t j,
         double val) {
   
       if (i!=table->get_cached_point().first) {
         table->get(i, table->get_cached_point().second);
         table->get_cached_point().first=i;
       }
       table->get_cached_point().second->set(j, 
           static_cast<typename TableType::CalcPrecision_t>(val));
     }
   }; 
 };
 
 template<typename TableType>
 struct Special {
   struct type {
     static void set(TableType *table, 
         index_t i, 
         index_t j,
         double val) {
       table->get_point_collection().dense->template get<
         typename TableType::DenseBasicStorageType_t>().set(j, i, val);
     }
   };
 };

}
template<typename TemplateMap>
void Table<TemplateMap>::set(index_t i, index_t j, double value) {
  boost::mpl::eval_if<
    IsMatrixOnly_t,
    table_set::Special<Table_t>,
    table_set::General<Table_t>
  >::type::set(this, i, j, value);  
}

namespace table_set_all {
  template<typename TableType>
  struct General {
    struct type {
      static void SetAll(TableType *table, double value) {
        typename TableType::Point_t point;
        for(index_t i=0; i<table->n_entries(); ++i) {
          table->get(i, &point);
          point.SetAll(value);
        }
      }
    };
  };

  template<typename TableType>
  struct Special {
    struct type {
      static void SetAll(TableType *table, double value) {
        table->get_point_collection().dense->template get<
          typename TableType::DenseBasicStorageType_t>().SetAll(value);
      }
    };
  };
}

template<typename TemplateMap>
void Table<TemplateMap>::SetAll(double value) {
  boost::mpl::eval_if<
    IsMatrixOnly_t,
    table_set_all::Special<Table_t>,
    table_set_all::General<Table_t>
  >::type::SetAll(this, value);
}

template<typename TemplateMap>
void Table<TemplateMap>::UpdatePlus(index_t i, index_t j, double value) {
  double old_value=this->get(i, j);
  this->set(i, j, old_value+value); 
}

template<typename TemplateMap>
void Table<TemplateMap>::UpdateMul(index_t i, index_t j, double value) {
  double old_value=this->get(i, j);
  this->set(i, j, old_value*value); 
}


template<typename TemplateMap>
void Table<TemplateMap>::push_back(std::string &point) {
  data_->push_back(point);
}

template<typename TemplateMap>
void Table<TemplateMap>::push_back(Point_t &point) {
  data_->push_back(point);
}

template<typename TemplateMap>
index_t Table<TemplateMap>::n_attributes() const {
  return data_->n_attributes();
}

template<typename TemplateMap>
const std::vector<index_t> Table<TemplateMap>::dense_sizes() const {
  return data_->dense_sizes();
}

template<typename TemplateMap>
const std::vector<index_t> Table<TemplateMap>::sparse_sizes() const {
  return data_->sparse_sizes();
}

template<typename TemplateMap>
index_t  Table<TemplateMap>::n_entries() const {
  return data_->n_points();
}

template<typename TemplateMap>
index_t Table<TemplateMap>::get_node_begin(Tree_t *node) const {
  return node->begin();
}

template<typename TemplateMap>
index_t Table<TemplateMap>::get_node_end(Tree_t *node) const {
  return node->end();
}

template<typename TemplateMap>
index_t  Table<TemplateMap>::get_node_count(Tree_t *node) const {
  return node->count();
}

template<typename TemplateMap>
const typename Table<TemplateMap>::Tree_t::Bound_t &
Table<TemplateMap>::get_node_bound(const Tree_t *node) const {
  return node->bound();
}

template<typename TemplateMap>
typename Table<TemplateMap>::Tree_t::Bound_t &Table<TemplateMap>::get_node_bound(
  typename Table<TemplateMap>::Tree_t *node) {
  return node->bound();
}

template<typename TemplateMap>
typename Table<TemplateMap>::Tree_t *Table<TemplateMap>::get_node_child(typename Table<TemplateMap>::Tree_t *node, index_t i) {
  return (node->children())[i];
}

template<typename TemplateMap>
index_t Table<TemplateMap>::get_node_id(typename Table<TemplateMap>::Tree_t *node) {
  return node->node_id();
}

template<typename TemplateMap>
const index_t Table<TemplateMap>::get_node_id(
    typename Table<TemplateMap>::Tree_t *node) const {
  return node->node_id();
}
// template<typename TemplateMap>
// typename Table<TemplateMap>::Statistic_t& Table<TemplateMap>::get_node_stat(
//  typename Table<TemplateMap>::Tree_t *node) const {
//  return node->stat();
// }

// template<typename TemplateMap>
// typename Table<TemplateMap>::Statistic_t*& Table<TemplateMap>::get_node_stat_ptr(
//  typename Table<TemplateMap>::Tree_t *node) const {
//  return node->stat_ptr();
// }

/*   const Statistic_t& get_node_stat(Tree_t *node) const {
     return node->stat();
   }
*/

template<typename TemplateMap>
bool Table<TemplateMap>::node_is_leaf(Tree_t *node) const {
  return node->is_leaf();
}

template<typename TemplateMap>
typename Table<TemplateMap>::Tree_t *Table<TemplateMap>::get_node_left_child(
  typename Table<TemplateMap>::Tree_t *node) const {
  return node->left();
}

template<typename TemplateMap>
typename Table<TemplateMap>::Tree_t *Table<TemplateMap>::get_node_right_child(
  typename Table<TemplateMap>::Tree_t *node) const {
  return node->right();
}

template<typename TemplateMap>
typename Table<TemplateMap>::TreeIterator Table<TemplateMap>::get_node_iterator(
  typename Table<TemplateMap>::Tree_t *node) const {
  return TreeIterator(*this, node);
}

template<typename TemplateMap>
typename Table<TemplateMap>::TreeIterator Table<TemplateMap>::get_node_iterator(
  index_t begin, index_t count) const {
  return TreeIterator(*this, begin, count);
}


template<typename TemplateMap>
void Table<TemplateMap>::RestrictTableToCentroidsUpToLevelRecursion_(Tree_t *node,
    index_t up_to_level, index_t level,
    Table<TemplateMap> *table) {
  if (level == up_to_level || node_is_leaf(node)) {
    Point_t point;
    get_node_bound(node).CalculateMidpoint(&point);
    table->data()->push_back(point);
    
  }
  else {
    level++;
    RestrictTableToCentroidsUpToLevelRecursion_(get_node_left_child(node),
                                       up_to_level, level, table);
    RestrictTableToCentroidsUpToLevelRecursion_(get_node_right_child(node),
                                       up_to_level, level, table);
  }
}

template<typename TemplateMap>
void Table<TemplateMap>::RestrictTableToCentroidsUpToDiameterRecursion_(
  Tree_t *node,
  CalcPrecision_t diameter,
  Table<TemplateMap> *table ) {
  CalcPrecision_t current_diameter = get_node_bound(node).MaxDistanceWithinBound();
  if (diameter <= current_diameter || node_is_leaf(node)) {
    Point_t point;
    get_node_bound(node).CalculateMidpoint(&point);
    table->data()->push_back(point);
  }
  else {
    RestrictTableToCentroidsUpToDiameterRecursion_(get_node_left_child(node),
                                       diameter,
                                       table);
    RestrictTableToCentroidsUpToDiameterRecursion_(get_node_right_child(node),
                                       diameter, 
                                       table);
  }
}

template<typename TemplateMap>
void Table<TemplateMap>::RestrictTableToSamplesUpToLevelRecursion_(Tree_t *node,
    index_t up_to_level, index_t current_level, 
    index_t num_of_samples,
    Table<TemplateMap> *table) {
  if (current_level == up_to_level 
      || (node_is_leaf(node) && current_level<=up_to_level) ) {
    TreeIterator it(*this, node);
    for (index_t i = 0; i < std::min(num_of_samples, get_node_count(node)); i++) {
      Point_t p;
      it.RandomPick(&p);
      table->data()->push_back(p);
    }
  }
  else {
    current_level++;
    RestrictTableToSamplesUpToLevelRecursion_(get_node_left_child(node),
                                       up_to_level, current_level, num_of_samples, table);
    RestrictTableToSamplesUpToLevelRecursion_(get_node_right_child(node),
                                       up_to_level, current_level, num_of_samples, table);
  }
}

template<typename TemplateMap>
void Table<TemplateMap>::RestrictTableToSamplesUpToDiameterRecursion_(Tree_t *node,
    CalcPrecision_t diameter,
    index_t num_of_samples,
    Table<TemplateMap> *table) {
  CalcPrecision_t current_diameter =
    get_node_bound(node).MaxDistanceWithinBound();
  if (diameter <= current_diameter || node_is_leaf(node)) {
    TreeIterator it(*this, node);
    for (index_t i = 0; i < std::min(num_of_samples, get_node_count(node)); i++) {
      Point_t p;
      it.RandomPick(&p);
      table->data()->push_back(p);
    }
  }
  else {
    RestrictTableToSamplesUpToDiameterRecursion_(get_node_left_child(node),
                                       diameter, num_of_samples, table);
    RestrictTableToSamplesUpToDiameterRecursion_(get_node_right_child(node),
                                       diameter, num_of_samples, table);
  }
}

template<typename TemplateMap>
void Table<TemplateMap>::direct_get_(index_t point_id,
                                     typename Table<TemplateMap>::Point_t *entry) const {
  if (sort_points) {
    data_->get(point_id, entry);
  }
  else {
    data_->get(shuffled_to_real_[point_id], entry);
  }
}

template<typename TemplateMap>
index_t Table<TemplateMap>::direct_get_id_(index_t point_id) const {
  return shuffled_to_real_[point_id];
}


}; // namesapce tree
}; // namespace fl

#endif
