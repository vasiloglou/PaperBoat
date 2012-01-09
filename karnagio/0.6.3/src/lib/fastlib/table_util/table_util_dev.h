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

#ifndef FL_LITE_FASTLIB_TABLE_UTIL_TABLE_UTIL_DEV_H_
#define FL_LITE_FASTLIB_TABLE_UTIL_TABLE_UTIL_DEV_H_
#include "fastlib/table_util/table_util.h"
#include "fastlib/math/fl_math.h"

namespace fl {namespace table {
  /**
   * @brief Split a dataset into a
   *   training set and a test set
   */
  template<typename TableType>
  void TableUtil<boost::mpl::void_>::SplitPoints(TableType &references, 
       double percentage_holdout, 
       TableType *train_table, 
       TableType *test_table) {

    train_table->Init(references.data()->dense_sizes(),
                      references.data()->sparse_sizes(),
                      0);
    test_table->Init(references.data()->dense_sizes(),
                      references.data()->sparse_sizes(),
                      0);
    for(index_t i=0; i<references.n_entries(); ++i) {
      typename TableType::Point_t p;
      references.get(i, &p);
      if (fl::math::Random<double>() > percentage_holdout) {
        train_table->data()->push_back(p);
      } else {
        test_table->data()->push_back(p);
      }
    }
  }  
  
  template<typename TableType, typename DataAccessType>
  void TableUtil<boost::mpl::void_>::SplitLabeled(TableType &references,
         DataAccessType *data,
         std::string prefix, 
         std::map<signed char, boost::shared_ptr<TableType> > *new_tables) {
    
    for(index_t i=0; i<references.n_entries(); ++i) {
      typename TableType::Point_t point;
      references.get(i, &point);
      if (new_tables->find(point.meta_data().template get<0>())==new_tables->end()) {
        (*new_tables)[point.meta_data().template get<0>()].reset(new TableType());

        const std::string local_prefix=prefix.append(boost::lexical_cast<std::string>(static_cast<int>(
                  point.meta_data().template get<0>())));
        data->Attach(local_prefix,
            references.data()->dense_sizes(),
            references.data()->sparse_sizes(), 0,
            (*new_tables)[point.meta_data().template get<0>()].get());      
      }
      (*new_tables)[point.meta_data().template get<0>()]->data()->push_back(point);
    }
  
  }

  template<typename TableType, typename LabelTableType, typename DataAccessType>
  void TableUtil<boost::mpl::void_>::SplitLabeled(TableType &references,
         LabelTableType &labels, DataAccessType *data,
         std::string prefix,
         std::map<signed char, boost::shared_ptr<TableType> > *new_tables) {

    for(index_t i=0; i<references.n_entries(); ++i) {
      typename TableType::Point_t point;
      references.get(i, &point);
      if (new_tables->find(labels[i])==new_tables->end()) {

        const std::string local_prefix=prefix.append(boost::lexical_cast<std::string>(static_cast<int>(
                  labels[i]))); 
        (*new_tables)[labels[i]].reset(new TableType());
         data->Attach(local_prefix,
            references.data()->dense_sizes(),
            references.data()->sparse_sizes(), 0,
            (*new_tables)[labels[i]].get());             
      }
      (*new_tables)[labels[i]]->data()->push_back(point);
    }
 
  }

  /**
   * @brief Split dataset into a training and a test set
   *        This version also samples on the dimensions too
   */ 
  template<typename TableType1, typename TableType2>
  void TableUtil<boost::mpl::void_>::SplitDimensions(TableType1 &references, 
       double dim_percentage_holdout,
       TableType2 *train_table, 
       TableType2 *test_table) {
 
    train_table->Init(references.data()->dense_sizes(),
                      references.data()->sparse_sizes(),
                      0);
    test_table->Init(references.data()->dense_sizes(),
                     references.data()->sparse_sizes(),
                     0);

    std::vector<index_t> sizes(references.data()->dense_sizes());
    for(index_t i=0; i<references.data()->sparse_sizes().size(); ++i) {
      sizes.push_back(references.data()->sparse_sizes()[i]);
    }
    for(index_t i=0; i<references.n_entries(); ++i) {
      typename TableType1::Point_t p1;
      typename TableType2::Point_t p2;
      typename TableType2::Point_t p3;
      p2.Init(sizes);
      references.get(i, &p1);
      p2.meta_data()=p1.meta_data();
      p3.Init(sizes);
      p3.meta_data()=p1.meta_data();
      for(typename TableType1::Point_t::iterator it = p1.begin();
          it!=p1.end(); ++it) {
        if (fl::math::Random<double>() > dim_percentage_holdout) {
          p2.set(it.attribute(), it.value());
        } else {
          p3.set(it.attribute(), it.value());
        }
      }
      train_table->data()->push_back(p2);
      test_table->data()->push_back(p3);
    }
  }


  /**
   * @brief Splits a dataset into a training
   *        and a test, but now it splits
   *        the table thata has the labels
   */
  template<typename DataTableType,
           typename LabelTableType>
  void TableUtil<boost::mpl::void_>::SplitPoints(DataTableType &references,
       LabelTableType &references_labels, 
       double percentage_holdout, 
       DataTableType *train_table,
       LabelTableType *train_table_labels,
       DataTableType  *test_table,
       LabelTableType *test_table_labels) {
 
    train_table->Init(references.data()->dense_sizes(),
                      references.data()->sparse_sizes(),
                      0);
    train_table_labels->Init(references_labels.data()->dense_sizes(),
        references_labels.data()->sparse_sizes(),
        0);
    test_table->Init(references.data()->dense_sizes(),
                      references.data()->sparse_sizes(),
                      0);
    test_table_labels->Init(references_labels.data()->dense_sizes(),
        references_labels.data()->sparse_sizes(),
        0);
    for(index_t i=0; i<references.n_entries(); ++i) {
      typename DataTableType::Point_t p;
      typename LabelTableType::Point_t label;
      references.get(i, &p);
      references_labels.get(i, &label);
      if (fl::math::Random<double>() > percentage_holdout) {
        train_table->data()->push_back(p);
        train_table_labels->data()->push_back(label);
      } else {
        test_table->data()->push_back(p);
        test_table_labels->data()->push_back(label);
      }
    }
  }  


 
  /**
   * @brief Splits a dataset into smaller ones
   */
  template<typename TableType>
  void TableUtil<boost::mpl::void_>::SplitPoints(TableType &references,
             double percentage, // the percentage of reference 
                                // that will be split into tables
             std::vector<TableType*> *new_tables) {
    
    for(index_t i=0; i<new_tables->size(); ++i) {
      (*new_tables)[i]=new TableType();
      new_tables->Init(references->data().dense_sizes(),
          references->data().sparse_sizes(), 0);
    }
    int num_of_tables=new_tables->size();  
    for(index_t i=0; i<references.n_entries(); ++i) {
      typename TableType::Point_t p;
      references.get(i, &p);
      double random_number=fl::math::Random<double>();
      if (random_number < percentage) {
        int random_pick=static_cast<int>(fl::math::Random<double>()*num_of_tables);
        (*new_tables)[random_pick].data()->push_back(p);
      }
    }
  }

  /**
   * @brief Splits a dataset into smaller ones
   */
  template<typename TableType>
  void TableUtil<boost::mpl::void_>::OverSplitPoints(TableType &references,
             double oversample_percentage, // the total size of all new tables 
                                           // will be (1+oversample_percentage)*references
             std::vector<TableType*> *new_tables) {
    
    for(index_t i=0; i<new_tables->size(); ++i) {
      (*new_tables)[i]=new TableType();
      new_tables->Init(references->data().dense_sizes(),
          references->data().sparse_sizes(), 0);
    }
    int num_of_tables=new_tables->size();  
    for(index_t i=0; i<references.n_entries(); ++i) {
      typename TableType::Point_t p;
      references.get(i, &p);
      double random_number=fl::math::Random<double>();
      int random_pick=static_cast<int>(fl::math::Random<double>()*num_of_tables);
      (*new_tables)[random_pick].data()->push_back(p);
      if (random_number < oversample_percentage) {
        int random_pick1=static_cast<int>(fl::math::Random<double>()*num_of_tables);
        while (random_pick1==random_pick) {
          random_pick1=static_cast<int>(fl::math::Random<double>()*num_of_tables);
        }
        (*new_tables)[random_pick1].data()->push_back(p);
      }
    }
  }

  /**
   * @brief Splits a dataset into smaller ones
   *        This version also samples the dimensions
   */
  template<typename TableType1, typename TableType2>
  void TableUtil<boost::mpl::void_>::SplitDimensions(TableType1 &references,
             double percentage, // the percentage of the references
                                // we will use
             std::vector<TableType2 *> *new_tables) {

    for(index_t i=0; i<references.n_entries(); ++i) {
      typename TableType1::Point_t p1;
      typename TableType2::Point_t p2;
      references.get(i, &p1);
      for(typename TableType1::Point_t::iterator it=p1.begin();
          it!=p1.end(); ++it) {
        double random_number=fl::math::Random<double>();
        if (random_number>percentage) {
          continue;
        } else {
          int random_pick=static_cast<int>(fl::math::Random<double>()*new_tables->size());
          (*new_tables)[random_pick]->get(i, &p2);
          p2.set(it.attribute(), it.value());
        }
      }
    }  
  }

  /**
   * @brief Splits a dataset into smaller ones
   *        This version also samples the dimensions
   */
  template<typename TableType1, typename TableType2>
  void TableUtil<boost::mpl::void_>::OverSplitDimensions(TableType1 &references,
             double over_percentage, // the total sum of files
             // will (be 1+over_percentage)*references.size()
             std::vector<TableType2 *> *new_tables) {

    for(index_t i=0; i<references.n_entries(); ++i) {
      typename TableType1::Point_t p1;
      typename TableType2::Point_t p2;
      references.get(i, &p1);
      for(typename TableType1::Point_t::iterator it=p1.begin();
          it!=p1.end(); ++it) {
        double random_number=fl::math::Random<double>();
        int random_pick=static_cast<int>(fl::math::Random<double>()*new_tables->size());
        (*new_tables)[random_pick]->get(i, &p2);
        p2.set(it.attribute(), it.value());
        int random_pick1=static_cast<int>(fl::math::Random<double>()*new_tables->size());
        random_number=fl::math::Random<double>();
        if (random_number<over_percentage) {
          while(random_pick==random_pick1) {
            random_pick1=static_cast<int>(fl::math::Random<double>()*new_tables->size());
          }
          (*new_tables)[random_pick1]->get(i, &p2);
          p2.set(it.attribute(), it.value());
        }
      }
    }  
  }


}} // namespaces

#endif
