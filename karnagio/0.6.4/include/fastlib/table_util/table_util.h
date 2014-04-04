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

#ifndef FASTLIB_TABLE_UTIL_TABLE_UTIL_H_
#define FASTLIB_TABLE_UTIL_TABLE_UTIL_H_
#include <vector>
#include <map>
#include <string>
#include "boost/mpl/void.hpp"
#include "boost/program_options.hpp"
#include "boost/shared_ptr.hpp"
#include "fastlib/base/base.h"

/**
 * @brief This file contains utilities for manipulating fl-lite files
 *  spliting them, picking cross validating samples etc.
 *
 */
namespace fl { namespace table {
 template<typename>
 class TableUtil;

 template<>
 class TableUtil<boost::mpl::void_> {
   public:
     /**
      * @brief Split a dataset into a
      *   training set and a test set
      */
     template<typename TableType>
     static void SplitPoints(TableType &references, 
          double percentage_holdout, 
          TableType *train_table, 
          TableType *test_table);

     /**
      * @brief Splits the reference file in files according to their labels
      */
     template<typename TableType, typename DataAccessType>
     static void SplitLabeled(TableType &references,
         DataAccessType *data,
         std::string prefix, 
         std::map<signed char, boost::shared_ptr<TableType> > *new_tables);     
     /**
      * @brief Splits the reference file in files according to their labels,
      *        the labels are stored on a separate file 
      */
    template<typename TableType, typename LabelTableType, typename DataAccessType>
    static void SplitLabeled(TableType &references,
         LabelTableType &labels, DataAccessType *data,
         std::string prefix,
         std::map<signed char, boost::shared_ptr<TableType> > *new_tables);
     /**
      * @brief Split dataset into a training and a test set
      *        This version also samples on the dimensions too
      */ 
     template<typename TableType1, typename TableType2>
     static void SplitDimensions(TableType1 &references, 
          double dim_percentage_holdout,
          TableType2 *train_table, 
          TableType2 *test_table);
   
   
     /**
      * @brief Splits a dataset into a training
      *        and a test, but now it splits
      *        the table thata has the labels
      */
     template<typename DataTableType,
              typename LabelTableType>
     static void SplitPoints(DataTableType &references,
          LabelTableType &references_labels, 
          double percentage_holdout, 
          DataTableType *train_table,
          LabelTableType *train_table_labels,
          DataTableType  *test_table,
          LabelTableType *test_table_labels);

     /**
      * @brief Splits a dataset into smaller ones
      */
     template<typename TableType>
     static void SplitPoints(TableType &references,
                double percentage, // the percentage of reference 
                                   // that will be split into tables
                std::vector<TableType*> *new_tables);

     /**
      * @brief Splits a dataset into smaller ones
      */
     template<typename TableType>
     static void OverSplitPoints(TableType &references,
                double oversample_percentage, // the total size of all new tables 
                                              // will be (1+oversample_percentage)*references
                std::vector<TableType*> *new_tables);

   
     /**
      * @brief Splits a dataset into smaller ones
      *        This version also samples the dimensions
      */
     template<typename TableType1, typename TableType2>
     static void SplitDimensions(TableType1 &references,
                double percentage, // the percentage of the references
                                   // we will use
                std::vector<TableType2 *> *new_tables);
   
  
     /**
      * @brief Splits a dataset into smaller ones
      *        This version also samples the dimensions
      */
     template<typename TableType1, typename TableType2>
     static void OverSplitDimensions(TableType1 &references,
                double over_percentage, // the total sum of files
                // will (be 1+over_percentage)*references.size()
                std::vector<TableType2 *> *new_tables);
    
    template<typename TableType>
    struct Core {
      template<typename DataAccessType>
      static int Main(DataAccessType *data,
                      boost::program_options::variables_map &vm);

    };
    template<typename DataAccessType, typename BranchType> 
    static int Main(DataAccessType *data,
                      const std::vector<std::string> &args);
    
     
 };  

}}

#endif
