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
#ifndef FL_LITE_FASTLIB_TABLE_FILE_DATA_ACCESS_DEFS_H_
#define FL_LITE_FASTLIB_TABLE_FILE_DATA_ACCESS_DEFS_H_

#include <fstream>
#include <iostream>
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/archive/xml_oarchive.hpp"
#include "boost/archive/xml_iarchive.hpp"
#include "boost/archive/binary_oarchive.hpp"
#include "boost/archive/binary_iarchive.hpp"
#include "file_data_access.h"

namespace fl {
namespace table {
template<typename TableParamsType>
void FileDataAccess::Attach(const std::string &name,
                            fl::table::Table<TableParamsType> * const table) {
  /*
  try {
    std::ifstream ifs(name.c_str());
    if (!ifs.good()) {
      fl::logger->Die()<<"Failed to open file: " <<name;
    }
    boost::archive::text_iarchive ia(ifs);
    ia >> *table;
    return;
  }
  catch(const boost::archive::archive_exception &e) {
  
  }
   
  try {
    std::ifstream ifs(name.c_str());
    if (!ifs.good()) {
      fl::logger->Die()<<"Failed to open file: " <<name;
    }
    boost::archive::xml_iarchive ia(ifs);
    ia >> BOOST_SERIALIZATION_NVP(*table);
    return;
  }
  catch(const boost::archive::archive_exception &e) {
  
  }

  try {
    std::ifstream ifs(name.c_str());
    if (!ifs.good()) {
      fl::logger->Die()<<"Failed to open file: " <<name;
    }
    boost::archive::binary_iarchive ia(ifs);
    ia >> *table;
    return;
  }
  catch(const boost::archive::archive_exception &e) {
  
  }  
*/
  table->data()->Init(name, "r");
}

template<typename TableParamsType>
void FileDataAccess::Attach(const std::string &name,
                            std::vector<index_t> dense_sizes,
                            std::vector<index_t> sparse_sizes,
                            const index_t num_of_points,
                            fl::table::Table<TableParamsType> * const table) {

  table->Init(name, dense_sizes, sparse_sizes, num_of_points);
}

template<typename TableParamsType>
void FileDataAccess::Detach(fl::table::Table<TableParamsType> &table) {

}

template<typename TableParamsType>
void FileDataAccess::Purge(fl::table::Table<TableParamsType> &table) {
  table.Save();
}

template<typename TableParamsType>
void FileDataAccess::Purge(fl::table::Table<TableParamsType> &table,
    const std::string &serialization) {
  if (serialization=="native") {
    table.Save();
    return;
  }

  fl::logger->Die() << "Serialization not supported yet";
  /*
  std::ofstream ofs(table.filename().c_str());
  if (!ofs.good()) {
    fl::logger->Die()<<"Failed to open file: " <<table.filename();
  }

  try {
    if (serialization=="text") {
       boost::archive::text_oarchive oa(ofs);
       oa << table; 
       return;
    }
  }
  catch(const boost::archive::archive_exception &e) {
    fl::logger->Die() << "Something went wrong while serializing\n"
      << e.what();  
  }
  try {
    if (serialization=="xml") {
      boost::archive::xml_oarchive oa(ofs);
      oa << BOOST_SERIALIZATION_NVP(table);
      return; 
    }
  }
  catch(const boost::archive::archive_exception &e) {
    fl::logger->Die() << "Something went wrong while serializing\n"
      << e.what();  
  }
 
  try {
    if (serialization=="binary") {
      boost::archive::binary_oarchive oa(ofs);
      oa << table;
      return;  
    }
  }
  catch(const boost::archive::archive_exception &e) {
    fl::logger->Die() << "Something went wrong while serializing\n"
      << e.what();  
  }
  */
}

template<typename TableParamsType1, typename TableParamsType2>
void FileDataAccess::TieLabels(fl::table::Table<TableParamsType1> *table,
                               fl::table::Table<TableParamsType2> *labels) {
  if (labels->n_entries() != table->n_entries()) {
    fl::logger->Die() << "Labels and table must have the same number of entries()";
  }
  typename fl::table::Table<TableParamsType2>::Point_t labels_vector;
  labels->get(0, &labels_vector);
  for (index_t i = 0; i < table->n_entries(); ++i) {
    typename fl::table::Table<TableParamsType1>::Point_t p;
    table->get(i, &p);
    p.meta_data().template get<0>() = labels_vector[i];
  }
}


} // table
} // fl

#endif
