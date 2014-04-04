/*
Copyright © 2010, Ismion Inc
All rights reserved.
http://www.ismion.com/

Redistribution and use in source and binary forms, with or without
modification IS NOT permitted without specific prior written
permission. Further, neither the name of the company, Ismion
LLC, nor the names of its employees may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE ISMION INC "AS IS" AND ANY
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

#ifndef FL_LITE_FASTLIB_TABLE_TABLE_SERIALIZATION_H_
#define FL_LITE_FASTLIB_TABLE_TABLE_SERIALIZATION_H_
#include "table.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/data/multi_dataset_reset.h"
#include "boost/archive/archive_exception.hpp"

namespace fl {
/**
 * @brief namespace table, contains table structures that look like database tables
 */
namespace table {
  template<typename TemplateMap>
  template<typename Archive>
  void Table<TemplateMap>::save(Archive &ar, const unsigned int version) const {
    try {
      ar << boost::serialization::make_nvp("real_to_shuffled", real_to_shuffled_);
      ar << boost::serialization::make_nvp("shuffled_to_real", shuffled_to_real_);
      ar << boost::serialization::make_nvp("sampled_data", sampled_data_);

      ar << boost::serialization::make_nvp("tree", tree_);
      ar << boost::serialization::make_nvp("num_of_nodes", num_of_nodes_);
      ar << boost::serialization::make_nvp("leaf_size", leaf_size_);
      ar << boost::serialization::make_nvp("metric_type_id", metric_type_id_);
      ar << boost::serialization::make_nvp("data", data_);
      ar << boost::serialization::make_nvp("filename", filename_);

    }
    catch(const boost::archive::archive_exception &e) {
      fl::logger->Die()<< "Table archiving (save): "<< e.what();
    }
  }
  
  template<typename TemplateMap>
  template<typename Archive>
  void Table<TemplateMap>::load(Archive &ar, const unsigned int version) {
    try {
      ar >> boost::serialization::make_nvp("real_to_shuffled", real_to_shuffled_);
      ar >> boost::serialization::make_nvp("shuffled_to_real", shuffled_to_real_);
      ar >> boost::serialization::make_nvp("sampled_data", sampled_data_);

      ar >> boost::serialization::make_nvp("tree", tree_);
      ar >> boost::serialization::make_nvp("num_of_nodes", num_of_nodes_);
      ar >> boost::serialization::make_nvp("leaf_size", leaf_size_);
      ar >> boost::serialization::make_nvp("metric_type_id", metric_type_id_);
      ar >> boost::serialization::make_nvp("data", data_);
      ar >> boost::serialization::make_nvp("filename", filename_);
    }
    catch(const boost::archive::archive_exception &e) {
      fl::logger->Die()<< "Table archiving (load): "<< e.what();
    }
  }
}}

#endif
