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

#ifndef FASTLIB_TABLE_TABLE_VECTOR_H_
#define FASTLIB_TABLE_TABLE_VECTOR_H_

#include "boost/mpl/map.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/mpl/int.hpp"
#include "boost/mpl/bool.hpp"
#include "table.h"
#include "fastlib/base/mpl.h"
#include "fastlib/tree/default_kdtree.h"
#include "fastlib/data/multi_dataset.h"


namespace fl {
namespace table {

  template<typename PrecisionType>
  struct TableVectorMap {
    struct DatasetArgs : public fl::data::DatasetArgs {
      typedef boost::mpl::vector1<PrecisionType> DenseTypes;
      typedef double CalcPrecisionType;
      typedef fl::data::DatasetArgs::Extendable StorageType;
    };

    struct TableArgs : public fl::table::TableArgs {
      typedef fl::data::MultiDataset<DatasetArgs> DatasetType;
      typedef boost::mpl::bool_<false> SortPoints;
    };

    struct TreeArgs : public fl::tree::KdTreeDefaultArgs {
      typedef fl::math::LMetric<2> MetricType;
      typedef boost::mpl::bool_<false> SortPoints;
    };

  };

  /**
   *  @brief TableVector is a special Table that works as a Vector
   */
  template<typename PrecisionType>
  class TableVector : public Table<TableVectorMap<PrecisionType> > {
    public:
      typedef PrecisionType value_type;
      typedef double CalcPrecision_t;  
      void Init(const std::string &file, const char* mode) {
        Table<TableVectorMap<PrecisionType> >::Init(file, mode);
        if (this->n_attributes()!=1) {
          fl::logger->Die()<<"You initialized a TableVector with a file that has multidimensional "
            "("<< this->n_attributes() << ")"<< " points. It can only contain one dimensional points";
        }
      }
      
      void Init(const std::string &name,
                const std::vector<index_t>& dense_dimensions,
                const std::vector<index_t>& sparse_dimensions,
                index_t num_of_points) {
        if (sparse_dimensions.size()!=0) {
          fl::logger->Die() << "This is a TableVector, you cannot initialize it with sparse dimensions.";
        }
        
        if (dense_dimensions.size()!=1 || dense_dimensions[0]!=1) {
          fl::logger->Die()<<"This is a TableVector, you can only initialize one dimension with 1";
        }
        Table<TableVectorMap<PrecisionType> >::Init(name, dense_dimensions, 
            sparse_dimensions, num_of_points);
      } 

      void Init(index_t num_of_points) {
        std::vector<index_t> dense_dimensions(1);
        dense_dimensions[0]=1;
        std::vector<index_t> sparse_dimensions;
        Table<TableVectorMap<PrecisionType> >::Init(dense_dimensions, 
            sparse_dimensions, num_of_points);

      }

      void set(index_t ind, PrecisionType value) {
        typename Table<TableVectorMap<PrecisionType> >::Point_t p;
        this->get(ind, &p);
        p.set(0, value);
      }

      const PrecisionType &operator[](index_t ind) const {
        typename Table<TableVectorMap<PrecisionType> >::Point_t p;
        this->get(ind, &p);
        return p[0];
      }

      const size_t size() const {
        return this->n_entries();
      }
  };
}} //namespaces
#endif
