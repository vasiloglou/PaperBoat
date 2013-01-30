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
#ifndef FL_LITE_FASTLAB_TABLE_UINTEGER_TABLE_H_
#define FL_LITE_FASTLAB_TABLE_UINTEGER_TABLE_H_

#include "boost/mpl/vector.hpp"
#include "boost/mpl/int.hpp"
#include "boost/mpl/bool.hpp"
#include "table.h"
#include "fastlib/base/mpl.h"
#include "fastlib/tree/default_kdtree.h"
#include "fastlib/data/multi_dataset.h"


namespace fl {
namespace table {
  struct  UIntegerTableMap {
    struct DatasetArgs : public fl::data::DatasetArgs {
      typedef boost::mpl::vector1<unsigned int> DenseTypes;
      typedef double CalcPrecision;
      typedef fl::data::DatasetArgs::Compact StorageType;
      typedef fl::MakeIntIndexedStruct <
        boost::mpl::vector3<signed char, double, int>
      >::Generated MetaDataType;
    };

    struct TableArgs : public fl::table::TableArgs {
      typedef fl::data::MultiDataset<DatasetArgs> DatasetType;
      typedef boost::mpl::bool_<true> SortPoints;
    };

    struct TreeArgs : public fl::tree::KdTreeDefaultArgs {
      typedef fl::math::LMetric<2> MetricType;
    };
  };

  typedef Table<UIntegerTableMap> UIntegerTable;
}}

#endif

