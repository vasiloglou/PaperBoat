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
#ifndef FL_LITE_FASTLAB_TABLE_DEFAULT_TABLE_H_
#define FL_LITE_FASTLAB_TABLE_DEFAULT_TABLE_H_

#include "boost/mpl/vector.hpp"
#include "boost/mpl/int.hpp"
#include "boost/mpl/bool.hpp"
#include "table.h"
#include "fastlib/base/mpl.h"
#include "fastlib/tree/default_kdtree.h"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/table/table_defs.h"
#include "fastlib/data/multi_dataset_defs.h"

namespace fl {
namespace table {
struct DefaultDatasetArgs : public fl::data::DatasetArgs {
  typedef boost::mpl::vector1<double> DenseTypes;
  typedef double CalcPrecision;
  typedef fl::data::DatasetArgs::Compact StorageType;
};

struct DefaultTableArgs : public fl::table::TableArgs {
  typedef fl::data::MultiDataset<DefaultDatasetArgs> DatasetType;
  typedef boost::mpl::bool_<true> SortPoints;
};

struct DefaultUnlabeledKdTreeArgs : public fl::tree::KdTreeDefaultArgs {
  typedef fl::math::LMetric<2> MetricType;
};

struct DefaultTableMap {
  typedef DefaultUnlabeledKdTreeArgs TreeArgs;
  typedef DefaultTableArgs TableArgs;
};

typedef Table<DefaultTableMap> DefaultTable;

struct DefaultLabeledDatasetArgs : public fl::data::DatasetArgs {
  typedef boost::mpl::vector1<double> DenseTypes;
  typedef double CalcPrecision;
  typedef fl::data::DatasetArgs::Compact StorageType;
  typedef fl::MakeIntIndexedStruct <
  boost::mpl::vector3<signed char, double, int>
  >::Generated MetaDataType;
};

struct DefaultLabeledTableArgs {
  typedef fl::data::MultiDataset<DefaultLabeledDatasetArgs> DatasetType;
  typedef boost::mpl::bool_<true> SortPoints;
};

struct DefaultLabeledKdTreeArgs : public fl::tree::KdTreeDefaultArgs {
  typedef fl::math::LMetric<2> MetricType;

};
struct DefaultLabeledTableMap {
  typedef DefaultLabeledKdTreeArgs TreeArgs;
  typedef DefaultLabeledTableArgs TableArgs;
};

typedef Table<DefaultLabeledTableMap> DefaultLabeledTable;


}
}

#endif

