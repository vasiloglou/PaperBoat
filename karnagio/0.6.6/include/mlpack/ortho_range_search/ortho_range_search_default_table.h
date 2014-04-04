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
#ifndef MLPACK_ORTHO_RANGE_SEARCH_ORTHO_RANGE_SEARCH_DEFAULT_TABLE_H
#define MLPACK_ORTHO_RANGE_SEARCH_ORTHO_RANGE_SEARCH_DEFAULT_TABLE_H

#include "fastlib/data/multi_dataset.h"
#include "fastlib/tree/kdtree.h"
#include "fastlib/tree/hyper_rectangle_tree.h"
#include "fastlib/dense/matrix.h"
#include "fastlib/table/table.h"
#include "fastlib/table/default_table.h"

namespace fl {
namespace ml {

namespace ortho_range_search_inst {
// Define the table for the window queries.
struct WindowTreeArgs : public fl::tree::TreeArgs {
  typedef fl::tree::HyperRectangleTree TreeSpecType;
  typedef boost::mpl::bool_<false> StoreLevel;
  typedef fl::data::MonolithicPoint<double> BoundType;
};

struct WindowTableMap : public fl::table::DefaultTableMap {
  typedef WindowTreeArgs TreeArgs;
};

typedef fl::table::Table<WindowTableMap> WindowTableType;
typedef WindowTableType::Tree_t WindowTreeType;

// Define the table for the reference points.
struct ReferenceTreeArgs : public fl::tree::TreeArgs {
  typedef fl::tree::MedianKdTree TreeSpecType;
  typedef boost::mpl::bool_<false> StoreLevel;
  typedef fl::tree::GenHrectBound<double, double, 2> BoundType;
};

struct ReferenceTableMap : public fl::table::DefaultTableMap {
  typedef ReferenceTreeArgs TreeArgs;
};

typedef fl::table::Table<ReferenceTableMap> ReferenceTableType;
typedef ReferenceTableType::Tree_t ReferenceTreeType;

// Output table type.
struct OutputDatasetArgs : public fl::data::DatasetArgs {
  typedef boost::mpl::vector1<bool> DenseTypes;
  typedef bool CalcPrecision;
  typedef fl::data::DatasetArgs::Compact StorageType;
};

struct OutputTableArgs : public fl::table::TableArgs {
  typedef fl::data::MultiDataset<OutputDatasetArgs> DatasetType;
};

struct OutputTableMap : public fl::table::DefaultTableMap {
  typedef OutputTableArgs TableArgs;
};

typedef fl::table::Table<OutputTableMap> OutputTableType;
typedef OutputTableType::Tree_t OutputTreeType;
};
};
};

#endif
