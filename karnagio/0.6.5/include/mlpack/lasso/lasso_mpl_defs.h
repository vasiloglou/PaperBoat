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
#ifndef MLPACK_REGRESSION_LASSO_MPL_DEFS_H
#define MLPACK_REGRESSION_LASSO_MPL_DEFS_H

#include "fastlib/data/multi_dataset.h"
#include "fastlib/dense/matrix.h"
#include "fastlib/table/table.h"

namespace fl {
namespace ml {

namespace gauss_seidel_lasso_inst {

typedef fl::data::MultiDataset <
boost::mpl::map3 <
boost::mpl::pair<fl::data::DatasetArgs::DenseTypes, boost::mpl::vector1<double> >,
boost::mpl::pair<fl::data::DatasetArgs::StorageType, fl::data::DatasetArgs::StorageType::Compact>,
boost::mpl::pair<fl::data::DatasetArgs::CalcPrecision, double>  > > Dataset;

typedef
boost::mpl::map<>::type
TreeOpts;

typedef boost::mpl::map2 <
boost::mpl::pair < fl::table::TableArgs::DatasetType,
Dataset > ,
boost::mpl::pair < fl::table::TableArgs::SortPoints,
boost::mpl::bool_<true> >
>::type TableOpts;

typedef boost::mpl::map2 <
boost::mpl::pair<fl::tree::TreeArgs, TreeOpts>,
boost::mpl::pair<fl::table::TableArgs, TableOpts>
>::type TemplateMap;

typedef fl::tree::Tree<TemplateMap> DefaultTreeType;
typedef fl::table::Table<TemplateMap> DefaultTableType;

};
};
};

#endif
