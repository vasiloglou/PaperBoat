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

#ifndef FL_LITE_FASTLIB_TABLE_CONVERT_TABLE_TO_DOUBLE_TABLE_H_
#define FL_LITE_FASTLIB_TABLE_CONVERT_TABLE_TO_DOUBLE_TABLE_H_
#include "boost/mpl/fold.hpp"
#include "boost/mpl/vector.hpp"
#include "fastlib/data/multi_dataset.h"
#include "fastlib/table/table.h"

namespace fl {

  namespace ConvertToDouble_namsepace {
    template<typename T>
    class dummy {
      typedef double type;
    };
  }

template<typename TableType>
class ConvertToDouble {
  public:
    typedef typename boost::mpl::fold<
      typename TableType::Dataset_t::DenseTypeList_t,
      boost::mpl::vector<>,
      boost::mpl::push_back<
        boost::mpl::placeholders::_1,
        ConvertToDouble_namsepace::dummy<boost::mpl::placeholders::_2>::type
      >
    >::type DenseTypeList_t;

    typedef typename boost::mpl::fold<
      typename TableType::Dataset_t::SparseTypeList_t,
      boost::mpl::vector<>,
      boost::mpl::push_back<
        boost::mpl::placeholders::_1,
        ConvertToDouble_namsepace::dummy<boost::mpl::placeholders::_2>::type
      >
    >::type SparseTypeList_t;

    typedef typename TableType::MetaDataType_t MetaDataType_t;
    struct TableMap {
      struct TableArgs {
        struct DatasetArgs : public fl::data::DatasetArgs {
          typedef SparseTypeList_t SparseTypes;
          typedef DenseTypeList_t DenseType;
          typedef MetaDataType_t MetaDataType;
          typedef double CalcPrecision;
          typedef typename TableType::Dataset_t::Storage_t  StorageType;
      };
      typedef fl::data::MultiDataset<DatasetArgs> DatasetType;
      typedef boost::mpl::bool_<true> SortPoints;
    };
    typedef typename TableType::TreeArgs_t TreeArgs;
  };
  typedef fl::table::Table<TableMap> type;
};

}

#endif

