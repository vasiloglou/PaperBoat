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
#ifndef FL_LITE_FASTLIB_TABLE_R_DATA_ACCESS_H
#define FL_LITE_FASTLIB_TABLE_R_DATA_ACCESS_H

#include "fastlib/data/multi_dataset_dev.h"
#include "table.h"
#include "default_table.h"
#include "uinteger_table.h"
#include "table_vector.h"
#include "default_sparse_int_table.h"
#include "default_sparse_double_table.h"

#include "R.h"
#include "Rinternals.h"
#include "Rdefines.h"

namespace fl {
namespace table {

class RDataAccess {
  public:
    typedef fl::table::DefaultTable DefaultTable_t;
    typedef fl::table::DefaultSparseIntTable DefaultSparseIntTable_t;
    typedef fl::table::DefaultSparseDoubleTable DefaultSparseDoubleTable_t;
    typedef fl::table::UIntegerTable UIntegerTable_t;
    template<typename PrecisionType>
    class TableVector : public fl::table::TableVector<PrecisionType> {
    };

    template<typename TableParamsType>
    void Attach(const std::string &name, 
        fl::table::Table<TableParamsType> * const table);
    
    template<typename TableParamsType>
    void Attach(const std::string &name,
                std::vector<index_t> dense_sizes,
                std::vector<index_t> sparse_sizes,
                const index_t num_of_points,
                fl::table::Table<TableParamsType> * const table);
    
    template<typename TableParamsType>
    void Detach(fl::table::Table<TableParamsType> &table);

    template<typename TableParamsType>
    void Purge(fl::table::Table<TableParamsType> &table);

    template<typename TableParamsType1, typename TableParamsType2>
    void TieLabels(fl::table::Table<TableParamsType1> *table,
        fl::table::Table<TableParamsType2> *labels);
    
    private:
      SEXP *environment_;

      void Get(const std::string &name, 
               double **ptr, 
               index_t *num_of_points,
               index_t *n_attributes);
     
  };
}
}

#include "r_data_access_defs.h"
#endif
