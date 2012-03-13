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

#ifndef PAPERBOAT_HPCC_SRC_WORKSPACE_H_
#define PAPERBOAT_HPCC_SRC_WORKSPACE_H_
#include "fastlib/workspace/workspace.h"

namespace fl { namespace hpcc {

  class WorkSpace : public fl::ws::WorkSpace {
    public:
      typedef typename boost::mpl::map<
        boost::mpl::pair<
          double,
          DefaultSparseDoubleTable_t
        >,
        boost::mpl::pair<
          int32,
          DefaultSparseIntTable_t
        >,
        boost::mpl::pair<
          int64,
          DefaultSparseIntTable_t
        >,
        boost::mpl::pair<
          uint8,
          fl::table::sparse::labeled::balltree::uint8::Table 
        >
      >::type SparseTables_t;

      typedef typename boost::mpl::map<
        boost::mpl::pair<
          double,
          DefaultTable_t
        >,
        boost::mpl::pair<
          uint8,
          DefaultTable_t
        >,
        boost::mpl::pair<
          int32,
          IntegerTable_t
        >,
        boost::mpl::pair<
          uint32,
          UIntegerTable_t
        >,
        boost::mpl::pair<
          uint64,
          UIntegerTable_t
        >,
        boost::mpl::pair<
          index_t,
          IntegerTable_t
        >
      >::type DenseTables_t;


      template<typename T>
      void LoadDenseHPCCDataSet(const std::string &name,
          index_t n_attributes,
          index_t n_entries,
          const void *in_data);

      template<typename T>
      void ExportDenseHPCCDataSet(const std::string &name,
          void **in_data, uint32 *length);       

      /**
       *  @brief it requires the input dataset to be sorted 
       *  according to file_id, id, number
       *  use ECL SORT(mydataset, file_id, id, number, LOCAL);
       *  The arguments will contain input files with the _in suffix
       *  The filenames must be of the form storage:number
       *  where storage is either dense or sparse. The number is 
       *  between 0-255
       */
      template<typename T>
      void LoadAllDenseHPCCDataSets(const std::string &arguments,
          const char *in_data,
          const uint64 data_len);
              
      /**
       *  @brief it requires the input dataset to be sorted 
       *  according to file_id, id, number
       *  use ECL SORT(mydataset, file_id, id, number, LOCAL);
       *  The arguments will contain input files with the _in suffix
       *  The filenames must be of the form storage:number
       *  where storage is either dense or sparse. The number is 
       *  between 0-255
       */
      template<typename T>
      void LoadAllSparseHPCCDataSets(const std::string &arguments,
          const char *in_data,
          const uint64 data_len);        
     /**
      *  @brief     
      *  The arguments will contain input files with the _out suffix
      *  The filenames must be of the form storage:number
      *  where storage is either dense or sparse. The number is 
      *  between 0-255
      */
      template<typename T>
      void ExportAllDenseHPCCDataSets(const std::string &arguments,
          void **out_data,
          uint32 *data_len);     
      /**
      *  @brief     
      *  The arguments will contain input files with the _out suffix
      *  The filenames must be of the form storage:number
      *  where storage is either dense or sparse. The number is 
      *  between 0-255
      */
      template<typename T>
      void ExportAllSparseHPCCDataSets(const std::string &arguments,
          void **out_data,
          uint64 *data_len); 
  };

}}


#endif
