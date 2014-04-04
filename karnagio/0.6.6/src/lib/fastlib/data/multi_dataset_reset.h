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
#ifndef FL_LITE_FASTLIB_DATA_MULTI_DATASET_RESET_H_
#define FL_LITE_FASTLIB_DATA_MULTI_DATASET_RESET_H_
#include "fastlib/data/multi_dataset.h"

namespace fl { namespace data {

  template<typename ParameterList>
  void MultiDataset<ParameterList>::Reset() {
    boost::mpl::for_each<DenseTypeList_t>(ResetIterators<DenseBox, DenseIterators>(
                                            &dense_, &dense_its_));
    boost::mpl::for_each<SparseTypeList_t>(ResetIterators<SparseBox, SparseIterators>(
                                             &sparse_, &sparse_its_));
    if (HasMetaData_t::value == true) {
      meta_it_ = meta_.begin();
    }
  }
}}
#endif
