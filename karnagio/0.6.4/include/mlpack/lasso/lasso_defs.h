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

#ifndef FL_LITE_MLPACK_LASSO_LASSO_DEFS_H
#define FL_LITE_MLPACK_LASSO_LASSO_DEFS_H
#include "lasso.h"
#include "gauss_seidel_lasso_defs.h"
#include "fastlib/workspace/task.h"

template<typename DataAccessType, typename BranchType>
int fl::ml::Lasso<boost::mpl::void_>::Main(
  DataAccessType *data,
  const std::vector<std::string> &args) {
  FL_SCOPED_LOG(Lasso);
  return GaussSeidelLasso<boost::mpl::void_>::Main<
    DataAccessType, BranchType>(data, args);
}

template<typename DataAccessType>
void fl::ml::Lasso<boost::mpl::void_>::Run(
      DataAccessType *data,
      const std::vector<std::string> &args) {
  fl::ws::Task<
    DataAccessType,
    &Main<
      DataAccessType, 
      typename DataAccessType::Branch_t
    >
  > task(data, args);
  data->schedule(task); 
}

#endif

