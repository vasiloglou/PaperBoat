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
#include <vector>
#include <string>
#include "mlpack/clustering/kmeans_dev.h"
#include "mlpack/xmeans/xmeans_dev.h"
#include "mlpack/clustering/kmeans_cv_dev.h"
#include "fastlib/workspace/workspace_defs.h"
#include "../../../ecl-pb-glue/workspace.h"
#include "fastlib/table/branch_on_table_dev.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/data/multi_dataset_dev.h"

template void fl::ml::KMeans<boost::mpl::void_>::Run<
    fl::hpcc::WorkSpace>(
  fl::hpcc::WorkSpace *data,
  const std::vector<std::string> &args);

template class  fl::ws::Task<
    fl::hpcc::WorkSpace,
    &fl::ml::KMeans<boost::mpl::void_>::Main<
      fl::hpcc::WorkSpace,
      fl::table::Branch
    >
>;
