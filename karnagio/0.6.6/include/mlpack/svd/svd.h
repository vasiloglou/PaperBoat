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
/**
* @file svd.h
*
* This file implements the interface of Svd class
* It approximates the original matrix by another matrix
* with smaller dimension to a certain degree specified by the
* user and then make SVD decomposition in the projected supspace.
*
*
* @see svd.cc
*/

#ifndef FL_LITE_MLPACK_SVD_SVD_H
#define FL_LITE_MLPACK_SVD_SVD_H

#include "boost/tuple/tuple.hpp"
#include "boost/program_options.hpp"
#include "boost/mpl/void.hpp"
#include "boost/mpl/if.hpp"
#include "boost/mpl/for_each.hpp"
#include "boost/type_traits.hpp"
#include "fastlib/dense/matrix.h"
#include "fastlib/la/linear_algebra.h"
#include <vector>
#include <queue>

namespace fl {
namespace ml {


template<typename TableType>
class Svd {
  public:
    typedef TableType Table_t;
    template<typename WorkSpaceType, typename TableTypeIn, typename ExportedTableType>
    static void ComputeFull(WorkSpaceType *ws,
                     int32 svd_rank,
                     const std::vector<std::string> &references_filenames,
                     std::string *sv_filename,
                     std::vector<std::string> *lsv_filenames,
                     std::string *right_trans_filename);

    template<typename ExportedTableType>
    static void ComputeLowRankSgd(Table_t &table,
                        double step0,
                        index_t n_epochs,
                        index_t n_iterations,
                        bool randomize,
                        ExportedTableType *left,
                        ExportedTableType *right_trans);
    template<typename ExportedTableType>
    static void ComputeLowRankLbfgs(Table_t &table,
                    ExportedTableType *naive_sv,
                    ExportedTableType *naive_left,
                    ExportedTableType *naive_right_trans);

    template<typename WorkSpaceType, typename ExportedTableType, typename ProjectionTableType>
    static void ComputeRandomizedSvd(WorkSpaceType *ws,
                           int32 svd_rank,
                           const std::vector<std::string> &references_names,
                           const std::vector<std::string> &projected_table_names,
                           int smoothing_p,
                           std::string *sv_filename,
                           std::vector<std::string> *left_filenames,
                           std::vector<std::string> *right_trans_filenames);
 
    template<typename ExportedTableType>
    static void ComputeConceptSvd(Table_t &table,
                           const std::vector<double> &l2norms,
                           int32 n_iterations,
                           double error_change,
                           ExportedTableType *sv,
                           ExportedTableType *left,
                           ExportedTableType *right_trans);

    template<typename ExportedTableType, typename TableVectorType>
    static void ComputeRecError(Table_t &table,
                         TableVectorType &sv,
                         ExportedTableType &left,
                         ExportedTableType &right_trans,
                         double *error);

};



  template<>
  class Svd<boost::mpl::void_> {
    public:
      template<typename WorkSpaceType>
      struct Core {
        public:
          Core(WorkSpaceType *ws, const std::vector<std::string> &args);
          template<typename TableType>
          void operator()(TableType&);
        private:
          WorkSpaceType *ws_;
          std::vector<std::string> args_;
      };
    
      template<typename WorkSpaceType>
      static int Run(WorkSpaceType *ws,
          const std::vector<std::string> &args);
  };


}}


#endif

