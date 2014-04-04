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
    template<typename TableTypeIn, typename ExportedTableType>
    void ComputeFull(TableTypeIn &table,
                     ExportedTableType *naive_sv,
                     ExportedTableType *naive_left,
                     ExportedTableType *naive_right_trans);

    template<typename ExportedTableType>
    void ComputeLowRankSgd(Table_t &table,
                        double step0,
                        index_t n_epochs,
                        index_t n_iterations,
                        bool randomize,
                        ExportedTableType *left,
                        ExportedTableType *right_trans);
    template<typename ExportedTableType>
    void ComputeLowRankLbfgs(Table_t &table,
                    ExportedTableType *naive_sv,
                    ExportedTableType *naive_left,
                    ExportedTableType *naive_right_trans);

    template<typename ExportedTableType, typename ProjectionTableType>
    void ComputeRandomizedSvd(Table_t &table,
                              int smoothing_p,
                              ProjectionTableType &projector,
                              ExportedTableType *sv,
                              ExportedTableType *left,
                              ExportedTableType *right_trans);
 
    template<typename ExportedTableType>
    void ComputeConceptSvd(Table_t &table,
                           const std::vector<double> &l2norms,
                           int32 n_iterations,
                           double error_change,
                           ExportedTableType *sv,
                           ExportedTableType *left,
                           ExportedTableType *right_trans);

    template<typename ExportedTableType, typename TableVectorType>
    void ComputeRecError(Table_t &table,
                         TableVectorType &sv,
                         ExportedTableType &left,
                         ExportedTableType &right_trans,
                         double *error);

};

template<>
class Svd<boost::mpl::void_> {
  public:
    template<typename TableType1>
    struct Core {
      typedef typename TableType1::CalcPrecision_t CalcPrecision_t;

      template<typename DataAccessType>
      static int Main(DataAccessType *data,
                      boost::program_options::variables_map &vm);
    };

    template<typename DataAccessType, typename BranchType>
    static int Main(DataAccessType *data, const std::vector<std::string> &args);

    template<typename DataAccessType>
    static void Run(DataAccessType *data,
        const std::vector<std::string> &args);

};

}}


#endif

