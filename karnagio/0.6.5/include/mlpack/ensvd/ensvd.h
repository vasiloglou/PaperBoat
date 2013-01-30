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

/**
* @file ensvd.h
*
* This file implements the interface for ensemble EnSvd class
* It is based on the current Svd implementation
*
* @see ensvd.cc
*/

#ifndef FL_LITE_MLPACK_ENSVD_ENSVD_H
#define FL_LITE_MLPACK_ENSVD_ENSVD_H

#include "boost/program_options.hpp"
#include "boost/mpl/void.hpp"
#include "fastlib/dense/matrix.h"
#include "fastlib/la/linear_algebra.h"
#include <vector>
#include <queue>

namespace fl {
namespace ml {


template<typename TableType>
class EnSvd {
  public:
    typedef TableType Table_t;
    
};

template<>
class EnSvd<boost::mpl::void_> {
  public:

    template<typename TableType1>
    struct Core {
      typedef typename TableType1::CalcPrecision_t CalcPrecision_t;

      template<typename WorkSpaceType>
      static int Main(WorkSpaceType *data,
                      boost::program_options::variables_map &vm);
    };

    template<class WorkSpaceType>
    class Project {
      public:
        Project(WorkSpaceType *ws,
                const boost::program_options::variables_map &vm);
        template<typename TableType>
        void operator()(TableType&); 

        template<typename TableType>
        static void MulTask(
            WorkSpaceType *ws,
            const std::string lsv_name,
            boost::shared_ptr<TableType> references_table,
            boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> rsv_table,
            boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> lsv_table);

      private:
        WorkSpaceType *ws_;
        const boost::program_options::variables_map vm_;
    };

    template<typename WorkSpaceType, typename BranchType>
    static int Main(WorkSpaceType *data, const std::vector<std::string> &args);

    template<typename WorkSpaceType>
    static void Run(WorkSpaceType *ws,
        const std::vector<std::string> &args);

};

}}


#endif

