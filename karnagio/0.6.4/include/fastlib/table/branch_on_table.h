#ifndef FL_LITE_FASTLIB_TABLE_BRANCH_ON_TABLE_H_
#define FL_LITE_FASTLIB_TABLE_BRANCH_ON_TABLE_H_
//#include <omp.h>
#include "boost/program_options.hpp"
#include "boost/program_options/parsers.hpp"

namespace fl { namespace table {
  class Branch {
    public:
      template<typename AlgorithmType, typename DataAccessType>
      static int BranchOnTable(DataAccessType *data,
          boost::program_options::variables_map &vm);
  };
}}

#endif
