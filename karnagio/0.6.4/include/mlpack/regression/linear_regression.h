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

#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_H
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_H
#include <deque>
#include <iostream>
#include "boost/program_options.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/if.hpp"
#include "boost/mpl/has_key.hpp"
#include "boost/mpl/int.hpp"
#include "boost/mpl/insert.hpp"
#include "boost/mpl/assert.hpp"
#include "boost/mpl/vector.hpp"
#include <string>
#include <vector>
#include "fastlib/base/base.h"

namespace fl {
namespace ml {
template <typename TemplateArgs>
class LinearRegression;

template<>
class LinearRegression<boost::mpl::void_> {
  public:
    template<typename TableType1>
    class Core {

      private:

        static void ApplyEqualityConstraints_(
          const TableType1 &equality_constraints_table,
          TableType1 *reference_table);

        template<typename Dataset_t>
        static void FindIndexWithPrefix_(
          const Dataset_t &dataset, const char *prefix,
          std::deque<int> &remove_indices,
          std::vector< std::string > *remove_feature_names,
          std::deque<int> *additional_remove_indices,
          bool keep_going_after_first_match);

        static void SetupIndices_(
          TableType1 &table,
          const std::vector< std::string > &remove_index_prefixes,
          const std::vector< std::string > &prune_predictor_index_prefixes,
          const std::string &prediction_index_prefix,
          std::deque<int> *predictor_indices,
          std::deque<int> *prune_predictor_indices,
          std::vector< std::string > *prune_predictor_feature_names,
          int *prediction_index);

      public:

        template<bool do_naive_least_squares, typename DataAccessType>
        static int Branch(DataAccessType *data,
                          boost::program_options::variables_map &vm);

        template<typename DataAccessType>
        static int Main(DataAccessType *data,
                        boost::program_options::variables_map &vm);
    };

    static bool ConstructBoostVariableMap(
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm);
    /**
     * @brief This is the main driver function that the user has to
     *        call.
     */
    template<typename DataAccessType, typename BranchType>
    static int Main(DataAccessType *data,
                    const std::vector<std::string> &args);


    template<typename DataAccessType>
    static void Run(DataAccessType *data,
        const std::vector<std::string> &args);

};
};
};

#endif
