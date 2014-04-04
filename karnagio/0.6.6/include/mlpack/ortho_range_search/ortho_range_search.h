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
/** @file ortho_range_search.h
 *
 *  This file contains an implementation of a tree-based algorithm for
 *  orthogonal range search.
 *
 *  @author Dongryeol Lee (dongryel)
 */
#ifndef MLPACK_ORTHO_RANGE_SEARCH_ORTHO_RANGE_SEARCH_H
#define MLPACK_ORTHO_RANGE_SEARCH_ORTHO_RANGE_SEARCH_H

#include "fastlib/table/table.h"
#include "fastlib/tree/spacetree.h"
#include "fastlib/dense/matrix.h"
#include "boost/utility.hpp"
#include "boost/program_options.hpp"
#include "fastlib/tree/hyper_rectangle_bound.h"
#include "fastlib/tree/hyper_rectangle_tree.h"

namespace fl {
namespace ml {

struct DefaultOrthoArgs {
  typedef boost::mpl::void_ WindowTableType;
  typedef boost::mpl::void_ ReferenceTableType;
};

template <typename TemplateArgs>
class OrthoRangeSearch: boost::noncopyable {
  private:
    typedef typename TemplateArgs::WindowTableType WindowTable_t;
    typedef typename WindowTable_t::Tree_t WindowTree_t;
    typedef typename TemplateArgs::ReferenceTableType ReferenceTable_t;
    typedef typename ReferenceTable_t::Tree_t ReferenceTree_t;

    /** @brief Flag determining a prune */
    enum PruneStatus {SUBSUME, INCONCLUSIVE, EXCLUDE};
    template<typename OutputTableType>
    static void OrthoSlowRangeSearch_(WindowTree_t *search_window_node,
                                      WindowTable_t &window_table,
                                      ReferenceTree_t *reference_node,
                                      ReferenceTable_t &reference_table,
                                      const index_t &start_dim, const index_t &end_dim,
                                      OutputTableType &candidate_points);

    template<typename OutputTableType>
    static void OrthoRangeSearch_(WindowTree_t *search_window_node,
                                  WindowTable_t &window_table,
                                  ReferenceTree_t *reference_node,
                                  ReferenceTable_t &reference_table,
                                  index_t start_dim, index_t end_dim,
                                  OutputTableType &candidate_points);

  public:

    ////////// User-level Functions //////////

    /** @brief Performs the multiple orthogonal range searches
     *         simultaneously.
     */
    template<typename OutputTableType>
    static void Compute(WindowTable_t &window_queries,
                        const index_t &window_leaf_size,
                        ReferenceTable_t &reference_points,
                        const index_t &reference_leaf_size,
                        OutputTableType *candidate_points);


};

template <>
class OrthoRangeSearch<boost::mpl::void_> : boost::noncopyable {
  public:
    template<typename TableType>
    class Core {
      public:
        struct TableMap {
          struct TableArgs {
            typedef typename TableType::Dataset_t DatasetType;
            typedef boost::mpl::bool_<false> SortPoints ;
          };

          typedef typename TableType::CalcPrecision_t CalcPrecision_t;
          struct TreeArgs : public fl::tree::TreeArgs {
            typedef fl::tree::HyperRectangleTree TreeSpecType;
            typedef boost::mpl::bool_<false> StoreLevel;
            typedef boost::mpl::bool_<false> SortPoints ;
            typedef fl::tree::HyperRectangleBound<CalcPrecision_t> BoundType;
          };
        };
        typedef fl::table::Table<TableMap>  QueryTable_t;
        typedef TableType ReferenceTable_t;
        struct CoreOrthoArgs {
          typedef QueryTable_t  WindowTableType;
          typedef ReferenceTable_t ReferenceTableType;
        };
        template<typename DataAccessType>
        static int Main(DataAccessType *data, boost::program_options::variables_map &vm);
    };

    template<typename DataAccessType, typename BranchType>
    static int Main(DataAccessType *data,
                    const std::vector<std::string> &args);

    template<typename DataAccessType>
    static void Run(DataAccessType *data,
        const std::vector<std::string> &args);


};

}
} // namespaces
#endif

