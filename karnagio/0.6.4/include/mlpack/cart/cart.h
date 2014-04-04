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

#ifndef FL_LITE_MLPACK_CART_CART_H
#define FL_LITE_MLPACK_CART_CART_H

#include "fastlib/base/base.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/base/base.h"
#include "fastlib/math/fl_math.h"
#include "fastlib/base/mpl.h"
#include "fastlib/table/table.h"
#include "fastlib/table/default_table.h"
#include "fastlib/table/integer_table.h"
#include "fastlib/tree/classification_decision_tree.h"
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

namespace fl {
namespace ml {

template<typename TableType>
class Cart {

  public:

    typedef typename TableType::Point_t Point_t;

    typedef typename TableType::Tree_t TreeType;

    typedef typename boost::mpl::at_c < typename Point_t::MetaData_t::TypeList_t, 0 >::type ClassLabelType;

  private:

    TableType *table_;

  private:

    void Prune_(double cost_complexity);

    int NumberOfLeaves_(TreeType *node);

    void CostComplexities_(TreeType *node);

    void TrainingErrors_(TreeType *node,
                         std::vector<TreeType *> *internal_nodes);

    void StringInternalNode_(
      TreeType *child_node,
      bool is_left_child_node,
      int level,
      std::string *text,
      int split_dimension,
      double numeric_split_value,
      const std::string &nominal_split_value,
      bool using_numeric_split);

    template<typename PointType, typename TableVectorType>
    void Classify_(const PointType &point,
                   TreeType *node,
                   TableVectorType &label_out,
                   int query_point_index);

  public:

    void CrossValidate(int num_folds);

    void Init(TableType *table_in, int leaf_size_in,
              const std::string &impurity_in);

    template<typename PointType, typename TableVectorType> 
    void Classify(const PointType &point,
                  TableVectorType &label_out,
                  int query_point_index);

    template<typename QueryTableType, typename TableVectorType> 
    void Classify(const QueryTableType &query_table,
                  TableVectorType *labels_out);

    void String(TreeType *node, int level, std::string *text);
};

template<>
class Cart<boost::mpl::void_> {
  public:
    template<typename TableType1>
    class Core {
      public:

        struct CartTableMap {
          struct TreeArgs : public fl::tree::TreeArgs {
            typedef fl::tree::ClassificationDecisionTree TreeSpecType;
            typedef fl::tree::CartBound<double, double, 2, signed char> BoundType;
            typedef typename TableType1::TableArgs_t::SortPoints SortPoints;
          };

          struct TableArgs {
            typedef typename TableType1::Dataset_t DatasetType;
            typedef typename TableType1::TableArgs_t::SortPoints SortPoints;
          };
        };

        typedef fl::table::Table<CartTableMap> TableType;

      public:
        template<typename DataAccessType>
        static int Main(DataAccessType *data,
                        boost::program_options::variables_map &vm);
    };

    // static bool ConstructBoostVariableMap(
    //   const std::vector<std::string> &args,
    //   boost::program_options::variables_map *vm);

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
