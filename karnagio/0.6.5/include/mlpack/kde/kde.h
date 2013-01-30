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

#ifndef FL_LITE_MLPACK_KDE_KDE_H
#define FL_LITE_MLPACK_KDE_KDE_H

#include "fastlib/base/base.h"
#include "fastlib/la/linear_algebra.h"
#include "mlpack/kde/kde_stat.h"
#include "fastlib/base/base.h"
#include "fastlib/math/fl_math.h"
#include "fastlib/base/mpl.h"
#include "fastlib/table/table.h"
#include "fastlib/table/default_table.h"
#include "boost/program_options.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/if.hpp"
#include "boost/mpl/has_key.hpp"
#include "boost/mpl/int.hpp"
#include "boost/mpl/insert.hpp"
#include "boost/mpl/assert.hpp"
#include "boost/mpl/vector.hpp"
#include "mlpack/kde/dualtree_trace.h"

/**
 *  @file kde.h
 *   Defines the class Kde that provides all the functionality for 
 *   computing Kernel Density Estimation. It implements the deterministic
 *   version (approximate and exact). It also implements a monte carlo version
 *   and a bandwidth optimizer.
 */

namespace fl {
namespace ml {
/**
 * @brief 
 *  @code
 *   template <typename TemplateArgs> class Kde;   
 *  @endcode
 *  To instantiate a Kde class you need to define a struct with the 
 *  following types:
 *  @code
 *   // An example of defining the  KdeArgs for Kde instantiation
 *   struct KdeArgs {
 *     //  DefaultTable can be found in include/fastlib/table/default_table.h
 *     typedef fl::table::DefaultTable TableType;
 *     //  GaussianStar kernel can be found in include/fastlib/math/fl_math.h 
 *     typedef fl::math::GaussianStarKernel<double, false> KernelType;
 *     typedef ... ComputationType;
 *   };
 *  @endcode
 *
 */
template<typename TemplateArgs>
class Kde {
  public:
    /** The table that contains the data for computing Kde */
    typedef typename TemplateArgs::TableType Table_t;
    /** Type of the kernel to use for Kde */
    typedef typename TemplateArgs::KernelType Kernel_t;
    /** Type of the tree that indexes the data  */
    typedef typename Table_t::Tree_t Tree_t;
    /** Type of point used in the table */
    typedef typename Table_t::Point_t Point_t;
    /** Type of the precision used to do computations*/
    typedef typename Table_t::CalcPrecision_t CalcPrecision_t;
    /** Type that keeps delta changes for approximating a given reference node */
    typedef typename TemplateArgs::ComputationType::Delta_t Delta_t;
    /** Type that keeps global normalization constants  */
    typedef typename TemplateArgs::ComputationType::Global_t Global_t;
    /** Type that keeps the postponed contribution of a node that it must be passed down 
     *  to its descendants */
    typedef typename TemplateArgs::ComputationType::Postponed_t Postponed_t;
    /** Type that holds the results of Kde */
    typedef typename TemplateArgs::ComputationType::Result_t Result_t;
    /** Type that keeps the statistics computed for each node */
    typedef typename TemplateArgs::ComputationType::Statistic_t Statistic_t;
    /** Keeps the lower and upper bound of the results for a particular node */
    typedef typename TemplateArgs::ComputationType::Summary_t Summary_t;

  public:
    /** 
     * @brief sets the bandwidth
     */ 
    void set_bandwidth(double bandwidth_in);
    /**
     * @brief returns a pointer to the query table
     */
    Table_t *query_table();
    /**
     * @brief returns a pointer to the reference table
     */
    Table_t *reference_table();
    /**
     * @brief returns a Global_t structure that has the normalization statistics
     */
    Global_t &global();
    /**
     * @brief When the reference table and the query table are the same then
     *        the Kde is called monochromatic
     */
    bool is_monochromatic() const;
    /**
     * @brief Initialize a Kde engine with 
     *   @param reference_table a pointer to the reference table
     *   @param query_table a pointer to the query table
     *   @param bandwidth_factor_in the bandwidth to use for Kde
     *   @param probability_in, is monte carlo version, probability 
     *    the probability the approximation error is true 
     */
    void Init(Table_t *reference_table,
              Table_t *query_table,
              double bandwidth_factor_in,
              double relative_error_in,
              double probability_in);

  private:
    Table_t *query_table_;
    Table_t *reference_table_;
    Global_t global_;
    bool is_monochromatic_;
};

/**
 * @brief All mlpack classes have an instantiation with boost::mpl::void_
 *  This instantiation has the Main driver that gets vector of strings as arguments
 * @code
 *  template<> class Kde<boost::mpl::void_>;
 * @endcode
 *
 */
template<>
class Kde<boost::mpl::void_> {
  public:
    /** @brief 
     *  @code 
     *  template<typename TableType> class Core
     *  @endcode
     *  This class is being used by our internal system for generating instantiations for
     *  different table types.
     */
    template<typename TableType1>
    class Core {

      public:
        /**
         * @brief This class is being used to store some options
         */
        template<typename DataAccessType>
        class BranchArgs {
          public:
            std::vector<TableType1 *> references_in;

            TableType1 *queries_in;

            std::vector< typename DataAccessType::DefaultTable_t * > densities_out;

            std::string kernel;

            double bandwidth;
	    
	    std::string bandwidth_selection;

            double probability;

            double relative_error;

            std::string metric;

            TableType1 *metric_weights_in;

            double dense_sparse_scale;

            std::string algorithm;

            int leaf_size;

            int iterations;

	    int num_lbfgs_restarts;

	    int num_line_searches;

            typename DataAccessType::template TableVector<int> * result_out;

          public:

            ~BranchArgs();

            BranchArgs();
        };

      private:
        /**
         * @brief 
         * @code
         *  template<typename DataAccessType>
         *  static void ParseArguments_(
         *    DataAccessType &data_,
         *    const boost::program_options::variables_map &vm,
         *    BranchArgs<DataAccessType> *args_out);
         * @endcode
         * @param data is a DataAccessType object, that does all the data interface
         * @param vm is a map that has all the program options parsed
         * @param args_out is a struct that has all the variables we need to run Kde
         */
        template<typename DataAccessType>
        static void ParseArguments_(
          DataAccessType &data,
          const boost::program_options::variables_map &vm,
          BranchArgs<DataAccessType> *args_out);
        /**
         * @brief 
         *  @code
         *   template < typename KernelType,
         *   template<typename> class ComputationType,
         *    typename MetricType, typename DataAccessType >
         *    static int Branch_(
         *      int reference_set_num, BranchArgs<DataAccessType> &arguments,
         *      int level);
         *  @endcode
         *  This function calls the appropriate instantiation of Kde 
         */
        template < typename KernelType,
        template<typename> class ComputationType,
        typename MetricType, typename DataAccessType >
        static int Branch_(
          int reference_set_num, BranchArgs<DataAccessType> &arguments,
          int level);

      public:
        /** @brief
         *  This is a class that defines typedefs that will be
         *  passed to Kde for proper instantiation
         *
         */
        template < typename KernelType1, typename MetricType1,
        template<typename> class ComputationType1 >
        struct KdeArgs {
          typedef KernelType1 KernelType;
          typedef TableType1 TableType;
          typedef MetricType1 MetricType;
          typedef typename TableType::CalcPrecision_t CalcPrecision_t;
          struct TemplateArgs {
            typedef KernelType1 KernelType;
            typedef TableType1 TableType;
            typedef typename TableType::CalcPrecision_t CalcPrecision_t;
            typedef KdeStatistic<typename TemplateArgs::CalcPrecision_t> Statistic_t;
          };
          typedef ComputationType1<TemplateArgs> ComputationType;
        };
        
        template<typename TemplateArgs>
        struct KdeStructArgs {
          typedef KdeDelta<typename TemplateArgs::CalcPrecision_t> Delta_t;
          typedef KdeGlobal<TemplateArgs> Global_t;
          typedef KdePostponed<typename TemplateArgs::CalcPrecision_t> Postponed_t;
          typedef KdeResult< std::vector<double> > Result_t;
          typedef typename TemplateArgs::Statistic_t Statistic_t;
          typedef KdeSummary<typename TemplateArgs::CalcPrecision_t> Summary_t;
        };
        /**
         * @brief This function is used by our internal system for instantiation
         *        of the main driver for several table options
         * @code
         *  template<typename DataAccessType>
         *  static int Main(DataAccessType *data,
         *  boost::program_options::variables_map &vm);
         * @endcode
         * @param data is a pointer to a DataAccessType object that provides interface between tables 
         *        and sources of data (ie filesystem, database etc)
         * @param vm is boost program options map
         */
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
    template<typename DataAccessType, typename Branch>
    static int Main(DataAccessType *data,
                    const std::vector<std::string> &args);

    template<typename DataAccessType>
    static void Run(DataAccessType *data,
        const std::vector<std::string> &args);

};



}
} // namespaces

#endif
