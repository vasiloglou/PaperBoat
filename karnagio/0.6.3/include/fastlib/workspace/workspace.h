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

#ifndef INCLUDE_FASTLIB_WORKSPACE_WORKSPACE_H_
#define INCLUDE_FASTLIB_WORKSPACE_WORKSPACE_H_
#include <map>
#include "boost/any.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/mpl/for_each.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/utility.hpp"
#include "boost/threadpool.hpp"
#include "fastlib/table/branch_on_table.h"
#include "fastlib/table/table_defs.h"
#include "fastlib/table/default_table.h"
#include "fastlib/table/matrix_table.h"
#include "fastlib/table/matrix_table.h"
#include "fastlib/table/uinteger_table.h"
#include "fastlib/table/integer_table.h"
#include "fastlib/table/table_vector.h"
#include "fastlib/table/default_sparse_int_table.h"
#include "fastlib/table/default_sparse_double_table.h"
#include "fastlib/metric_kernel/weighted_lmetric.h"
#include "fastlib/table/default/categorical/labeled/balltree/table.h"
#include "fastlib/table/default/dense/labeled/balltree/table.h"
#include "fastlib/table/default/dense/labeled/kdtree/table.h"
#include "fastlib/table/default/dense_categorical/labeled/balltree/table.h"
#include "fastlib/table/default/dense_sparse/labeled/balltree/table.h"
#include "fastlib/table/default/sparse/labeled/balltree/table.h"
#include "fastlib/table/default/sparse/labeled/balltree/uint8/table.h"
#include "fastlib/table/default/sparse/labeled/balltree/uint16/table.h"
#include "fastlib/table/default/sparse/labeled/balltree/float32/table.h"


namespace fl { namespace ws {
class WorkSpace : boost::noncopyable {
  public:
    typedef fl::table::Branch Branch_t;
    typedef fl::table::dense::labeled::kdtree::Table DefaultTable_t;
    typedef fl::table::MatrixTable MatrixTable_t;
    typedef fl::table::DefaultSparseIntTable DefaultSparseIntTable_t;
    typedef fl::table::sparse::labeled::balltree::Table DefaultSparseDoubleTable_t;
    typedef fl::table::UIntegerTable UIntegerTable_t;
    typedef fl::table::IntegerTable IntegerTable_t;
    template<typename PrecisionType>
    class TableVector : public fl::table::TableVector<PrecisionType> {
    };
    
    typedef boost::mpl::vector10<
      DefaultTable_t,
      MatrixTable_t,
      DefaultSparseIntTable_t,
      DefaultSparseDoubleTable_t,
      UIntegerTable_t,
      IntegerTable_t,
      TableVector<index_t>,
      TableVector<int>,
      TableVector<double>,
      TableVector<signed char>
    > ParameterTables_t;

    typedef boost::mpl::vector8<
      fl::table::dense::labeled::kdtree::Table,
      fl::table::dense::labeled::balltree::Table,
      fl::table::sparse::labeled::balltree::Table,
      fl::table::dense_sparse::labeled::balltree::Table,
      fl::table::categorical::labeled::balltree::Table,
      fl::table::dense_categorical::labeled::balltree::Table,
      fl::table::sparse::labeled::balltree::uint8::Table,
      fl::table::sparse::labeled::balltree::uint16::Table
    > DataTables_t;

    WorkSpace();
    ~WorkSpace();
    // We need these two utility functions so that we can still use WorkSpace
    // for executables 
    void LoadAllTables(const std::vector<std::string> &args);
    void IndexAllReferencesQueries(const std::vector<std::string> &args);
    void ExportAllTables(const std::vector<std::string> &args);
    /*
     * @brief this is an auxiliary function that loads a table 
     *        from a file to the workspace. This function 
     *        needs the TableType to be specified
     **/
    template<typename TableType>
    void LoadTable(const std::string &name, boost::shared_ptr<TableType> table);
    /**
     * @brief this is an auxiliary function that loads a table
     *        from a file to the workspace. This function 
     *        needs an mpl::vector of table types that it will
     *        help the function to find the type of the loading
     *        matrix
     */
    template<typename TableSetType>
    void LoadFromFile(const std::string &name,
      const std::string &filename);
    /**
     * @brief loads a data table to the workspace. You should 
     *        use this one for data tables
     */
    void LoadDataTableFromFile(const std::string &name,
      const std::string &filename);
    /**
     * @brief loads a data table to the workspace. You should use
     *         this one for parameter tables
     */
    void LoadParameterTableFromFile(const std::string &name,
      const std::string &filename);
    /**
     * @brief this function indexes a table in place. It mutates the table.
     *        It is a blocking function, but also dangerous. You have to 
     *        make sure that nobody can access the data while being indexed.
     *        The advantage of this method is that avoids copying the data
     */
    void IndexTable(const std::string &variable, 
        const std::string &metric,
        const std::string &metric_args,
        const int leaf_size);
    /**
     * @brief this function indexes a table variable. The indexed table
     *        is stored in the new variable_indexed. The advantage of this
     *        method is that it is safe. 
     *
     */
    void IndexTable(const std::string &variable, 
        const std::string &variable_indexed,
        const std::string &metric,
        const std::string &metric_args,
        const int leaf_size);

    void ExportToFile(const std::string &name, const std::string &filename);
 
    /**
     * @brief This function returns immediately. All it does is to check
     *        if the table exists and if it does it checks if it is ready
     *        to be attached. In other words if you call IsTableAvailable
     *        and it returns true then calling Attach will not block 
     *        because it means the table is ready
     */
    bool IsTableAvailable(const std::string &name);

    template<typename TableType>
    void Attach(const std::string &name, 
        boost::shared_ptr<TableType> *table);
    
    template<typename TableType>    
    void TryToAttach(const std::string &name);

    template<typename TableType>
    void Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<TableType> *table);
    
    void Detach(const std::string  &table_name);

    void Purge(const std::string &table_name);

    template<int Index, typename TableType1, typename TableType2, typename TableType3>
    void TieLabels(boost::shared_ptr<TableType1> table,
        boost::shared_ptr<TableType2> labels, 
        const std::string &new_name,
        boost::shared_ptr<TableType3> *new_table);
     
    template<typename TableType1, typename TableType2>
    void CopyAndDestruct(boost::shared_ptr<TableType1> table1,
        boost::shared_ptr<TableType2> *table2);

    void schedule(boost::threadpool::task_func const &task);
    /**
     * @brief schedule mode = 0 uses the threadpool
     *                      = 1 uses a vector of threads
     */
    void set_schedule_mode(int schedule_mode);
    void set_pool(int n_threads);
    void CancelAllTasks();
    void WaitAllTasks();
    void MakeACopy(WorkSpace *ws); 
    const std::string GiveTempVarName();

  protected:
    struct LoadMeta {
      public:
        LoadMeta(const std::string &name,
            const std::string &filename, 
            boost::mutex *mutex,
            boost::mutex &global_mutex,
            std::map<std::string, boost::any> *var_map,
            bool *success) : name_(name),
          filename_(filename), mutex_(mutex), global_mutex_(global_mutex),
          var_map_(var_map), success_(success) {
        }
        template<typename TableType>
        void operator()(TableType&);
      private:
        const std::string &name_;
        const std::string &filename_;
        boost::mutex *mutex_;
        boost::mutex &global_mutex_;
        std::map<std::string, boost::any> *var_map_; 
        bool *success_;
    };
    
    struct SaveMeta {
      public:
        SaveMeta(std::map<std::string, boost::any> *var_map, 
            const std::string &name, 
            const std::string &filename, 
            bool *success) : var_map_(var_map), name_(name), 
                             filename_(filename), success_(success)
        {
        } 
        template<typename TableType>
        void operator()(TableType&);

      private:
        std::map<std::string, boost::any> *var_map_;
        const std::string &name_;
        const std::string &filename_;
        bool *success_;  
    };

    struct IndexMeta {
      public:
        IndexMeta(WorkSpace *ws,
            std::map<std::string, boost::any> *var_map, 
            const std::string &variable,
            const std::string &metric,
            const std::string &metric_args,
            const int &leaf_size,
            bool *success) : ws_(ws),
          var_map_(var_map), variable_(variable), metric_(metric), 
          metric_args_(metric_args), leaf_size_(leaf_size), success_(success) {
        }
        template<typename TableType>
        void operator()(TableType&);

      private:
        WorkSpace *ws_;
        std::map<std::string, boost::any> *var_map_;
        const std::string &variable_;
        const std::string &metric_;
        const std::string &metric_args_;
        const int &leaf_size_;
        bool *success_;
    };

    std::string temp_var_prefix_;
    index_t temp_var_counter_;
    boost::mutex temp_var_mutex_;
    std::string workspace_name_;
    std::map<std::string, boost::any> var_map_;
    std::map<std::string, boost::shared_ptr<boost::mutex> > mutex_map_;
    boost::mutex global_mutex_;
    boost::scoped_ptr<boost::threadpool::pool> pool_; 
    std::vector<boost::shared_ptr<boost::thread> > vector_pool_;
    /**
     * @brief this variable selects which scheduler to use
     *        if it is set to 0 it uses the boost threadpool
     *        otherwise if it is 1 it uses a vector of 
     */
    int schedule_mode_;
    boost::mutex schedule_mutex_;
    void ExportAllTablesTask(const std::vector<std::string> &args);
    
    void DummyThreadCancel(boost::shared_ptr<boost::thread> thread);
    void DummyThreadLaunch(boost::threadpool::task_func const & task);

};
}}
#endif

