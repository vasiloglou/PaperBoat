/*
Copyright © 2010, Ismion Inc.
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

#include "fastlib/workspace/workspace.h"
#include "fastlib/workspace/workspace_defs.h"
#include "fastlib/table/table_defs.h"
#include "fastlib/table/default/categorical/labeled/balltree/table.h"
#include "fastlib/table/default/dense/labeled/balltree/table.h"
#include "fastlib/table/default/dense/labeled/kdtree/table.h"
#include "fastlib/table/default/dense_categorical/labeled/balltree/table.h"
#include "fastlib/table/default/dense_sparse/labeled/balltree/table.h"
#include "fastlib/table/default/sparse/labeled/balltree/table.h"
#include "fastlib/table/default/sparse/labeled/balltree/uint8/table.h"
#include "fastlib/table/default/sparse/labeled/balltree/uint16/table.h"
#include "fastlib/table/default/sparse/labeled/balltree/float32/table.h"
#include "fastlib/table/uinteger_table.h"
#include "fastlib/table/integer_table.h"
#include "fastlib/table/table_vector.h"

namespace fl { namespace ws {

  template void WorkSpace::LoadFromFile<WorkSpace::ParameterTables_t>(const std::string &name,
      const std::string &filename);
  template void WorkSpace::LoadFromFile<WorkSpace::DataTables_t>(const std::string &name,
      const std::string &filename);

  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<fl::table::dense::labeled::kdtree::Table> table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<fl::table::dense::labeled::balltree::Table> table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<fl::table::sparse::labeled::balltree::Table> table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<fl::table::dense_sparse::labeled::balltree::Table> table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<fl::table::categorical::labeled::balltree::Table> table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<fl::table::dense_categorical::labeled::balltree::Table> table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<fl::table::sparse::labeled::balltree::uint8::Table> table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<fl::table::sparse::labeled::balltree::uint16::Table> table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<WorkSpace::DefaultSparseIntTable_t> table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<WorkSpace::IntegerTable_t> table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<WorkSpace::UIntegerTable_t> table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<WorkSpace::TableVector<int> > table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<WorkSpace::TableVector<double> > table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<WorkSpace::TableVector<long> > table);
  template void WorkSpace::LoadTable(const std::string &name, 
        boost::shared_ptr<WorkSpace::TableVector<long long> > table);



  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<fl::table::dense::labeled::kdtree::Table> *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<fl::table::dense::labeled::balltree::Table> *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<fl::table::sparse::labeled::balltree::Table> *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<fl::table::dense_sparse::labeled::balltree::Table> *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<fl::table::categorical::labeled::balltree::Table> *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<fl::table::dense_categorical::labeled::balltree::Table> *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<fl::table::sparse::labeled::balltree::uint8::Table> *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<fl::table::sparse::labeled::balltree::uint16::Table> *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<WorkSpace::DefaultSparseIntTable_t> *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<WorkSpace::IntegerTable_t> *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<WorkSpace::UIntegerTable_t> *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<WorkSpace::TableVector<int> > *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<WorkSpace::TableVector<double> > *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<WorkSpace::TableVector<long> > *table);
  template void WorkSpace::Attach(const std::string &name, 
        boost::shared_ptr<WorkSpace::TableVector<long long> > *table);



  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<fl::table::dense::labeled::kdtree::Table> *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<fl::table::sparse::labeled::balltree::Table> *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<fl::table::dense_sparse::labeled::balltree::Table> *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<fl::table::categorical::labeled::balltree::Table> *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<fl::table::dense_categorical::labeled::balltree::Table> *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<fl::table::sparse::labeled::balltree::uint8::Table> *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<fl::table::sparse::labeled::balltree::uint16::Table> *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<WorkSpace::DefaultSparseIntTable_t> *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<WorkSpace::IntegerTable_t> *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<WorkSpace::UIntegerTable_t> *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<WorkSpace::TableVector<int> > *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<WorkSpace::TableVector<long> > *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<WorkSpace::TableVector<long long> > *table);
  template void WorkSpace::Attach(const std::string &name,
      const std::vector<index_t> dense_sizes,
      const std::vector<index_t> sparse_sizes,
      const index_t num_of_points,
      boost::shared_ptr<WorkSpace::TableVector<double> > *table);

}}
