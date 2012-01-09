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

#ifndef FL_LITE_FASTLIB_TABLE_BRANCH_ON_TABLE_DEV_H_
#define FL_LITE_FASTLIB_TABLE_BRANCH_ON_TABLE_DEV_H_
//#include <omp.h>
#include "fastlib/table/branch_on_table.h"
#include "boost/program_options.hpp"
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
#include "fastlib/data/linear_algebra.h"

namespace fl {
namespace table {
  template<typename AlgorithmType, typename DataAccessType>
  int Branch::BranchOnTable(DataAccessType *data,
                           boost::program_options::variables_map &vm) {
    if (vm.count("references_in")==0) {
      fl::logger->Die() <<"No --references_in given, cannot run algorithm";
    }
    try {
      data->template TryToAttach<fl::table::dense::labeled::kdtree::Table>(
          vm["references_in"].as<std::string>());
        return AlgorithmType::template Core <
             fl::table::dense::labeled::kdtree::Table >::Main(data, vm);
    }
    catch(const fl::TypeException &e) {
      
    }
    try {
      data->template TryToAttach<fl::table::dense::labeled::balltree::Table>(
          vm["references_in"].as<std::string>());
        return AlgorithmType::template Core <
             fl::table::dense::labeled::kdtree::Table >::Main(data, vm);
    }
    catch(const fl::TypeException &e) {
      
    }
    try {
      data->template TryToAttach<fl::table::sparse::labeled::balltree::Table>(
          vm["references_in"].as<std::string>());
      return AlgorithmType::template Core <
               fl::table::sparse::labeled::balltree::Table >::Main(data, vm);
    }
    catch(const fl::TypeException &e) {

    }
    try {
      data->template TryToAttach<fl::table::dense_sparse::labeled::balltree::Table>(
          vm["references_in"].as<std::string>());
      return AlgorithmType::template Core <
             fl::table::dense_sparse::labeled::balltree::Table >::Main(data, vm);
     
    }
    catch(const fl::TypeException &e) {
  
    }
    try{
      data->template TryToAttach<fl::table::categorical::labeled::balltree::Table>(
          vm["references_in"].as<std::string>());
      return AlgorithmType::template Core <
             fl::table::categorical::labeled::balltree::Table >::Main(data, vm);
    }
    catch(const fl::TypeException &e) {
    }
    try {
      //fl::logger->Die()<<"You don't have a valid license for dense_categorical data";
      data->template TryToAttach<fl::table::dense_categorical::labeled::balltree::Table>(
          vm["references_in"].as<std::string>());
      return AlgorithmType::template Core <
          fl::table::dense_categorical::labeled::balltree::Table >::Main(data, vm);
    }
    catch(const fl::TypeException &e) {
    }  
    try {
      data->template TryToAttach<fl::table::sparse::labeled::balltree::uint8::Table>(
          vm["references_in"].as<std::string>());
      return AlgorithmType::template Core <
        fl::table::sparse::labeled::balltree::uint8::Table >::Main(data, vm);   
    }
    catch(const fl::TypeException &e) {
    }
    try {
      data->template TryToAttach<fl::table::sparse::labeled::balltree::uint16::Table>(
          vm["references_in"].as<std::string>());
      return AlgorithmType::template Core <
        fl::table::sparse::labeled::balltree::uint16::Table >::Main(data, vm);   
    }
    catch(const fl::TypeException &e) {
    }
    fl::logger->Die() <<"Your input data are of an unrecognizable type";
    return -1;
  }

}} // namespaces

#endif

