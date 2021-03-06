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

#ifndef FL_LITE_MLPACK_ENSVD_ENSVD_DEFS_H
#define FL_LITE_MLPACK_ENSVD_ENSVD_DEFS_H

#include <string>
#include "fastlib/util/string_utils.h"
#include "boost/program_options.hpp"
#include "fastlib/dense/matrix.h"
#include "fastlib/data/multi_dataset.h"
#include "mlpack/svd/svd.h"
#include "ensvd.h"
#include "fastlib/workspace/task.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/table/linear_algebra.h"
#include "fastlib/workspace/based_on_table_run.h"

template<typename WorkSpaceType>
inline void GetSequenceFileNames(
    WorkSpaceType *ws,
    const boost::program_options::variables_map &vm,
    const std::string &suffix,
    const std::string &arg_name, 
    std::vector<std::string> *table_names);

template<class WorkSpaceType, typename BranchType>
int fl::ml::EnSvd<boost::mpl::void_>::Main(WorkSpaceType *ws, const std::vector<std::string> &args) {

  FL_SCOPED_LOG(EnSvd);
  ////////// READING PARAMETERS AND LOADING DATA /////////////////////
  boost::program_options::options_description desc("Available options");
  desc.add_options()
  ("help", "Display help on Ensemble SVD")
  ("references_in", 
   boost::program_options::value<std::string>(),
   "a comma separated list of files that contain the data to perform svd"
  )
  ("references_prefix_in",
   boost::program_options::value<std::string>(),
   "prefix for the reference files to be imported"
  )
  ("references_num_in",
   boost::program_options::value<int32>(),
   "number for reference files when used references_prefix. We assume "
   "that when references_prefix is dummy, then we assume that the input "
   "files are dummy1,dummy2,...,dummy11,dummy12..."
  )
  ("lsv_out", 
   boost::program_options::value<std::string>(),
   "a comma separated list of files to output the left side of SVD. "
   "The number of lsv_out files must match the number of references in files "
  ) 
  ("rsv_out",
   boost::program_options::value<std::string>(),
    "a comma separated list of files to output the right transposed side of SVD. "
    "The rows of the transposed side are distributed evenly in all files"
  )
  ("sv_out",
    boost::program_options::value<std::string>(),
    "a file containing the singular values"
  )
  ("lsv_prefix_out",
   boost::program_options::value<std::string>(),
   "the prefix for the left hand side, when we output it in a sequence of files"  
   " for example dummy1, dummy2, dummy3..."
  )
  ("lsv_num_out",
   boost::program_options::value<int32>(),
   "number of left hand side files to output, when using --lsv_prefix_out "
   "flag"
  ); 


  boost::program_options::variables_map vm;
  std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args, "");
  boost::program_options::command_line_parser clp(args1);
  clp.style(boost::program_options::command_line_style::default_style
            ^ boost::program_options::command_line_style::allow_guessing);
  try {
    boost::program_options::store(clp.options(desc).run(), vm);
  }
  catch(const boost::program_options::invalid_option_value &e) {
	  fl::logger->Die() << "Invalid Argument: " << e.what();
  }
  catch(const boost::program_options::invalid_command_line_syntax &e) {
	  fl::logger->Die() << "Invalid command line syntax: " << e.what(); 
  }
  catch (const boost::program_options::unknown_option &e) {
    fl::logger->Die() << e.what()
      <<" . This option will be ignored";
  }
  catch ( const boost::program_options::error &e) {
    fl::logger->Die() << e.what();
  } 
  boost::program_options::notify(vm);
   if (vm.count("help")) {
    fl::logger->Message() << fl::DISCLAIMER << "\n";
    fl::logger->Message() << desc << "\n";
    return 1;
  }
  fl::ws::RequiredOrArgs(vm,
    "references_in,references_prefix_in");


  // load all data
  std::vector<std::string> references_filenames;
  GetSequenceFileNames(ws, 
      vm,
    "_in",
    "references", 
    &references_filenames);
  std::string lsv_prefix="lsv"+ws->GiveTempVarName();
  std::string rsv_prefix="rsv"+ws->GiveTempVarName();
  // start k svds 
  for(int32 i=0; i<references_filenames.size(); ++i) {
    fl::ws::Arguments svd_args;
    svd_args.Add(fl::ws::MakeArgsFromPrefix(args, "svd"));
    svd_args.Add("references_in", 
        references_filenames[i]);
    svd_args.Add("lsv_out", ws->GiveFilenameFromSequence(lsv_prefix, i));
    svd_args.Add("rsv_out", ws->GiveFilenameFromSequence(rsv_prefix, i));
    fl::logger->Message()<<"Running SVD on "
      <<ws->GiveFilenameFromSequence("references", i)<< std::endl;
    fl::ml::Svd<boost::mpl::void_>::Run(ws, svd_args.args());
  }
  // merge all right hand side 
  boost::shared_ptr<typename WorkSpaceType::MatrixTable_t>
    concat_rsv_table;
  index_t n_attributes;
  index_t n_entries;
  ws->GetTableInfo(ws->GiveFilenameFromSequence(rsv_prefix, 0), &n_entries, &n_attributes, NULL, NULL);
  std::string concat_rsv=ws->GiveTempVarName();
  ws->Attach(concat_rsv,
      std::vector<index_t>(1, n_entries),
      std::vector<index_t>(), 
      references_filenames.size()*n_attributes,
      &concat_rsv_table);
  int32 results_merged=0;
  std::list<std::string> result_tables;
  for(int32 i=0; i<references_filenames.size(); ++i) {
    result_tables.push_back(ws->GiveFilenameFromSequence(rsv_prefix, i));
  }
  typename WorkSpaceType::MatrixTable_t::Point_t point;
  index_t counter=0;
  while(results_merged<references_filenames.size()) {
    for(std::list<std::string>::iterator it=result_tables.begin();
        it!=result_tables.end(); ++it) {
      if (ws->IsTableAvailable(*it)) {
        boost::shared_ptr<
          typename WorkSpaceType::MatrixTable_t
        > rsv_table;
        ws->Attach(*it, &rsv_table);
        for(index_t j=0; j<rsv_table->n_attributes(); ++j) { 
          for(index_t i=0; i<rsv_table->n_entries(); ++i) {
            concat_rsv_table->set(counter, 
                i,
                rsv_table->get(i, j));
          }
          counter++;
        }
        it=result_tables.erase(it); 
        results_merged++;
      }    
    }
  }
  ws->Purge(concat_rsv);
  ws->Detach(concat_rsv);
  // orthonormalize 
  fl::logger->Message()<<"Orthonormalizing/Aggregating"<<std::endl;
  fl::ws::Arguments svd_args;
  // svd_args.Add(fl::ws::MakeArgsFromPrefix(args, "svd"));
  svd_args.Add("algorithm", "covariance");
  svd_args.Add("references_in", concat_rsv);
  boost::program_options::variables_map vm1(vm);
  if (vm.count("rsv_out")) {
    svd_args.Add("rsv_out", vm["rsv_out"].as<std::string>());
    vm1.insert(std::make_pair("rsv_in", 
        boost::program_options::variable_value(vm["rsv_out"].as<std::string>(), false)));
  } else {
    std::string temp=ws->GiveTempVarName();
    svd_args.Add("rsv_out", temp); 
    vm1.insert(std::make_pair("rsv_in", 
          boost::program_options::variable_value(temp, false)));
  }
  if (vm.count("sv_out")) {
    svd_args.Add("sv_out", vm["sv_out"].as<std::string>());
  }
  std::map<std::string, std::string> svd_map=fl::ws::GetArgumentPairs(
      fl::ws::MakeArgsFromPrefix(args, "svd"));
  if (svd_map.count("--svd_rank")!=0) {
    svd_args.Add("svd_rank", svd_map["--svd_rank"]);
  }
  fl::ml::Svd<boost::mpl::void_>::Run(ws, svd_args.args());
  // project 
  boost::program_options::notify(vm1);
  Project<WorkSpaceType> project(ws, vm1);
  fl::ws::BasedOnTableRun(ws, 
      references_filenames[0],
      project);
  return 0;
}

template<typename WorkSpaceType>
void fl::ml::EnSvd<boost::mpl::void_>::Run(
      WorkSpaceType *ws,
      const std::vector<std::string> &args) {
  fl::ws::Task<
    WorkSpaceType,
    &Main<
      WorkSpaceType, 
      typename WorkSpaceType::Branch_t
    > 
  > task(ws, args);
  ws->schedule(task); 
}

template<typename WorkSpaceType>
inline void GetSequenceFileNames(
    WorkSpaceType *ws,
    const boost::program_options::variables_map &vm,
    const std::string &suffix,
    const std::string &arg_name, 
    std::vector<std::string> *table_names) {
  if (suffix!="_in" && suffix!="_out") {
    fl::logger->Die()<<"suffix can only be _in or _out";
  }
  if (vm.count(arg_name+suffix)!=0) {
    std::string filenames=vm[arg_name+suffix].as<std::string>();
    *table_names=fl::SplitString(filenames, ",");
  } else {
    if (vm.count(arg_name+"_prefix"+suffix)!=0) {
      std::string prefix=vm[arg_name+"_prefix"+suffix].as<std::string>();
      if (vm.count(arg_name+"_num"+suffix)==0) {
        fl::logger->Die()<<"--"<<arg_name+"_prefix"+suffix+" "<<"was set"
          <<" ("<<prefix <<")"
          <<" but --"<<arg_name+"_num"+suffix<< " was not set.";
      }
      int32 n_files=vm[arg_name+"_num"+suffix].as<int32>();
      for(int32 i=0; i<n_files; ++i) {
        table_names->push_back(ws->GiveFilenameFromSequence(prefix, i));
      }
    } 
  }
}



template<typename WorkSpaceType>
template<typename TableType>
void fl::ml::EnSvd<boost::mpl::void_>::Project<WorkSpaceType>::operator()(TableType&) {
  boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> rsv_table;
  ws_->Attach(vm_["rsv_in"].as<std::string>(), &rsv_table);
  std::vector<std::string> lsv_out;
  GetSequenceFileNames(ws_, vm_, "_out", "lsv", &lsv_out);
  if (lsv_out.size()==0) {
    return;
  }
  int32 n_references=lsv_out.size();
  std::vector<std::string> references;
  GetSequenceFileNames(ws_, vm_, "_in", "references", &references);
  if (references.size()!=lsv_out.size()) {
    fl::logger->Die()<<"The number of --lsv_out must be the same "
      "as references_in";
  }
  for(int32 i=0; i<n_references; ++i) {
    boost::shared_ptr<TableType> references_table;
    ws_->Attach(references[i],
               &references_table);
    boost::shared_ptr<typename WorkSpaceType::MatrixTable_t>
      lsv_table; 
    ws_->Attach(lsv_out[i],
               std::vector<index_t>(1,rsv_table->n_attributes()),
               std::vector<index_t>(), 
               references_table->n_entries(),
               &lsv_table);
    ws_->schedule(boost::bind(&EnSvd<boost::mpl::void_>::Project<WorkSpaceType>::MulTask<TableType>,
          ws_, lsv_out[i], references_table, rsv_table, lsv_table));
  }
}

template<typename WorkSpaceType>
fl::ml::EnSvd<boost::mpl::void_>::Project<WorkSpaceType>::Project(
    WorkSpaceType *ws,
    const boost::program_options::variables_map &vm) :
  ws_(ws), vm_(vm) {}

template<typename WorkSpaceType>
template<typename TableType>
void fl::ml::EnSvd<boost::mpl::void_>::Project<WorkSpaceType>::MulTask(
    WorkSpaceType *ws,
    const std::string lsv_name,
    boost::shared_ptr<TableType> references_table,
    boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> rsv_table,
    boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> lsv_table) {

  fl::table::Mul<fl::la::NoTrans, fl::la::NoTrans>(
      *references_table,
      *rsv_table, 
      lsv_table.get());
  ws->Purge(lsv_name);
  ws->Detach(lsv_name);
}

#endif

