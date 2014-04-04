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
  ("help", "Display help on Ensemble SVD\n"
   "All SVD related parameters will be passes with the prefix svd:\n"
   "in order to get the svd flags run svd --help")
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
  )(
    "aggregate_phase_algorithm",
    boost::program_options::value<std::string>()->default_value("randomized"),
    "the svd algorthm for aggregating the orthonormal basis from all the chunks"
  )(
    "global_dimensionality",
    boost::program_options::value<index_t>(),
    "references files might have different dimensionalities and the data points are "
    "encoded with point_ids. The references tables must be square. In that case it is a good idea "
    "to provide the global dimensionality (the maximum point_id)"
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
  // check if the references table have point ids
  index_t global_dimensionality=0;
  bool has_point_ids=false;
  if (vm.count("global_dimensionality")>0) {
    global_dimensionality=vm["global_dimensionality"].as<index_t>();
    fl::logger->Message()<<"Tables have point_ids, global dimensionality provided ";
    has_point_ids=true;
  } else {
    fl::logger->Warning()<<"Cannot determine if the dataset has global dimensionality";
//    boost::shared_ptr<TableType> references_table;
//    ws->Attach(references_filenames[0], &references_table);
//    if (references_table->n_entries()>=2) {
//      typename TableType::Point_t p1, p2;
//      references_table->get(0, &p1);
//      references_table->get(1, &p2);
//      if (p1.meta_data().template get<2>()!=p2.meta_data().template.get<2>()) {
//        has_point_ids=true;
//        fl::logger->Message()<<"Detected point ids in the reference file, scanning reference_data to "
//          "scanning all reference data to detect global_dimensionality";
//        for(int32 i=0; i<references_filenames.size(); ++i) {
//          boost::shared_ptr<TableType> references_table;
//          ws->Attach(references_filenames[i], &references_table);
//          typename TableType::Point_t point;
//          for(index_t j=0; j<references_table->n_entries(); ++j) {
//            references_table.get(j, &point);
//            global_dimensionality=std::max(
//                global_dimensionality,
//                point.meta_data().template get<2>()
//                );
//          }
//        }
//      }
//    }
  }

  // merge all right hand side 
  boost::shared_ptr<typename WorkSpaceType::MatrixTable_t>
    concat_rsv_table_dense;

  boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t>
    concat_rsv_table_sparse;

  index_t n_attributes;
  index_t n_entries;
  ws->GetTableInfo(ws->GiveFilenameFromSequence(rsv_prefix, 0), &n_entries, &n_attributes, NULL, NULL);
  std::string concat_rsv=ws->GiveTempVarName();
  if (global_dimensionality==0) {
    global_dimensionality=n_entries;
    ws->Attach(concat_rsv,
      std::vector<index_t>(1, global_dimensionality),
      std::vector<index_t>(), 
      references_filenames.size()*n_attributes,
      &concat_rsv_table_dense);
  } else  {
    ws->Attach(concat_rsv,
      std::vector<index_t>(), 
      std::vector<index_t>(1, global_dimensionality),
      references_filenames.size()*n_attributes,
      &concat_rsv_table_sparse);
  }
  int32 results_merged=0;
  std::list<std::pair<std::string, int32> > result_tables;
  for(int32 i=0; i<references_filenames.size(); ++i) {
    result_tables.push_back(
        std::make_pair(ws->GiveFilenameFromSequence(rsv_prefix, i), i));
  }
  typename WorkSpaceType::MatrixTable_t::Point_t point;
  index_t counter=0;
  while(results_merged<references_filenames.size()) {
    for(std::list<std::pair<std::string, int32> >::iterator it=result_tables.begin();
        it!=result_tables.end(); ++it) {
      if (ws->IsTableAvailable(it->first)) {
        boost::shared_ptr<
          typename WorkSpaceType::MatrixTable_t
        > rsv_table;
        boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> lsv_table;
        ws->Attach(ws->GiveFilenameFromSequence(lsv_prefix, it->second), &lsv_table);
        typename WorkSpaceType::MatrixTable_t::Point_t point;
        ws->Attach(it->first, &rsv_table);
        if (!has_point_ids 
            && rsv_table->n_entries()!=concat_rsv_table_dense->n_attributes()) {
          fl::logger->Die()<<"It seems to me that your data chunks have different dimensionalities "
            "consider using the global dimensionality option";
        }
        for(index_t j=0; j<rsv_table->n_attributes(); ++j) { 
          std::vector<std::pair<index_t, double> > loading_point;
          for(index_t i=0; i<rsv_table->n_entries(); ++i) {
            lsv_table->get(i, &point);
            if (has_point_ids) {
              index_t id=point.meta_data().template get<2>();
              if (id>=global_dimensionality || id<0) {
                fl::logger->Die()<<"found point id ("<<id<<") outside the number of columns "
                  "divided by --global_dimensionality="<<global_dimensionality;
              }
              loading_point.push_back(std::make_pair(id, rsv_table->get(i, j)));
            } else {
              concat_rsv_table_dense->set(counter, 
                  i,
                  rsv_table->get(i, j));
            }
          }
          if (has_point_ids) {
             typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t point;
             concat_rsv_table_sparse->get(counter, 
                  &point);
             point. template sparse_point<double>().Load(loading_point.begin(), loading_point.end());
          }
          counter++;
        }
        ws->Purge(it->first);
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
  svd_args.Add(fl::ws::MakeArgsFromPrefix(args, "svd2"));
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
    references_table->filename()=references[i];
    boost::shared_ptr<typename WorkSpaceType::MatrixTable_t>
      lsv_table; 
    ws_->Attach(lsv_out[i],
               std::vector<index_t>(1,rsv_table->n_attributes()),
               std::vector<index_t>(), 
               references_table->n_entries(),
               &lsv_table);
    {
      if (references_table->n_entries()>=2) {
        typename TableType::Point_t p1,p2;
        references_table->get(0, &p1);
        references_table->get(1, &p2);
        if (p1.meta_data().template get<2>()!=
            p2.meta_data().template get<2>()) {
          typename TableType::Point_t ref_point;
          typename WorkSpaceType::MatrixTable_t::Point_t lsv_point;
          for(index_t i=0; i<references_table->n_entries(); ++i) {
            references_table->get(i, &ref_point);
            lsv_table->get(i, &lsv_point);
            lsv_point.meta_data().template get<2>()=
              ref_point.meta_data().template get<2>();
          }
        }
      }
    }
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

  // Check if the dimensions do not match. If they don't then this is
  // probably because the references have different dimensions
  if (rsv_table->n_entries()!=references_table->n_attributes()) {
    typename WorkSpaceType::MatrixTable_t rsv_table1;
    rsv_table1.Init("",
        std::vector<index_t>(1, rsv_table->n_attributes()),
        std::vector<index_t>(),
        references_table->n_attributes());
    typename TableType::Point_t point;
    typename WorkSpaceType::MatrixTable_t::Point_t p1, p2;
    for(index_t i=0; i<references_table->n_entries(); ++i) {
      references_table->get(i, &point);
      rsv_table->get(point.meta_data().template get<2>(), &p1);
      rsv_table1.get(i, &p2);
      p2.CopyValues(p1);
    }
    fl::table::Mul<fl::la::NoTrans, fl::la::NoTrans>(
        *references_table,
        rsv_table1, 
        lsv_table.get());
  } else {
    fl::table::Mul<fl::la::NoTrans, fl::la::NoTrans>(
        *references_table,
        *rsv_table, 
        lsv_table.get());
  }
  ws->Purge(lsv_name);
  ws->Detach(lsv_name);
  ws->Purge(rsv_table->filename());
  ws->Detach(rsv_table->filename());
  ws->Purge(references_table->filename());
  ws->Detach(references_table->filename());
}

#endif

