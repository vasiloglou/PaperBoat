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

#ifndef FASTLIB_TABLE_SELF_OUTER_PROD_DEFS_H_
#define FASTLIB_TABLE_SELF_OUTER_PROD_DEFS_H_
#include "fastlib/base/base.h"
#include "fastlib/table_util/self_outer_prod.h"
#include "fastlib/table/linear_algebra.h"
#include "fastlib/base/logger.h"
#include "fastlib/workspace/based_on_table_run.h"
#include "fastlib/util/string_utils.h"


namespace fl { namespace table {

template<typename WorkSpaceType>
template<typename TableType>
void SelfOuterProd<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
  FL_SCOPED_LOG(SouterProd);
  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::string>(),
    "REQUIRED file containing reference data"
  )(
    "result_out",
    boost::program_options::value<std::string>(),
    "REQUIRED file to store the result"
  )(
    "mode",
    boost::program_options::value<std::string>()->default_value("sparse"),
    "OPTIONAL if your --references_in table is not long then you "
    "might want to use --mode=dense. If your result is expected to be "
    " sparse then use --mode=sparse"   
  )(
    "clip_value",
    boost::program_options::value<double>()->default_value(-std::numeric_limits<double>::max()),
    "OPTIONAL when in sparse mode if you want to clip values that are too small set this flag. "
    "Everything below this flag will be clipped to zero"  
  );

  boost::program_options::variables_map vm;
  boost::program_options::command_line_parser clp(args_);
  clp.style(boost::program_options::command_line_style::default_style
     ^boost::program_options::command_line_style::allow_guessing );
  
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
    fl::logger->Die() << e.what() << std::endl;
  }
  catch ( const boost::program_options::error &e) {
    fl::logger->Die() << e.what();
  } 

  boost::program_options::notify(vm);
  if (vm.count("help")) {
    std::cout << fl::DISCLAIMER << "\n";
    std::cout << desc << "\n";
    return ;
  }

  if (vm.count("references_in")==0) {
    fl::logger->Die()<<"You need to define --references_in";
  } 
  if (vm.count("result_out")==0) {
    fl::logger->Die()<<"You need to define --result_out";
  }
  boost::shared_ptr<TableType> references_table;
  ws_->Attach(vm["references_in"].as<std::string>(),
      &references_table);
  fl::logger->Message()<<"Computing X * X^T"<<std::endl;
  if (vm["mode"].as<std::string>()=="dense") {
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> result_table; 
    ws_->Attach(vm["result_out"].as<std::string>(),
          std::vector<index_t>(1, references_table->n_entries()),
          std::vector<index_t>(),
          references_table->n_entries(),
          &result_table
        );
    fl::table::SelfOuterProduct<TableType, WorkSpaceType>(*references_table,
        result_table.get());
    ws_->Purge(result_table->filename());
    ws_->Detach(result_table->filename());
  } else {
    if (vm["mode"].as<std::string>()=="sparse") {
      boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t> result_table; 
      ws_->Attach(vm["result_out"].as<std::string>(),
          std::vector<index_t>(),
          std::vector<index_t>(1, references_table->n_entries()),
          references_table->n_entries(),
          &result_table
        );
    fl::table::SelfOuterProduct<TableType, WorkSpaceType>(*references_table,
        vm["clip_value"].as<double>(),
        result_table.get());
    ws_->Purge(result_table->filename());
    ws_->Detach(result_table->filename());
    } else {
      fl::logger->Die()<<"--mode can be either dense or sparse";
    }
  }
  fl::logger->Message()<<"Done"<<std::endl;
}

template<typename WorkSpaceType>
int SelfOuterProd<boost::mpl::void_>::Run(
    WorkSpaceType *ws,
    const std::vector<std::string> &args) {

  bool found=false;
  std::string references_in;
  for(size_t i=0; i<args.size(); ++i) {
    if (fl::StringStartsWith(args[i],"--references_prefix_in=")) {
      found=true;
      std::vector<std::string> tokens=fl::SplitString(args[i], "=");
      if (tokens.size()!=2) {
        fl::logger->Die()<<"Something is wrong with the --references_in flag";
      }
      references_in=tokens[1]+"0";
      break;
    }
    if (fl::StringStartsWith(args[i],"--references_in=")) {
      found=true;
      std::vector<std::string> tokens=fl::SplitString(args[i], "=");
      if (tokens.size()!=2) {
        fl::logger->Die()<<"Something is wrong with the --references_in flag";
      }
      std::vector<std::string> filenames=fl::SplitString(tokens[1], ":,"); 
      references_in=filenames[0];
      break;
    }
  }

  if (found==false) {
    Core<WorkSpaceType> core(ws, args);
    typename WorkSpaceType::DefaultTable_t t;
    core(t);
    return 1;
  }

  Core<WorkSpaceType> core(ws, args);
  fl::ws::BasedOnTableRun(ws, references_in, core);
  return 0;
}

template<typename WorkSpaceType>
SelfOuterProd<boost::mpl::void_>::Core<WorkSpaceType>::Core(
   WorkSpaceType *ws, const std::vector<std::string> &args) :
 ws_(ws), args_(args)  {}



}}
#endif
