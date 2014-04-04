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

#ifndef FL_LITE_FASTLIB_TABLE_UTIL_TABLE_SUMMARY_DEFS_H_
#define FL_LITE_FASTLIB_TABLE_UTIL_TABLE_SUMMARY_DEFS_H_
#include <set>
#include <unordered_map>
#include "fastlib/table_util/table_summary.h"
#include "fastlib/base/logger.h"
#include "fastlib/workspace/based_on_table_run.h"
#include "fastlib/util/string_utils.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/math/fl_math.h"

namespace fl {namespace table {

template<typename WorkSpaceType>
template<typename TableType>
void TableSummary<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::string>(),
    "file or comma separating list of files containing reference data"
  )(
    "references_prefix_in",
    boost::program_options::value<std::string>(),
    "prefix for reference data"
  )(
    "references_num_in",
    boost::program_options::value<int32>(),
    "number of references_in as loaded by references having a prefix"
  )(
    "cont_granularity",
    boost::program_options::value<int32>()->default_value(10),
    "granullarity for histogram of continuous values"
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
    return;
  }
  std::vector<std::string> references=fl::ws::GetFileSequence("references", vm);
  boost::shared_ptr<TableType> table;
  index_t n_attributes=0;
  ws_->GetTableInfo(references[0], NULL, &n_attributes, NULL, NULL);
  std::vector<double> means;
  std::vector<double> variances;
  std::set<double> zero_attributes;
  std::vector<double> minimums;
  std::vector<double> maximums;


  means.resize(n_attributes);
  std::fill(means.begin(), means.end(), 0.0);
 
  variances.resize(n_attributes);
  std::fill(variances.begin(), variances.end(), 0.0);
 
  
  minimums.resize(n_attributes);
  std::fill(minimums.begin(), minimums.end(), 
        std::numeric_limits<double>::max());
 
  
  maximums.resize(n_attributes);
  std::fill(maximums.begin(), maximums.end(),
        -std::numeric_limits<double>::max());
  
  std::unordered_map<int32, index_t> class_label_statistics;
  std::unordered_map<int32, index_t> cont_label_statistics;
  //int32 cont_granularity=vm["cont_granularity"].as<int32>();
  index_t n_entries=0;
  for(auto reference : references) {
    ws_->Attach(reference, &table);
    typename TableType::Point_t point;
    typename TableType::Point_t::iterator it;
    n_entries+=table->n_entries();
    for(index_t i=0; i<table->n_entries(); ++i) {
      table->get(i, &point);
      for(it=point.begin(); it!=point.end(); ++it) {
        means[it.attribute()]+=static_cast<double>(it.value());
        minimums[it.attribute()]=std::min(minimums[it.attribute()], static_cast<double>(it.value()));
        maximums[it.attribute()]=std::max(maximums[it.attribute()], static_cast<double>(it.value()));
      }
    }
    ws_->Purge(reference);
    ws_->Detach(reference);
  }
  
  for(size_t i=0; i<n_attributes; ++i) {
    means[i]/=n_entries;
 
    if ((minimums[i]==maximums[i])==0) {
      zero_attributes.insert(i);    
    }
  }
  std::string mean_str;
  for(auto i=0; i<means.size(); ++i) {
    mean_str+=boost::lexical_cast<std::string>(i)+":"+boost::lexical_cast<std::string>(means[i])+",";
  }
  fl::logger->Message()<<"Means="<<mean_str;
  std::string zero_attributes_str;
  for(auto z : zero_attributes) {
    zero_attributes_str+=boost::lexical_cast<std::string>(z)+",";
  }
  fl::logger->Message()<<"Zero Attributes="<<zero_attributes.size()<<"  "<<zero_attributes_str;
  std::string bound_str;
  for(auto i=0;i<minimums.size(); ++i) {
    bound_str+=boost::lexical_cast<std::string>(i)+"["+
      boost::lexical_cast<std::string>(minimums[i])+","
      +boost::lexical_cast<std::string>(maximums[i])+"]";
  }
  fl::logger->Message()<<"Bounds="<<bound_str;
  for(auto reference : references) {
    ws_->Attach(reference, &table);
    typename TableType::Point_t point;
    typename TableType::Point_t::iterator it;
    for(index_t i=0; i<n_entries; ++i) {
      table->get(i, &point);
      for(it=point.begin(); it!=point.end(); ++it) {
        variances[it.attribute()]+=
          fl::math::Pow<double, 2, 1>(static_cast<double>(it.value())-means[it.attribute()]);
      }
    }
    ws_->Purge(reference);
    ws_->Detach(reference);
  }
  for(size_t i=0; i<variances.size(); ++i) {
    variances[i]/=n_entries;
  }
  std::string variance_str;
  for(auto i=0; i<variances.size(); ++i) {
    variance_str+=boost::lexical_cast<std::string>(i)+":"+boost::lexical_cast<std::string>(variances[i])+",";
  }
  fl::logger->Message()<<"Variances="<<variance_str;


  return ;
}


template<typename WorkSpaceType>
int TableSummary<boost::mpl::void_>::Run(
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
      references_in=ws->GiveFilenameFromSequence(tokens[1], 0);
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
TableSummary<boost::mpl::void_>::Core<WorkSpaceType>::Core(
   WorkSpaceType *ws, const std::vector<std::string> &args) :
 ws_(ws), args_(args)  {}


}}

#endif
