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

#ifndef PAPERBOAT_FASTLIB_TABLE_SCALE_TABLE_SCALE_DEFS_H_
#define PAPERBOAT_FASTLIB_TABLE_SCALE_TABLE_SCALE_DEFS_H_
#include "fastlib/workspace/arguments.h"
#include "fastlib/table_scale/table_scale.h"
#include "fastlib/table/convert_table_to_double.h"
#include "fastlib/base/logger.h"
#include "fastlib/workspace/based_on_table_run.h"
#include "fastlib/util/string_utils.h"
#include "fastlib/math/fl_math.h"
#include "fastlib/la/linear_algebra.h"

namespace fl {namespace table {

template<typename WorkSpaceType>
template<typename TableType>
void TableScale<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
  FL_SCOPED_LOG(Tscale);
  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::string>(),
    "REQUIRED file containing reference data"
  )(
    "references_prefix_in",
    boost::program_options::value<std::string>(),
    "prefix for the reference files to be imported"
  )(
    "references_num_in",
    boost::program_options::value<int32>(),
    "number for reference files when used references_prefix. We assume "
    "that when references_prefix is dummy, then we assume that the input "
    "files are dummy1,dummy2,...,dummy11,dummy12..."
  )(
    "scaled_out",
    boost::program_options::value<std::string>(),
    "the output file "
  )(
    "scaled_prefix_out",
    boost::program_options::value<std::string>(),
    "the prefix for the output file, when we output it in a sequence of files"  
    " for example dummy1, dummy2, dummy3..."
  )(
    "scaled_num_out",
    boost::program_options::value<int32>(),
    "number of the output files, when using --normalized_prefix_out "
    "flag"
  )(
    "scale_in",
    boost::program_options::value<std::string>(),
    "the table that has the scaling factors. It can be either a table with one point "
    "that has dimensionality equal to the n_attributes of references_in or it is one dimensional "
    "but it has number of entries equal to the number of attributes of the references_in"    
  )(
    "use_double_container",
    boost::program_options::value<bool>()->default_value(false),
    "sometimes the input table is in low precision like bool, the program will convert everything to doubles "
    "so that it can do the normalizations"
  );

  boost::program_options::variables_map vm;
  boost::program_options::command_line_parser clp(args_);
  clp.style(boost::program_options::command_line_style::default_style
     ^boost::program_options::command_line_style::allow_guessing);
  
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
 
  std::vector<std::string> references_in=fl::ws::GetFileSequence("references", vm);
  if (vm.count("scaled_out")==false && vm.count("scaled_prefix_out")==false) {
    fl::logger->Die()<<"What's the point running this util if you don't "
      "define a normalized_out table";
  }
  if (references_in.size()==0) {
    fl::logger->Die()<<"You should provice either --references_in or --references_prefix_in";
  }
  std::vector<std::string> references_out=fl::ws::GetFileSequence("scaled", vm);

  if (references_in.size()!=references_out.size()) {
    fl::logger->Die()<<"number of references and scaled tables must be the same";
  }
  typename TableType::Point_t point;
  boost::shared_ptr<TableType> scale_table;
  if (vm.count("scale_in")==0) {
    fl::logger->Die()<<"You should provice a --scale_in table";
  }
  ws_->Attach(vm["scale_in"].as<std::string>(), &scale_table);
  std::vector<double> col_scaler;
  if (scale_table->n_entries()!=1) {
    for(index_t i=0; i<scale_table->n_entries(); ++i) {
      col_scaler.push_back(scale_table->get(i,index_t(0)));
    }
  } else {
    for(index_t i=0; i<scale_table->n_attributes(); ++i) {
      col_scaler.push_back(scale_table->get(0, i));
    }
  }
  ws_->Purge(vm["scale_in"].as<std::string>());
  ws_->Detach(vm["scale_in"].as<std::string>());
  fl::logger->Message()<<"Scaling the data"<<std::endl;
  index_t row_counter=0;
  for(index_t k=0; k<references_in.size(); ++k) {
    const auto references_name=references_in[k];
    boost::shared_ptr<TableType> references;
    ws_->Attach(references_name,
         &references);

    boost::shared_ptr<TableType> references_out_table;
    ws_->Attach(references_out[k], 
      references->dense_sizes(),
      references->sparse_sizes(), 
      references->n_entries(),
      &references_out_table);
    typename TableType::Point_t point1, point2;
  
    for(index_t i=0; i<references->n_entries(); ++i) {
      references->get(i, &point1);
      references_out_table->get(i, &point2);
      point2.CopyValues(point1);
      Scaler scaler(0, col_scaler);
      row_counter++;
      point2.Transform(scaler);
    }
    ws_->Purge(references_name);
    ws_->Detach(references_name);
    ws_->Purge(references_out[k]);
    ws_->Detach(references_out[k]);
  }
  fl::logger->Message()<<"Done"<<std::endl;
  return;
}


template<typename WorkSpaceType>
int TableScale<boost::mpl::void_>::Run(
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
TableScale<boost::mpl::void_>::Core<WorkSpaceType>::Core(
   WorkSpaceType *ws, const std::vector<std::string> &args) :
 ws_(ws), args_(args)  {}


}}

#endif
