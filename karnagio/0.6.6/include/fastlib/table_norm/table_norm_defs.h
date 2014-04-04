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
#ifndef FL_LITE_FASTLIB_TABLE_NORM_TABLE_NORM_DEFS_H_
#define FL_LITE_FASTLIB_TABLE_NORM_TABLE_NORM_DEFS_H_
#include "fastlib/workspace/arguments.h"
#include "fastlib/table_norm/table_norm.h"
#include "fastlib/table/convert_table_to_double.h"
#include "fastlib/base/logger.h"
#include "fastlib/workspace/based_on_table_run.h"
#include "fastlib/util/string_utils.h"
#include "fastlib/math/fl_math.h"
#include "fastlib/la/linear_algebra.h"

namespace fl {namespace table {

template<typename WorkSpaceType>
template<typename TableType>
void TableNorm<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
  FL_SCOPED_LOG(Tnorm);
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
    "normalized_out",
    boost::program_options::value<std::string>(),
    "OPTIONAL, the output file "
  )(
    "normalized_prefix_out",
    boost::program_options::value<std::string>(),
    "the prefix for the output file, when we output it in a sequence of files"  
    " for example dummy1, dummy2, dummy3..."
  )(
    "normalized_num_out",
    boost::program_options::value<int32>(),
    "number of the output files, when using --normalized_prefix_out "
    "flag"
  )(
    "norm", 
    boost::program_options::value<std::string>()->default_value("rows_l2"),
    "REQUIRED The type can be:\n"
    "  rows_l1    : normalizes the rows with L1 metric\n"
    "  rows_l2    : normalizes the rows with L2 metric\n"
    "  cols_l1    : normalizes the columns with L1 metric\n"
    "  cols_l2    : normalizes the columns with L2 metric\n"
    "A valid option would also be rows_l1,columns_l1 or any combination "
   " between columns and row" 
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
  if (vm.count("normalized_out")==false && vm.count("normalized_prefix_out")==false) {
    fl::logger->Die()<<"What's the point running this util if you don't "
      "define a normalized_out table";
  }
  std::vector<std::string> references_out=fl::ws::GetFileSequence("normalized", vm);

  if (references_in.size()!=references_out.size()) {
    fl::logger->Die()<<"number of references and normalized tables must be the same";
  }
  typename TableType::Point_t point;
  std::vector<std::string> type=fl::SplitString(vm["norm"].as<std::string>(), ",");
  int row=0;
  int col=0;
  for(std::vector<std::string>::iterator it=type.begin(); it!=type.end(); ++it) {
    if (*it=="rows_l1") {
      row=1;
    } else {
      if (*it=="rows_l2") {
        row=2;
      } else {
        if (*it=="cols_l1") {
          col=1;
        } else {
          if (*it=="cols_l2") {
            col=2;
          } else {
            fl::logger->Die()<<"Unknown or wrong option for --norm="<<
              vm["norm"].as<std::string>();
          }
        }
      }
    }
  }

  std::vector<double> row_sums;
  std::vector<double> col_sums;
  index_t n_attributes;
  ws_->GetTableInfo(references_in[0], NULL, &n_attributes, NULL, NULL);
  if (col>0) {
    col_sums.resize(n_attributes);
  }
  fl::logger->Message() << "Computing row/column norms"<<std::endl;
  for(auto references_name=references_in.begin();
      references_name != references_in.end(); ++references_name) {
    boost::shared_ptr<TableType> references;
    ws_->Attach(*references_name,
         &references);
    for(index_t i=0; i<references->n_entries(); ++i) {
      references->get(i, &point);
      if (col==0) {
        if (row==2) {
          row_sums.push_back(fl::la::Dot(point, point));
        } else {
          if (row==1) {
            row_sums.push_back(0);
            for(typename TableType::Point_t::iterator it=point.begin();
                it!=point.end(); ++it) {
              row_sums.back() +=fabs(it.value());
            }
          } 
        }
      } else {
        if (col==1) {
          if (row==0) {
            for(typename TableType::Point_t::iterator it=point.begin();
                it!=point.end(); ++it) {
              col_sums[it.attribute()]+=fabs(it.value());
            }
          } else {
            if (row==1) {
              row_sums.push_back(0);
              for(typename TableType::Point_t::iterator it=point.begin();
                  it!=point.end(); ++it) {
                col_sums[it.attribute()]+=fabs(it.value());
                row_sums.back() +=fabs(it.value());
              }
            } else {
              row_sums.push_back(0);
              for(typename TableType::Point_t::iterator it=point.begin();
                  it!=point.end(); ++it) {
                col_sums[it.attribute()]+=fabs(it.value());
                row_sums.back() +=fl::math::Pow<double,2,1>(it.value());
              }
            }
          }
        } else {
          if (row==0) {
            for(typename TableType::Point_t::iterator it=point.begin();
                it!=point.end(); ++it) {
              col_sums[it.attribute()]+=fl::math::Pow<double,2,1>(it.value());
            }
          } else {
            if (row==1) {
              for(typename TableType::Point_t::iterator it=point.begin();
                  it!=point.end(); ++it) {
                col_sums[it.attribute()]+=fl::math::Pow<double,2,1>(it.value());
                row_sums.back() +=fabs(it.value());
              }
            } else {
              row_sums.push_back(0);
              for(typename TableType::Point_t::iterator it=point.begin();
                  it!=point.end(); ++it) {
                col_sums[it.attribute()]+=fl::math::Pow<double,2,1>(it.value());
                row_sums.back() +=fl::math::Pow<double,2,1>(it.value());
              }
            }
          }
        }
      }
      if (row==2) {
       row_sums.back()=fl::math::Pow<double,1,2>(row_sums.back());
      }
    }
    ws_->Purge(*references_name);
    ws_->Detach(*references_name);
  }

  if (col==2) {
    for(size_t i=0; i<col_sums.size(); ++i) {
      col_sums[i]=fl::math::Pow<double,1,2>(col_sums[i]); 
    }
  }
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
      Scaler scaler(row_sums.size()>0?row_sums[row_counter]:0, col_sums);
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
int TableNorm<boost::mpl::void_>::Run(
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
TableNorm<boost::mpl::void_>::Core<WorkSpaceType>::Core(
   WorkSpaceType *ws, const std::vector<std::string> &args) :
 ws_(ws), args_(args)  {}


}}

#endif
