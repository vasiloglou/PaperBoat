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

#ifndef PAPERBOAT_MLPACK_TIME_SERIES_TIME_SERIES_SMOOTHING_H_
#define PAPERBOAT_MLPACK_TIME_SERIES_TIME_SERIES_SMOOTHING_H_
#include "boost/program_options.hpp"
#include "time_series_smoothing.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/workspace/based_on_table_run.h"
#include "fastlib/util/string_utils.h"
#include "holt_forecaster.h"
#include "holt_winters_additive_forecaster.h"
#include "holt_winters_multiplicative_forecaster.h"
#include "croston_forecaster.h"

namespace fl { namespace ml {

  template<typename WorkSpaceType>
  template<typename TableType>
  void TimeSeriesSmoothing<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
    FL_SCOPED_LOG(TimeSeriesSmoothing);
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "references_in",
      boost::program_options::value<std::string>(),
      "the reference data "
    )(
      "alpha",
      boost::program_options::value<double>()->default_value(0.5),
      "the alpha parameter, it must be 0<a<1" 
    )(
      "beta_star",
      boost::program_options::value<double>()->default_value(0.3),
      "the beta star parameter it must be 0<b<a"
    )(
      "gamma",
      boost::program_options::value<double>()->default_value(0.45),
      "the gamma parameter"
    )(
      "horizon",
      boost::program_options::value<index_t>()->default_value(1),
      "the prediction horizon"
    )(
      "method",
      boost::program_options::value<std::string>->default("simple"),
      "simple \n"
      "croston \n"
      "holton_winters_additive \n"
      "holton_winters_multiplicative "
    )(
      "residual_out",
      boost::program_options::value<std::string>(),
      "the prediction residual"
    )(
      "smoothed_out",
      boost::program_options::value<std::string>(),
      "the smoothed output of the references_in"
    );

    boost::program_options::variables_map vm;
    std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args_, "");
    boost::program_options::command_line_parser clp(args1);
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
      return ;
    }
  }

  template<typename WorkSpaceType>
  int TimeSeriesSmoothing<boost::mpl::void_>::Run(
      WorkSpaceType *ws,
      const std::vector<std::string> &args) {

    bool found=false;
    std::string references_in;
    for(size_t i=0; i<args.size(); ++i) {
      if (fl::StringStartsWith(args[i],"--references_in=")) {
        found=true;
        std::vector<std::string> tokens=fl::SplitString(args[i], "=");
        if (tokens.size()!=2) {
          fl::logger->Die()<<"Something is wrong with the --references_in flag";
        }
        references_in=tokens[1];
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
  TimeSeriesSmoothing<boost::mpl::void_>::Core<WorkSpaceType>::Core(
     WorkSpaceType *ws, const std::vector<std::string> &args) :
   ws_(ws), args_(args)  {}



}}
