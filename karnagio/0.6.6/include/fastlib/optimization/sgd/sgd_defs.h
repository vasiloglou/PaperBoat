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

#ifndef PAPERBOAT_FASTLIB_OPTIMIZATION_SGD_SGD_DEFS_H_
#define PAPERBOAT_FASTLIB_OPTIMIZATION_SGD_SGD_DEFS_H_

#include "sgd.h"
#include "fastlib/base/logger.h"
#include "boost/program_options.hpp"
#include <unordered_map>

namespace fl {namespace ml {

  template <typename FunctionType>
  void StochasticGradientDescent<FunctionType>::set_objective(FunctionType *function) {
    FL_SCOPED_LOG(Sgd);
    function_=function;
  }
  
  template <typename FunctionType>
  void StochasticGradientDescent<FunctionType>::set_initial_learning_rate(double eta0) {
    FL_SCOPED_LOG(Sgd);
    eta0_=eta0;
  }
 
  template <typename FunctionType>
  void StochasticGradientDescent<FunctionType>::set_iterations(int32 iterations) {
    FL_SCOPED_LOG(Sgd);
    iterations_=iterations;
  } 

  template <typename FunctionType>
  void StochasticGradientDescent<FunctionType>::set_epochs(int32 epochs) {
    FL_SCOPED_LOG(Sgd);
    epochs_=epochs;
  } 

  template <typename FunctionType>
  void StochasticGradientDescent<FunctionType>::set_optimization_parameters(const std::vector<std::string> &args) {
    FL_SCOPED_LOG(Sgd);
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "eta0",
      boost::program_options::value<double>()->default_value(1.0),
      "the initial learning rate"
    )(
      "iterations",
      boost::program_options::value<int32>()->default_value(100),
      "the number of iterations" 
    )(
      "epochs",
      boost::program_options::value<int32>()->default_value(1),
      "number of epochs, every iteration has epochs, on every iteration the eta is constant" 
     )(
       "max_trials",
       boost::program_options::value<int32>()->default_value(3),
       "the number of trials for updating the model on a point, for tuning the step eta"
     );

    boost::program_options::variables_map vm;
    boost::program_options::command_line_parser clp(args);
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
      return;
    }
    eta0_=vm["eta0"].as<double>();
    iterations_=vm["iterations"].as<int32>();
    epochs_=vm["epochs"].as<int32>();
    max_trials_=vm["max_trials"].as<int32>();
  }
  
  template <typename FunctionType>
  template<typename WorkSpaceType, typename TableType>
  bool StochasticGradientDescent<FunctionType>::Optimize(WorkSpaceType *ws,
                    const std::vector<std::string> &tables,
                    fl::data::MonolithicPoint<double> *model) {
    FL_SCOPED_LOG(Sgd);
    double eta=eta0_;
    typename TableType::Point_t point;
    index_t invalid_updates=0;
    index_t valid_updates=0;
    std::vector<size_t> table_index;
    for(size_t i=0; i<tables.size(); ++i) {
      table_index.push_back(i);
    }
    boost::shared_ptr<TableType> table;
    if (tables.size()==1) {
      ws->Attach(tables[0], &table);
    }
    for(int32 it=0; it<iterations_; ++it) {
      eta=eta0_/(it+1);
      for(int32 epoch=0; epoch<epochs_; ++epoch) {
        invalid_updates=0;
        valid_updates=0;
        std::random_shuffle(table_index.begin(), table_index.end());
        for(size_t i=0; i<tables.size(); ++i) {
          if (tables.size()>1) {
            ws->Attach(tables[table_index[i]], &table);
          }
          std::vector<index_t> indices;
          for(index_t j=0; j<table->n_entries(); ++j) {
            indices.push_back(j);
          }
          std::random_shuffle(indices.begin(), indices.end());
          for(auto iti=indices.begin(); iti!=indices.end(); ++iti) {
            table->get(*iti, &point);
            std::vector<std::pair<index_t, double> > local_gradient=function_->Gradient(*model, point);
            std::vector<std::pair<index_t, double> > cache;
            double previous_error=function_->LocalError(*model, point);
           
            double current_error=0; 
            
            double local_eta=eta;
              for(auto itl=local_gradient.begin(); itl!=local_gradient.end(); ++itl) {
                cache.push_back(std::make_pair(itl->first, (*model)[itl->first]));
                (*model)[itl->first]-=local_eta*itl->second;
              }
              current_error=function_->LocalError(*model, point);

//             for(index_t trial=0; trial<max_trials_; trial++) {
//               for(auto itl=local_gradient.begin(); itl!=local_gradient.end(); ++itl) {
//                 cache.push_back(std::make_pair(itl->first, (*model)[itl->first]));
//                 (*model)[itl->first]-=local_eta*itl->second;
//               }
//               current_error=function_->LocalError(*model, point);
//               //std::cout<<trial<<": "<<current_error<<" < "<<previous_error<<std::endl; 
//               if (current_error >= previous_error) {
//                 for(size_t k=0; k<cache.size(); ++k) {
//                   (*model)[cache[k].first]=cache[k].second;  
//                 }
//                 cache.clear();
//                 local_eta*=eta0_;
//               } else {
//                 break;
//               }
//             }
            
            //std::cout<<current_error <<" < "<<previous_error<<std::endl;

            if (current_error>previous_error) {
              invalid_updates++;
              for(size_t k=0; k<cache.size(); ++k) {
                 (*model)[cache[k].first]=cache[k].second;  
              }
            } else {
              valid_updates++;
            }
          }
          function_->set_references(table.get());
          fl::logger->Message()<<std::setprecision(2)<<tables[i]<<":"
            <<"iteration="<<it<<", epoch="<<epoch<<", "
            <<", eta="<<eta
            <<", objective="<<function_->Evaluate(*model)<<", "
            <<"invalid_updates="
            <<100*(1.0*invalid_updates/(invalid_updates+valid_updates))<<"%";
          if (valid_updates<invalid_updates) {
            epoch=epochs_;
          }
          if (tables.size()>1) {
            ws->Purge(tables[table_index[i]]);
            ws->Detach(tables[table_index[i]]);
          }
        }
      }
    }
    if (tables.size()==1) {
      ws->Purge(tables[0]);
      ws->Detach(tables[0]);
    }

    if (valid_updates==0) {
      return false;
    } else {
      return true;
    }
  }  

}}
#endif
