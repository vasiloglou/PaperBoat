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

#ifndef FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_MASTER_SLAVE_DEFS_H_
#define FL_LITE_MLPACK_REGRESSION_LINEAR_REGRESSION_MASTER_SLAVE_DEFS_H_
#include "linear_regression_master_slave.h"

bool fl::ml::LinearRegressionMasterSlave<boost::mpl::void_>::ConstructBoostVariableMap(
  const std::vector<std::string> &args,
  std::vector<std::vector<std::string> > *task_arguments,
  boost::program_options::variables_map *vm) {

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////
  boost::program_options::options_description desc("Available options");
  desc.add_options()
  ("help", "Display help on ridge regression")
  ("algorithm", boost::program_options::value<std::string>()->default_value(
     "fast"),
   "The type of algorithm to use: One of\n"
   "  fast, naive"
  )
  ("conf_prob", boost::program_options::value<double>()->default_value(0.9),
   "Specifies the probability converage of the confidence intervals."
  )
  ("exclude_bias_term",
   "If present, the bias term will not be included in the linear model."
  )
  ("equality_constraints_in", boost::program_options::value<std::string>(),
   "data file containing the equality constraints. If present, the linear "
   "model will be adjusted according to this constraint.")
  ("queries_in", boost::program_options::value<std::vector<std::string> >(),
   "data file containing the test set to be evaluated on the --run_mode="
   "eval mode."
  )
  ("references_in", boost::program_options::value<std::vector<std::string> >(),
   "data file containing the predictors and the predictions")
  ("remove_index_prefixes",
   boost::program_options::value< std::vector< std::string> >(),
   "The list of strings, each of which denotes the prefix that should be"
   " removed from the consideration of predictor set. We remove one "
   "feature for each prefix.")
  ("prune_predictor_index_prefixes",
   boost::program_options::value< std::vector< std::string > >(),
   "The list of strings, each of which denotes the prefix that should be "
   "included in the consideration of variance inflation factor based "
   "pruning.")
  ("prediction_index_prefix",
   boost::program_options::value< std::string >(),
   "The index for which should be used as the prediction.")
  ("correlation_pruning",
   "If present, the correlation coefficient should be used for initial "
   "pruning.")
  ("stepwise",
   "If present, the stepwise regression should be used after VIF "
   "selection.")
  ("stepdirection",
   boost::program_options::value< std::string >()->default_value("bidir"),
   "The direction of stepwise regression: One of\n"
   "  bidir, forward, backward"
  )
  ("input_coeffs",
   boost::program_options::value<std::string>(),
   "The input file for the coefficients"
  )
  ("input_coeffs_bias_term_index",
   boost::program_options::value<int>(),
   "The index position of the bias term if present in the input coefficients."
  )
  ("output_coeffs",
   boost::program_options::value<std::string>()->default_value("coefficients.txt"),
   "The output file for the coefficients")
  ("output_standard_errors",
   boost::program_options::value<std::string>()->default_value("standard_errors.txt"),
   "The output file for the standard errors")
  ("output_conf_los",
   boost::program_options::value<std::string>()->default_value("conf_los.txt"),
   "The output file for the lower bound for the confidence intervals")
  ("output_conf_his",
   boost::program_options::value<std::string>()->default_value("conf_his.txt"),
   "The output file for the upper bound for the confidence intervals")
  ("output_t_values",
   boost::program_options::value<std::string>()->default_value("t_values.txt"),
   "The output file for the t values")
  ("output_p_values", boost::program_options::value<std::string>()->default_value("p_values.txt"),
   "The output file for the p values")
  ("output_adjusted_r_squared",
   boost::program_options::value<std::string>()->default_value("adjusted_r_squared.txt"),
   "The output file for the adjusted r-squared value")
  ("output_f_statistic",
   boost::program_options::value<std::string>()->default_value("f_statistic"),
   "The output file for the f-statistics")
  ("output_r_squared",
   boost::program_options::value<std::string>()->default_value("r_squared.txt"),
   "The output file for the r-squared value")
  ("output_sigma",
   boost::program_options::value<std::string>()->default_value("sigma.txt"),
   "The output file for the sigma")
  (
    "point",
    boost::program_options::value<std::string>()->default_value("dense"),
    "Point type used by KDE.  One of:\n"
    "  dense, sparse, dense_sparse, categorical, dense_categorical"
  )(
    "tree",
    boost::program_options::value<std::string>()->default_value("balltree"),
    "Tree structure used by KDE.  One of:\n"
    "  kdtree, balltree"
  )("correlation_threshold",
    boost::program_options::value<double>()->default_value(0.9),
    "During the correlation pruning, each attribute will be pruned if "
    "it has absolute correlation factor more than this value with any other "
    "attributes."
   )
  ("stepwise_threshold",
   boost::program_options::value<double>()->default_value(
     std::numeric_limits<double>::min()),
   "A minimum required improvement in the score needed during stepwise "
   "regression in order to continue the selection."
  )
  (
    "vif_threshold",
    boost::program_options::value<double>()->default_value(8.0),
    "During the pruning stage via variance inflation factor, "
    "each attribute will be pruned if the threshold exceeds this value."
  )("run_mode",
    boost::program_options::value<std::string>()->default_value("train"),
    "Linear regression as every machine learning algorithm has two modes, "
    "the training and the evaluations."
    " When you set this flag to --run_mode=train it will find the set of "
    "coefficients."
    " Once you have found these coefficients and you want to predict on a "
    "new set of points, you should run it in the --run_mode=eval. Of course "
    "you should provide the data to be evaluated"
    " over the linear regression coefficients, by setting the --queries_in "
    "flag appropriately."
   )(
     "log",
     boost::program_options::value<std::string>()->default_value(""),
     "A file to receive the log, or omit for stdout."
   )(
     "loglevel",
     boost::program_options::value<std::string>()->default_value("debug"),
     "Level of log detail.  One of:\n"
     "  debug: log everything\n"
     "  verbose: log messages and warnings\n"
     "  warning: log only warnings\n"
     "  silent: no logging"
   )("host", boost::program_options::value<std::vector<std::string> >(), 
     "The hosts for processing, must be in the form url:port or url (defaults to port )"
     "or just port number, (defaults to localhost)");

  boost::program_options::command_line_parser clp(args);
  clp.style(boost::program_options::command_line_style::default_style
            ^ boost::program_options::command_line_style::allow_guessing);
  try {
    boost::program_options::store(clp.options(desc).run(), *vm);
  }
  catch (const boost::program_options::invalid_option_value &e) {
    fl::logger->Die() << "Invalid Argument: " << e.what();
  }
  catch (const boost::program_options::invalid_command_line_syntax &e) {
    fl::logger->Die() << "Invalid command line syntax: " << e.what();
  }
  catch (const boost::program_options::unknown_option &e) {
    fl::logger->Die() << "Unknown option: " << e.what() ;
  }
  boost::program_options::notify(*vm);
  if (vm->count("help")) {
    std::cout << fl::DISCLAIMER << "\n";
    std::cout << desc << "\n";
    return true;
  }

  // Set the logger level.
  fl::logger->SetLogger((*vm)["loglevel"].as<std::string>());

  // Do argument checking.
  if (!(vm->count("references_in"))) {
    fl::logger->Die() << "Missing required --references_in";
    return true;
  }


  std::string run_mode_in;
  try {
    run_mode_in = (*vm)["run_mode"].as<std::string>();
  }
  catch (const boost::bad_lexical_cast &e) {
    fl::logger->Die() << "Flag --run_mode must be set to a string";
  }
  if (run_mode_in != "train" && run_mode_in != "eval") {
    fl::logger->Die() << "--run_mode accepts one of: train, eval";
  }
  if (run_mode_in == "eval" && vm->count("queries_in") == 0) {
    fl::logger->Die() << "--run_mode=eval requires --queries_in argument.";
  }
  if (!vm->count("host")) {
    fl::logger->Die()<<"you should provide the hosts";
  }

  index_t num_of_tasks=0;
  if (vm->count("references_in")) {
    std::vector<std::string> references_in=vm->operator[]("references_in").as<
      std::vector<std::string> >();
    num_of_tasks=references_in.size();
  }
  if (vm->count("queries_in")) {
    std::vector<std::string> queries_in=vm->operator[]("queries_in").as<
      std::vector<std::string> >();
    num_of_tasks=queries_in.size();
   }
 // now start constructing the arguments for every task
  task_arguments->resize(num_of_tasks);
  // first copy all the 
  for(int i=0; i<args.size(); ++i) {
    if (args[i].find("references_in")==std::string::npos &&
        args[i].find("queriess_in")==std::string::npos &&
        args[i].find("host")==std::string::npos) {
      for(int j=0; j<task_arguments->size(); ++j) {
        task_arguments->operator[](j).push_back(args[i]);
        if (args[i].find("output")!=std::string::npos) {
          task_arguments->operator[](j).back().append(boost::lexical_cast<std::string>(j));         
        }
      }
    }
  }
  // now append on the command line arguments the references_in
  if (vm->count("references_in")) {
    std::vector<std::string> references_in=vm->operator[]("references_in").as<
      std::vector<std::string> >();
    for(index_t j=0; j<references_in.size(); ++j) {
      std::string temp="--references_in=";
      temp.append(references_in[j]);
      task_arguments->operator[](j).push_back(temp);
    }
//    for(index_t j=0; j<references_in.size();++j) {
//      for(index_t i=0; i<task_arguments->operator[](j).size();++i) {
//        std::cout<<"\n"<< task_arguments->operator[](j)[i]<<std::endl;    
//      }
//    }
  }
  if (vm->count("queries_in")) {
    std::vector<std::string> queries_in=vm->operator[]("queries_in").as<
      std::vector<std::string> >();
    for(index_t j=0; j<queries_in.size(); ++j) {
      std::string temp="--queries_in=";
      temp.append(queries_in[j]);
      task_arguments->operator[](j).push_back(temp);
    }
  }

  return false;
}

fl::ml::LinearRegressionMasterSlave<boost::mpl::void_>::ResponderMaster::ResponderMaster(
    std::vector<std::string> *args) : args_(args) {

}

bool fl::ml::LinearRegressionMasterSlave<boost::mpl::void_>::ResponderMaster::operator()(
    const boost::system::error_code& e,
    fl::com::connection_ptr conn) {
    //send the arguments
    conn->sync_write(*args_);
    // wait for the signal that everything is done
    int done;
    conn->sync_read(done);
    return true;
}

template<typename DataAccessType>
fl::ml::LinearRegressionMasterSlave<boost::mpl::void_>::TaskMaster<DataAccessType>::TaskMaster(
    boost::mutex *host_port_mutex,
    boost::mutex *finished_tasks_mutex,
    std::deque<std::pair<std::string, std::string> > *host_ports,
    index_t *finished_tasks,
    index_t total_tasks,
    DataAccessType *data,
    ResponderMaster *responder) : host_port_mutex_(host_port_mutex),
  finished_tasks_mutex_(finished_tasks_mutex),
  finished_tasks_(finished_tasks),
  total_tasks_(total_tasks),
  host_ports_(host_ports),
  data_(data),
  responder_(responder) {
                                             
  }

template<typename DataAccessType>
void fl::ml::LinearRegressionMasterSlave<boost::mpl::void_>::TaskMaster<DataAccessType>::operator()() {
  boost::asio::io_service io_service;
  std::pair<std::string, std::string> host_port_pair;
  // we need to find which server is available
  // and get the address. For that reason we 
  // want to lock the deque with the available hosts
  {
    boost::mutex::scoped_lock lock(*host_port_mutex_);
    host_port_pair=host_ports_->front();
    host_ports_->pop_front();
  }
  // now we start the client
  fl::com::Client<ResponderMaster> client(io_service, 
      host_port_pair.first,
      host_port_pair.second,
      responder_);
  io_service.run();
  // after we are done
  // we need to return the server address
  // back to the available ones
  host_port_mutex_->lock(); 
  host_ports_->push_back(host_port_pair);
  host_port_mutex_->unlock();
  // now say that you finished the task
  finished_tasks_mutex_->lock();
  (*finished_tasks_)++;
  if (*finished_tasks_==total_tasks_) {
    // send a signal to shutdwon the data access server
    data_->StopServer();
  }
  finished_tasks_mutex_->unlock();
}

template<class DataAccessType, typename BranchType>
fl::ml::LinearRegressionMasterSlave<boost::mpl::void_>::ResponderSlave<DataAccessType, BranchType>::ResponderSlave(
    DataAccessType *data) : data_(data)  {

}

template<class DataAccessType, typename BranchType>
bool fl::ml::LinearRegressionMasterSlave<boost::mpl::void_>::ResponderSlave<DataAccessType, BranchType>::operator()(
    const boost::system::error_code& e,
    fl::com::connection_ptr conn) {
    std::vector<std::string> args;
    //receive the arguments
    conn->sync_read(args);
    for(index_t i=0; i<args.size(); ++i) {
      std::cout<<args[i]<<std::endl;
    }

    int done;
    try {
      fl::ml::LinearRegression<boost::mpl::void_>::Main<DataAccessType, BranchType>(
        data_, args);
       done=0; 
    }
    catch(const fl::Exception &ex) {
      done=-1;
    } 
    catch(const std::exception &ex) {
      fl::logger->Message() <<ex.what();
    }
    // send the signal that everything is done
    conn->sync_write(done);
    return true;
}

template<class DataAccessType>
int fl::ml::LinearRegressionMasterSlave<boost::mpl::void_>::MainMaster(
  DataAccessType *data,
  const std::vector<std::string> &args
) {

  boost::program_options::variables_map vm;
  std::vector<std::vector<std::string> > task_arguments;
  bool help_specified = ConstructBoostVariableMap(args, &task_arguments, &vm);
  if (help_specified) {
    return 1;
  }

  std::vector<std::string> hosts=
  vm["host"].as<std::vector<std::string> >();
  std::deque<std::pair<std::string, std::string > > host_port; 
  for(index_t i=0; i<hosts.size(); ++i) {
    std::vector<std::string> tokens;
    boost::algorithm::split(tokens, hosts[i], boost::algorithm::is_any_of(":"));
    if (tokens.size()==2) {
      host_port.push_back(std::make_pair(tokens[0], tokens[1]));
    } else {
      if (tokens.size()==1) {
        try {
          host_port.push_back(std::make_pair(std::string("127.0.0.1"), tokens[0]));
        }
        catch(const boost::bad_lexical_cast &e) {
          fl::logger->Warning()<<"No port specified for host " <<
              tokens[0] << ", defaulting to port " << fl::com::DEFAULT_PORT<<std::endl;
          host_port.push_back(std::make_pair(tokens[0], std::string(fl::com::DEFAULT_PORT_STR)));
        }
      } else {
        fl::logger->Die() << "The host option " <<hosts[i]<<" is incorrect, "
            <<"it must be of the form  url:port or url or port";
      }
    }
  }
  // make the responders
  std::vector<boost::shared_ptr<ResponderMaster> > responders(task_arguments.size());
  for(index_t i=0; i<responders.size(); ++i) {
    responders[i].reset(new ResponderMaster(&task_arguments[i]));
  }
  boost::mutex host_port_mutex;
  boost::mutex finished_task_mutex;
  boost::threadpool::pool tp(host_port.size());

  std::vector<boost::shared_ptr<TaskMaster<DataAccessType> > > tasks(task_arguments.size());
  index_t finished_tasks=0;
  for(index_t i=0; i<task_arguments.size(); ++i) {
    tasks[i].reset(new TaskMaster<DataAccessType>(&host_port_mutex, 
          &finished_task_mutex,
          &host_port, 
          &finished_tasks,
          task_arguments.size(),
          data,
          responders[i].get()));
    tp.schedule(*tasks[i]);
  }
  return 0;
}

template<class DataAccessType, typename BranchType>
int fl::ml::LinearRegressionMasterSlave<boost::mpl::void_>::MainSlave(
  DataAccessType *data,
  const std::vector<std::string> &args) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()
  ("help", "Display help on ridge regression")
  ("port", boost::program_options::value<std::string>()->default_value("4554"),
   "Port number for running the service");
  boost::program_options::variables_map vm;
  boost::program_options::command_line_parser clp(args);
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
    fl::logger->Die() << "Unknown option: " << e.what() ;
  }
   
  boost::program_options::notify(vm);
  if (vm.count("help")) {
    std::cout << fl::DISCLAIMER << "\n";
    std::cout << desc << "\n";
    return 1;
  }

   ResponderSlave<DataAccessType, BranchType> responder(data);
   boost::asio::io_service io_service;
   fl::com::Server<ResponderSlave<DataAccessType, BranchType> > 
     server(io_service, vm["port"].as<std::string>(), &responder);
   io_service.run();
   return 0;
}
#endif
