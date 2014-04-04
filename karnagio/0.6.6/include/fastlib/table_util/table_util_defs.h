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
#ifndef FL_LITE_FASTLIB_TABLE_UTIL_TABLE_UTIL_DEFS_H_
#define FL_LITE_FASTLIB_TABLE_UTIL_TABLE_UTIL_DEFS_H_
#include "fastlib/table_util/table_util.h"
#include "fastlib/base/logger.h"
#include "fastlib/workspace/based_on_table_run.h"
#include "fastlib/util/string_utils.h"

namespace fl {namespace table {

template<typename WorkSpaceType>
template<typename TableType>
void TableUtil<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::string>(),
    "REQUIRED file containing reference data"
  )(
    "labels_in",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file containing the labels of reference data"
  )(
    "references_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL, if you use the rqp or rqd task you have to specify it"
  )(
    "references_labels_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL if you choose the rqp or rqd task and your data is labeled then "
    "you have to specify it."
  )(
    "queries_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL You have to specify it if you choose the rqp or rqd task"
  )(
    "queries_labels_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL: You have to specify it if you choose the rqp or rqd task "
    "and your reference data have a labels file associated"
  )(
    "references_out_prefix",
    boost::program_options::value<std::string>()->default_value(""),
    "When you split a file into many files, use this flags for the prefix, "
    "and the program will append _{n} for every file. "
    " for example if --references_out_prefix=test and --n_files=3 it"
    " will output files test_0, test_1, test_2."
  )(
    "references_labels_out_prefix",
    boost::program_options::value<std::string>()->default_value(""),
    " This is the same as references_out_prefix but it is used for the files that "
    " contain the labels"
  )(
    "percentage",
    boost::program_options::value<double>(),
    "REQUIRED the percentage of the references_in that will be processed. "
    "For the rqp, rqd, it must be strictly  between 0 and 1, for the rnp and rnd "
    "it must be just greater than 0"
  )(
    "n_files",
    boost::program_options::value<int>()->default_value(2),
    "OPTIONAL, You have to specify it only if you use --task=rnp or rnd. This"
    " is the the number of files you want to split the references file"
  )(
    "task", 
    boost::program_options::value<std::string>(),
    "REQUIRED The task can be:\n"
    "  rqp : Splits the references_in in 2 files references_out and queries_out"
             " the queries_out size will be approximately sizeof(references_in)*percentage \n"
    "  rqd : Splits the references_in in 2 files references_out and queries_out"
             " both queries_out and references_out will have the same number of points"
             " If references_in has N nonzero entries then queries_out will have"
             " N*percentage nonzero entries \n"
    "  rlp : Splits the references according to the labels. The new files will have "
             " points of the same class label. You also need to provide the "
             "--references_out_prefix. The class identifier will be appended "
             "on every file\n"    
    "  rnp : Splits the references_in in --n_files files that will be named as "
             " {--references_out_prefix}_n with n ranging from 0 to {--n_files}-1,"
             " for example if --references_out_prefix=test and --n_files=3 it"
             " will output files test_0, test_1, test_2. The files will have"
             " approximately the same number of points. The total number of points"
             " of the output files will be sizeof({--references_in})*{--percentage}\n"
   "   rnd : Splits the references_in in --n_files files that will be named as "
             " {--references_out_prefix}_n with n ranging from 0 to {--n_files}-1,"
             " for example if --references_out_prefix=test and --n_files=3 it"
             " will output files test_0, test_1, test_2. If the --references_in"
             " has N nonzero entries then the sum of nonzero entries of all the"
             " resulting files will be N*{--perencentage}"
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

  std::string references_in;
  std::string labels_in;
  std::string references_out;
  std::string references_labels_out;
  std::string queries_out;
  std::string queries_labels_out;
  std::string references_out_prefix;
  std::string references_labels_out_prefix;
  double percentage=0;
  std::string task;
  // check if arguments exist

  if (!vm.count("task")) {
    fl::logger->Die() << "--task is a required option";
  }
  if (!vm.count("percentage") && vm["task"].as<std::string>()!="rlp") {
    fl::logger->Die() << "--percentage is a required option" ;
  }
  try {
    references_in = vm["references_in"].as<std::string>();
    labels_in=vm["labels_in"].as<std::string>();
    references_out=vm["references_out"].as<std::string>();
    references_labels_out=vm["references_labels_out"].as<std::string>();
    queries_out=vm["queries_out"].as<std::string>();
    queries_labels_out=vm["queries_labels_out"].as<std::string>();
    references_out_prefix=vm["references_out_prefix"].as<std::string>();
    references_labels_out_prefix=vm["references_labels_out_prefix"].as<std::string>();
    if (vm.count("percentage")) {
      percentage=vm["percentage"].as<double>();
    }
    task=vm["task"].as<std::string>();
  }
  catch(const boost::program_options::invalid_option_value &e) {
  	fl::logger->Die() << "Invalid Argument: " << e.what(); 
  }

  boost::shared_ptr<TableType> references;

  if (task=="rqp") {
    if (percentage<0 || percentage>1) {
      fl::logger->Die() << "--percentage=" <<percentage<< " must be between 0 and 1 "
        "for this task";
    }
    if (queries_out=="") {
      fl::logger->Die() << "For this task(="<< task << ") you should provide "
        "--queries_out option";
    }
    if (references_out=="") {
      fl::logger->Die() << "For this task(="<< task << ") you should provide "
        "--references_out option";
    }
    
    if (labels_in!="") {
      if (references_labels_out=="") {
        fl::logger->Die() << "For this task(="<< task <<") you should provide "
        "--references_labels_out option, since you have provided labels_in(="
        <<labels_in<<")";             
      }
      if (queries_labels_out=="") {
       fl::logger->Die() << "For this task(="<< task <<") you should provide "
        "--queries_labels_out option, since you have provided labels_in(="
        <<labels_in<<")";
      }      
    }
    ws_->Attach(references_in,
        &references);
    fl::logger->Message() << "Loading reference data from " <<references_in
      <<std::endl;
    boost::shared_ptr<TableType> train;
    boost::shared_ptr<typename WorkSpaceType::template TableVector<index_t> > train_labels;
    boost::shared_ptr<TableType> test;
    boost::shared_ptr<typename WorkSpaceType::template TableVector<index_t> > test_labels;
    boost::shared_ptr<typename WorkSpaceType::template TableVector<index_t> > labels;

    if (labels_in!="") {
      fl::logger->Message() << "Loading refernece labels from " <<labels_in <<std::endl;
      ws_->Attach(labels_in, &labels);
      ws_->Attach(references_labels_out,
        std::vector<index_t>(1, 1),
         std::vector<index_t>(),
        0,
        &train_labels);
      ws_->Attach(queries_labels_out,
        std::vector<index_t>(1, 1),
        std::vector<index_t>(),
        0,
        &test_labels);
    }
    ws_->Attach(references_out, 
        references->dense_sizes(),
        references->sparse_sizes(),
        0,
        &train);
    ws_->Attach(queries_out,
       references->dense_sizes(),
       references->sparse_sizes(),
       0,
       &test);
    if (labels_in=="") {
      fl::logger->Message() << "Splitting the reference file"<<std::endl;
      SplitPoints(*references, percentage, train.get(), test.get());
    } else  {
      fl::logger->Message() << "Splitting the reference file with labels"<<std::endl;
      SplitPoints(*references, 
                  *labels,
                  percentage, 
                  train.get(), 
                  train_labels.get(),
                  test.get(),
                  test_labels.get());
      fl::logger->Message() << "Saving the labels of the new files "
        << references_labels_out << ", " << queries_labels_out << std::endl;
      ws_->Purge(train_labels->filename());
      ws_->Purge(test_labels->filename());
      ws_->Detach(train_labels->filename());
      ws_->Detach(test_labels->filename());

    }
    fl::logger->Message() <<"Saving the new reference file in " 
      << references_out <<std::endl;
    ws_->Purge(train->filename());
    fl::logger->Message() <<"Saving the new query file in " 
      << queries_out <<std::endl;
    ws_->Purge(test->filename());
    ws_->Detach(train->filename());
    ws_->Detach(test->filename());
    return ; 
  }
  if (task=="rqd") {
    if (percentage<0 || percentage>1) {
      fl::logger->Die() << "--percentage=" <<percentage<< " must be between 0 and 1 "
        "for this task";
    }
    if (queries_out=="") {
      fl::logger->Die() << "For this task(="<< task <<") you should provide "
        "--queries_out option";
    }
    if (references_out=="") {
      fl::logger->Die() << "For this task(="<< task <<") you should provide "
        "--references_out option";
    }
    
    if (labels_in!="") {
      if (references_labels_out=="") {
        fl::logger->Die() << "For this task(="<< task <<") you should provide "
        "--references_labels_out option, since you have provided labels_in(="
        <<labels_in<<")";             
      }
      if (queries_labels_out=="") {
       fl::logger->Die() << "For this task(="<< task <<") you should provide "
        "--queries_labels_out option, since you have provided labels_in(="
        <<labels_in<<")";
      }      
    }

    ws_->Attach(references_in,
        &references);
    fl::logger->Message() << "Loading reference data from " <<references_in
      <<std::endl;
    boost::shared_ptr<TableType> train;
    boost::shared_ptr<typename WorkSpaceType::template TableVector<index_t> > train_labels;
    boost::shared_ptr<TableType> test;
    boost::shared_ptr<typename WorkSpaceType::template TableVector<index_t> > test_labels;
    if (labels_in!="") {
      fl::logger->Warning() << "The program will not output new labels for the "
        <<"new reference and new query file. When we sample dimensions the labels "
        <<"will be the same for all the files"<<std::endl;
    }
    ws_->Attach(references_out, 
        references->dense_sizes(),
        references->sparse_sizes(),
        0,
        &train);
    ws_->Attach(queries_out,
       references->dense_sizes(),
       references->sparse_sizes(),
       0,
       &test);
    fl::logger->Message() << "Splitting the dimensions of the references file"
      <<std::endl;
    SplitDimensions(*references, percentage, train.get(), test.get());
    fl::logger->Message() << "Saving the new references file in " 
      << references_out<<std::endl;
    ws_->Purge(train->filename());
    fl::logger->Message() << "Saving the new queries file in " 
      << queries_out<<std::endl;
    ws_->Purge(test->filename());
    ws_->Detach(train->filename());
    ws_->Detach(test->filename());

    return ;
  }

  if (task=="rlp") {
    if (vm["references_out_prefix"].as<std::string>() == "") {
      fl::logger->Die() << "For this task(="<< task <<") you should provide "
        "--references_out prefix";
    }
    if (labels_in=="") {
      fl::logger->Warning() << "You didn't provide a --labels_in file, I will load the "
        "labels from the --references_in file"<<std::endl;
    }
    ws_->Attach(references_in,
        &references);
    fl::logger->Message() << "Loading reference data from " <<references_in
      <<std::endl;
    boost::shared_ptr<TableType> train;
    boost::shared_ptr<typename WorkSpaceType::template TableVector<index_t> > train_labels;
    boost::shared_ptr<TableType> test;
    boost::shared_ptr<typename WorkSpaceType::template TableVector<index_t> > test_labels;
    boost::shared_ptr<typename WorkSpaceType::template TableVector<index_t> > labels;
    if (labels_in!="") {
      fl::logger->Message() << "Loading refernece labels from " <<labels_in <<std::endl;
      ws_->Attach(labels_in, &labels);
    }
    fl::logger->Message() << "Splitting the references file in its classes" 
      <<std::endl;
    std::map<signed char, boost::shared_ptr<TableType> > new_tables;
    if (labels_in=="") {
     SplitLabeled(*references,
                  ws_,
                  vm["references_out_prefix"].as<std::string>(), 
                  &new_tables);
    } else {
      SplitLabeled(*references,
                   *labels,
                   ws_,
                   vm["references_out_prefix"].as<std::string>(), 
                   &new_tables);
    }   

    for(typename std::map<signed char, boost::shared_ptr<TableType> >::iterator it=new_tables.begin();
        it!=new_tables.end(); ++it) {
      
      fl::logger->Message() << "Saving the new references files in " 
          << it->second->filename() <<std::endl;
      ws_->Purge(it->second->filename());
      ws_->Detach(it->second->filename());
    }

    return ; 
  } 

  if (task=="rnp") {
    fl::logger->Die() << "task rnp " << fl::NOT_SUPPORTED_MESSAGE;
    return ;
  }
  if (task=="rnd") {
    fl::logger->Die() << "task rnd " << fl::NOT_SUPPORTED_MESSAGE;
    return ;
  }
  fl::logger->Die() << "Unknown task: " <<task;
  return ;
}


template<typename WorkSpaceType>
int TableUtil<boost::mpl::void_>::Run(
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
TableUtil<boost::mpl::void_>::Core<WorkSpaceType>::Core(
   WorkSpaceType *ws, const std::vector<std::string> &args) :
 ws_(ws), args_(args)  {}


}}

#endif
