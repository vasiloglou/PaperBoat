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

#ifndef FL_LITE_INCLUDE_MLPACK_MIXTURE_OF_EXPERTS_MOE_DEFS_H_
#define FL_LITE_INCLUDE_MLPACK_MIXTURE_OF_EXPERTS_MOE_DEFS_H_
#include "moe.h"
#include "regression_expert_dev.h"
#include "fastlib/util/string_utils.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/workspace/task.h"

template<typename TableType1>
template<class WorkSpaceType>
int fl::ml::Moe<boost::mpl::void_>::Core<TableType1>::Main(
    WorkSpaceType *ws,
    boost::program_options::variables_map &vm) {
  FL_SCOPED_LOG(Moe);
  if (vm.count("references_in")==0) {
    fl::logger->Die()<<"Option --references_in is required";
  }
  boost::shared_ptr<TableType1> references_table;
  ws->Attach(vm["references_in"].as<std::string>(), &references_table);
  fl::logger->Message()<<"Loaded references_in from ("<<vm["references_in"].as<std::string>()<<")"<<std::endl;
  int32 k_clusters=vm["k_clusters"].as<int32>();
  int32 iterations=vm["iterations"].as<int32>();
  int32 n_restarts=vm["n_restarts"].as<int32>();
  double error_tolerance=vm["error_tolerance"].as<double>();
  std::string expert=vm["expert"].as<std::string>();
  std::string expert_args_str=vm["expert_args"].as<std::string>();
  bool expert_log=vm["log_expert"].as<bool>();
  fl::StringReplace(&expert_args_str, ":", "=");
  std::vector<std::string> expert_args=SplitString(expert_args_str, ",");
  std::vector<std::string> final_expert_args=expert_args;
  std::string memberships_out=vm["memberships_out"].as<std::string>();
  std::string scores_out=vm["scores_out"].as<std::string>();
  std::vector<index_t> memberships;
  std::vector<double>  scores;
  std::string predefined_memberships_in; 
  boost::shared_ptr<typename WorkSpaceType::DefaultSparseIntTable_t> predefined_memberships_table;
  if (vm.count("predefined_memberships_in")) {
    predefined_memberships_in = 
      vm["predefined_memberships_in"].as<std::string>();
    ws->Attach(predefined_memberships_in, 
             &predefined_memberships_table);
  }

  if (fl::StringStartsWith(expert, "regression")!=std::string::npos) {
    typedef RegressionExpert<TableType1, WorkSpaceType> Expert_t;
    typedef Moe<Expert_t> Moe_t;
    Moe_t moe;
    if (predefined_memberships_in!="") {
      std::map<index_t, int32> predefined_memberships;
      typename WorkSpaceType::DefaultSparseIntTable_t::Point_t point;
      for(index_t i=0; i<predefined_memberships_table->n_entries(); ++i) {
        predefined_memberships_table->get(i, &point);
        for(typename WorkSpaceType::DefaultSparseIntTable_t::Point_t::iterator it=point.begin();
            it!=point.end(); ++it) {
          predefined_memberships[it.attribute()]=i;
        }   
      }
      moe.set_predefined_memberships(predefined_memberships);
    }
    moe.set_k_clusters(k_clusters);
    moe.set_iterations(iterations);
    moe.set_n_restarts(n_restarts);
    moe.set_error_tolerance(error_tolerance);
    moe.set_references(references_table);
    moe.set_expert_args(expert_args);
    moe.set_initial_clusters();
    moe.set_expert_log(expert_log);
    moe.Compute(&memberships,
                &scores);
  } else {
    fl::logger->Die()<<"Expert ("<<expert<<") is not supported";
  } 
  if (memberships_out!="") {
    fl::logger->Message()<<"Exporting memberships to "<<memberships_out<<std::endl;
    boost::shared_ptr<typename WorkSpaceType::UIntegerTable_t> memberships_table;
    ws->Attach(memberships_out,
        std::vector<index_t>(1,1),
        std::vector<index_t>(),
        references_table->n_entries(),
        &memberships_table);
    typename WorkSpaceType::UIntegerTable_t::Point_t point;
    for(index_t i=0; i<memberships.size(); ++i) {
      memberships_table->get(i, &point);
      point.set(0, memberships[i]);      
    }
    ws->Purge(memberships_out);
    ws->Detach(memberships_out);
  }

  if (scores_out!="") {
    fl::logger->Message()<<"Exporting scores to "<<scores_out<<std::endl;
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> scores_table;
     ws->Attach(scores_out,
        std::vector<index_t>(1,1),
        std::vector<index_t>(),
        k_clusters,
        &scores_table);
    typename WorkSpaceType::DefaultTable_t::Point_t point;
    for(index_t i=0; i<k_clusters; ++i) {
      scores_table->get(i, &point);
      point.set(0, scores[i]);      
    }
    ws->Purge(scores_out);
    ws->Detach(scores_out);
  }
  // Running the final experts 
  {
    fl::logger->Message()<<"Running the final pass on experts"<<std::endl;
    std::string final_expert_args_str=vm["final_expert_args"].as<std::string>();
    fl::StringReplace(&final_expert_args_str, ":", "=");
    std::vector<std::string> expert_outputs = SplitString(final_expert_args_str, ",");
    std::vector<std::string> expert_tokens=fl::SplitString(expert, ":");
    std::vector<boost::shared_ptr<TableType1> > reference_tables(k_clusters);
    for(int32 i=0; i<k_clusters; ++i) {
      reference_tables[i].reset(new TableType1());
      ws->Attach(ws->GiveFilenameFromSequence("reference", i),
          references_table->dense_sizes(),
          references_table->sparse_sizes(), 
          0,
          &reference_tables[i]);      
      reference_tables[i]->labels()=references_table->labels();
    }
    typename TableType1::Point_t point1;
    for(size_t i=0; i<memberships.size(); ++i) {
      references_table->get(i, &point1);
      reference_tables[memberships[i]]->push_back(point1);    
    }   
    for(int32 i=0; i<k_clusters; ++i) {
      ws->Purge(ws->GiveFilenameFromSequence("reference", i));
      ws->Detach(ws->GiveFilenameFromSequence("reference", i));
    } 
    for(int32 i=0; i<k_clusters; ++i) {
      std::vector<std::string> local_args=expert_args;
      local_args.push_back(fl::StitchStrings("--references_in=", 
            ws->GiveFilenameFromSequence("reference",
            i)));
      local_args.push_back("--check_columns=1");
      // This needs to change
      for(int32 j=0; j<expert_outputs.size(); ++j) {
        std::string arg(expert_outputs[j]);
        local_args.push_back(ws->GiveFilenameFromSequence(arg, i));
      }
      //std::cout<<"args"<<std::endl;
      //for(std::vector<std::string>::iterator it=local_args.begin();
      //    it!=local_args.end(); ++it) {
      //  std::cout<<*it<<", ";
      //}
      //std::cout<<std::endl;

      if (expert_log==false) {
        logger->SuspendLogging();
      }
      fl::ml::LinearRegression<boost::mpl::void_>::Main<WorkSpaceType, 
        typename WorkSpaceType::Branch_t>(ws, local_args);
      if (expert_log==false) {
        logger->ResumeLogging();
      }
      ws->ExportAllTables(local_args);
    }
  }
  fl::logger->Message()<<"Moe finished"<<std::endl;
  return 1;
}

template<typename WorkSpaceType, typename BranchType>
int fl::ml::Moe<boost::mpl::void_>::Main(
    WorkSpaceType *ws,
    const std::vector<std::string> &args) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::string>(),
    "REQUIRED in the --run_mode=train, file containing data to be clustered."
  )(
    "k_clusters",
    boost::program_options::value<int32>()->default_value(5),
    "REQUIRED number of clusters for the mixure of experts"  
  )(
    "error_tolerance",
    boost::program_options::value<double>()->default_value(1e-6),
    "OPTIONAL if the percentage change in the objective is less than that "
    "then the process terminates"
  )(
    "iterations",
    boost::program_options::value<int32>()->default_value(100),
    "OPTIONAL number of iterations "
  )(
    "n_restarts",
    boost::program_options::value<int32>()->default_value(1),
    "OPTIONAL number of restarts of the MOE algorthms. Since it get stuck "
    "to local optima it makes sense to restart it several times and pick "
    "the best result" 
  )(
    "predefined_memberships_in",
    boost::program_options::value<std::string>(),
    "OPTIONAL in some cases there are hard constraints about points being "
    " in the same cluster. This argument refers to a sparse int table "
    " that has m rows (buckets of points that have to cluster together). "
    "The dimensionality of "
    " each point is N (total number of points) and the nonzero entries of "
    " each point indicate reference points that have to be in the same "
    " cluster."
  )(
    "memberships_out",
    boost::program_options::value<std::string>()->default_value(""),
    "file to export the memberships"
  )(
    "scores_out",
    boost::program_options::value<std::string>()->default_value(""),
    "file to export the scores of the clustering model"
  )(
    "expert",
    boost::program_options::value<std::string>()->default_value("regression:1"),
    "the only expert supported for the moment is regression:x, "
    "x is the regressor of which we try to increase the confidence, "
    "if x is zero then we are trying to minimize the mse of the expert"
  )(
    "expert_args",
    boost::program_options::value<std::string>()->default_value(""),
    "a comma separated list witht the arguments for the expert"  
  )(
    "log_expert",
    boost::program_options::value<bool>()->default_value(false),
    "Experts do extensive logging which might be annoying. If you need to see "
    "what is going on you can set this true"
  )(
    "final_expert_args",
    boost::program_options::value<std::string>()->default_value(""),
    "arguments for the final experts. The final experts will use the expert "
    "args for the parameters and the final experts args will have the prefixes "
    "of the outputs" 
   );

  std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args, "");
  boost::program_options::variables_map vm;
  boost::program_options::command_line_parser clp(args1);
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
  catch ( const boost::program_options::error &e) {
    fl::logger->Die() << e.what();
  } 
  boost::program_options::notify(vm);
  if (vm.count("help")) {
    fl::logger->Message() << fl::DISCLAIMER << "\n";
    fl::logger->Message() << desc << "\n";
    return 1;
  }
 
  return BranchType::template BranchOnTable<Moe<boost::mpl::void_>, WorkSpaceType>(ws, vm);
 
}

template<typename WorkSpaceType>
void fl::ml::Moe<boost::mpl::void_>::Run(
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

#endif
