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

#ifndef PAPERBOAT_MLPACK_APPROXIMATE_MEAN_SHIFT_DEFS_H_
#define PAPERBOAT_MLPACK_APPROXIMATE_MEAN_SHIFT_DEFS_H_
#include "boost/program_options.hpp"
#include "approximate_meanshift.h"
#include "mlpack/graph_diffuser/graph_diffuser.h"
#include "mlpack/kde/kde.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/workspace/based_on_table_run.h"
#include "fastlib/util/string_utils.h"

namespace fl { namespace ml {

  template<typename WorkSpaceType>
  template<typename TableType>
  void ApproximateMeanShift<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
    FL_SCOPED_LOG(ApproxMeanShift);
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "references_in",
      boost::program_options::value<std::string>(),
      "the reference data "
    )(
      "is_references_in_a_graph",
      boost::program_options::value<bool>()->default_value(false),
      "the --references_in data can either be raw data or a graph. If it "
      "is a graph then you should set this flag true"
    )(
      "densities_in",
      boost::program_options::value<std::string>(),
      "if the references are given in as a graph then you have to provide the densities "
      "so that approximate meanshift can be computed"
    )(
      "max_iterations",
      boost::program_options::value<int32>()->default_value(100),
      "number of iterations to run the meanshift"
    )(
      "full_logging",
      boost::program_options::value<bool>()->default_value(false),
      "This program runs some subprograms that they also log. If you don't "
      "want this extra logging set the --full_logging=false" 
    )(
      "clusters_out",
      boost::program_options::value<std::string>(),
      "the id of the points that are the cluster centers"  
    )(
      "memberships_out",
      boost::program_options::value<std::string>(),
      "the memberships of each point to the cluster. The cluster is identified "
      "by the id of the centroid"  
    )(
      "cluster_statistics_out",
      boost::program_options::value<std::string>(),
      "the number of points per cluster"  
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
    fl::ws::RequiredArgs(vm, "cluster_statistics_out", "clusters_out");
    //fl::ws::RequiredArgValues(vm, "is_references_in_a_graph:1", "densities_in:*");
    // we need to fix this, the RequirdArgValues must be templatized for 
    // the value types
    //fl::ws::RequiredArgValues(vm, "densities_in:*", "is_references_in_a_graph:1");
    std::map<std::string, std::string> graph_map;
    std::vector<std::string> graph_args=fl::ws::MakeArgsFromPrefix(args_, "graphd");
    graph_map=fl::ws::GetArgumentPairs(graph_args);
    std::string graph_file;
    if (graph_map.count("graph_out")>0) {
      graph_file=graph_map["graph_out"];
    } else {
      graph_file=ws_->GiveTempVarName();
    }
    if (vm["is_references_in_a_graph"].as<bool>()==false) {
      fl::logger->Message()<<"Building the graph"<<std::endl;
      if (vm["full_logging"].as<bool>()==false) {
        fl::logger->SuspendLogging();
      }
      std::map<std::string, std::string> graph_map;
      graph_map=fl::ws::GetArgumentPairs(graph_args);
      if (graph_map.count("--references_in")) {
        fl::logger->Die()<<"You can't set --graphd:references_in";
      }
      if (graph_map.count("--run_diffusion")==0) {
        graph_args.push_back("--run_diffusion=0");
      }
      if (graph_map.count("--connect_nodes")==0) {
        graph_args.push_back("--connect_nodes=snn");
      }
      if (graph_map.count("--graph_out")==0) {
        graph_args.push_back("--graph_out="+graph_file);
      } else {
        fl::logger->Warning()<<"Setting --graphd:graph_out is pointless and dangerous";
        graph_file=graph_map["--graph_out"];
      }
      graph_args.push_back("--references_in="+vm["references_in"].as<std::string>());
      fl::ml::GraphDiffuser<boost::mpl::void_>::Run(ws_, graph_args);
      if (vm["full_logging"].as<bool>()==false) {
        fl::logger->ResumeLogging();
      }
      fl::logger->Message()<<"Finished building the graph"<<std::endl;
    }
    
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> densities_table;
    std::string densities_file;
    if (vm.count("densities_in")==0) {
      std::vector<std::string> kde_args=fl::ws::MakeArgsFromPrefix(args_, "kde");
      std::map<std::string, std::string> kde_map;
      kde_map=fl::ws::GetArgumentPairs(kde_args);
      if (kde_map.count("densities_out")>0) {
        densities_file=kde_map["densities_out"];
      } else {
        densities_file=ws_->GiveTempVarName();
        kde_args.push_back("--densities_out="+densities_file);
      }
      if (kde_map.count("--references_in")) {
        fl::logger->Die()<<"You can't set --kde:references_in";
      }
      kde_args.push_back("--references_in="+vm["references_in"].as<std::string>());
      fl::logger->Message()<<"Computing KDE on the graph nodes"<<std::endl;
      if (vm["full_logging"].as<bool>()==false) {
        fl::logger->SuspendLogging();
      }
      ws_->IndexAllReferencesQueries(&kde_args);
      Kde<boost::mpl::void_>::Run(ws_, kde_args);
      if (vm["full_logging"].as<bool>()==false) {
        fl::logger->ResumeLogging();
      }
      fl::logger->Message()<<"Finished Computing KDE"<<std::endl;
    } else {
      densities_file=vm["densities_in"].as<std::string>();  
    }
    ws_->Attach(densities_file, &densities_table);

    fl::logger->Message()<<"Started computing clustering"<<std::endl;
    typedef typename WorkSpaceType::DefaultTable_t::Point_t DPoint_t;
    boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t> graph_table;
    typedef typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t GPoint_t;
    
    if (vm["is_references_in_a_graph"].as<bool>()==true) {
      try{
        ws_->template TryToAttach<typename WorkSpaceType::DefaultSparseDoubleTable_t>(vm["references_in"].as<std::string>());
      }
      catch(...) {
        fl::logger->Die()<<"Only sparse double tables are supported currently as graphs";
      }
      ws_->Attach(vm["references_in"].as<std::string>(), &graph_table);
    } else {
      ws_->Attach(graph_file, &graph_table);
    }
    int32 max_iterations=vm["max_iterations"].as<int32>();
    GPoint_t gpoint;
    std::vector<std::pair<index_t, double> > result(graph_table->n_entries());
    for(index_t i=0; i<densities_table->n_entries(); ++i) {
      result[i].first=i;
      result[i].second=densities_table->get(i, int64(0));
    }

    fl::logger->Message()<<"Iterating over graph"<<std::endl;
    for(int32 iteration=0; iteration<max_iterations; ++iteration) {
      bool change=false;
      for(index_t i=0; i<graph_table->n_entries(); ++i) {
        graph_table->get(i, &gpoint);     
        for(typename GPoint_t::iterator it=gpoint.begin(); it!=gpoint.end(); ++it) {
          // this can be computed offline and save some time
          if (densities_table->get(i, int64(0))>densities_table->get(it.attribute(), int64(0))) {
            continue;
          }
          const double neighbor_density=result[it.attribute()].second;
          if (result[i].second<neighbor_density) {
            result[i]=result[it.attribute()];
            change=true;
          }            
        }
      }
      if (change==false) {
        fl::logger->Message()<<"Algorithm converged"<<std::endl;
        break;
      }
      fl::logger->Message()<<"Finished iteration="<<iteration<<std::endl;
    }
    fl::logger->Message()<<"Finished computing clusters"<<std::endl; 
    boost::shared_ptr<typename WorkSpaceType::IntegerTable_t> memberships_table;
    if (vm.count("memberships_out")>0) {
      fl::logger->Message()<<"Exporting memberships"<<std::endl;
      ws_->Attach(vm["memberships_out"].as<std::string>(),
          std::vector<index_t>(1, 1),
          std::vector<index_t>(),
          result.size(),
          &memberships_table);
      for(size_t i=0; i<result.size(); ++i) {
        memberships_table->set(i, 0, result[i].first);
      }
      ws_->Purge(memberships_table->filename());
      ws_->Detach(memberships_table->filename());
      fl::logger->Message()<<"Finished exporting memeberships"<<std::endl;
    }

    std::map<index_t, index_t> clusters;
    if (vm.count("clusters_out")>0) {
      fl::logger->Message()<<"Exporting clusters"<<std::endl;
      boost::shared_ptr<typename WorkSpaceType::IntegerTable_t> clusters_table;
      for(size_t i=0; i<result.size(); ++i) {
        clusters[result[i].first]+=1;
      }
      fl::logger->Message()<<"Found "<< clusters.size() <<" clusters"<<std::endl;
      ws_->Attach(vm["clusters_out"].as<std::string>(),
          std::vector<index_t>(1, 1),
          std::vector<index_t>(),
          clusters.size(),     
          &clusters_table);
      int32 counter=0;
      for(std::map<index_t, index_t>::iterator it=clusters.begin(); 
          it!=clusters.end(); ++it) {
        clusters_table->set(counter, 0, it->first);
        ++counter;
      }
      ws_->Purge(clusters_table->filename());
      ws_->Detach(clusters_table->filename());
      fl::logger->Message()<<"Finished exporting clusters"<<std::endl;
    }
    if(vm.count("cluster_statistics_out")) {
      boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> stats_table;
      ws_->Attach(vm["cluster_statistics_out"].as<std::string>(),
          std::vector<index_t>(1, 1),
          std::vector<index_t>(), 
          clusters.size(),
          &stats_table);
      index_t counter=0;
      for(std::map<index_t, index_t>::iterator it=clusters.begin(); 
          it!=clusters.end(); ++it) {
        stats_table->set(counter, 0, it->second);
      }
      ws_->Purge(stats_table->filename());
      ws_->Purge(stats_table->filename());
    }
  }

  template<typename WorkSpaceType>
  int ApproximateMeanShift<boost::mpl::void_>::Run(
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
  ApproximateMeanShift<boost::mpl::void_>::Core<WorkSpaceType>::Core(
     WorkSpaceType *ws, const std::vector<std::string> &args) :
   ws_(ws), args_(args)  {}



}}
#endif
