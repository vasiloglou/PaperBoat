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

#ifndef PAPERBOAT_INCLUDE_MLPACK_HMM_DEFS_H_
#define PAPERBOAT_INCLUDE_MLPACK_HMM_DEFS_H_
#include "hmm.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/util/string_utils.h"
#include "discrete_distribution_defs.h"
#include "kde_distribution_defs.h"
#include "gmm_distribution_defs.h"

namespace fl { namespace ml {

  template<typename HmmArgsType>
  void Hmm<HmmArgsType>::Init(
      index_t n_states,
      const std::vector<std::string> &distribution_args,
      std::vector<
          boost::shared_ptr<Table_t> 
      > &references) {
    transition_matrix_.reset(new TransitionTable_t());
    // we need to try and see which initialization works
    if (TransitionTable_t::Dataset_t::IsDenseOnly_t::value) {
      transition_matrix_->Init("",
          std::vector<index_t>(1, n_states),
          std::vector<index_t>(),
          n_states);
    } else {
      transition_matrix_->Init("",
            std::vector<index_t>(),
            std::vector<index_t>(1, n_states),
            n_states);
    }
    
    distributions_.resize(n_states); 
    // each distribution is initialized and an id is given
    for(index_t i=0; i<distributions_.size(); ++i) {
      distributions_[i].Init(
          distribution_args,
          i,
          references[0]->dense_sizes(), 
          references[0]->sparse_sizes());
    }
    n_states_=n_states;
    // do an initial training of the distributions
    typename Table_t::Point_t point;
    index_t counter=0;
    for(size_t i=0; i<references.size(); ++i) {
      for(index_t j=0; j<references[i]->n_entries(); ++j) {
        references[i]->get(j, &point);
        index_t k=counter % distributions_.size();
        distributions_[k].AddPoint(point);
        counter++;
      }
    }
    for(index_t i=0; i<distributions_.size(); ++i) {
      distributions_[i].Train();
    }
  }

  template<typename HmmArgsType>
  void Hmm<HmmArgsType>::Train(
      int32 iterations,
      std::vector<
        boost::shared_ptr<Table_t> 
      > &references) {
    fl::logger->Message()<<"Started Training"<<std::endl;
    std::cout<<"distribution_count=";
    for(index_t i=0; i<distributions_.size(); ++i) {
      std::cout<<"("<<i<<","<<distributions_[i].count()<<"), ";
    }
    std::cout<<std::endl;

    for(int32 it=0; it<iterations; ++it) {
      std::vector<StateSequence_t> state_sequences(references.size());
      fl::logger->SuspendLogging();
      for(size_t i=0; i<references.size(); ++i) {
        state_sequences[i]=
          ComputeMostLikelyStateSequence(*references[i]);
      }
      Update(references, state_sequences);
      fl::logger->ResumeLogging();
      fl::logger->Message()<<"Iteration="<<it<<std::endl;
    }
  }

  template<typename HmmArgsType>
  double Hmm<HmmArgsType>::Eval(Table_t &table) {
    index_t total_time=table.n_entries();
    std::vector<std::vector<double> > delta(n_states_);
    for(size_t i=0; i<delta.size(); ++i) {
      delta[i].resize(total_time);
      std::fill(delta[i].begin(), 
          delta[i].end(), 
          -std::numeric_limits<double>::max());
    }
    Point_t point;
    table.get(0, &point);
    for(std::map<index_t, double>::iterator it=initial_probabilities_.begin(); 
        it!=initial_probabilities.end(); ++it) {
      delta[it->first][0]=it->second+distributions_[it->first].LogDensity(point);
    }
    for(index_t t=1; t<total_time; ++t) {
      table.get(t, &point);
      for(int32 i=0; i<n_states_; ++i) {
        double emission_likelihood=distributions_[i].LogDensity(point);
        // this is wrong we need to use an iterator over the transition table
        typename TransitionTable_t::Point_t tpoint;
        transition_matrix_->get(i, &tpoint);
        for(typename TransitionTable_t::Point_t::iterator it=tpoint.begin();
            it!=tpoint.end(); ++it) {
          double new_delta=delta[it.attribute()][t-1]
          //warning did you check if transition matrix is in loglikelihoods?
          +it.value()
          +emission_likelihood;
          if (new_delta>delta[it.attribute()][t]) {
            delta[it.attribute()][t]=new_delta;
          }
        }
      }
    }
    //scan the final time to find the best  sequence
    index_t best_state=-1;
    index_t best_value=-std::numeric_limits<double>::max();
    for(size_t i=0; i<n_states_; ++i) {
      if (delta[i][total_time]>best_value) {
        best_state=i;
        best_value=delta[i][total_time];
      }
    }
    return best_value;
  }

  template<typename HmmArgsType>
  void Hmm<HmmArgsType>::Update(
               std::vector<
                 boost::shared_ptr<
                   Table_t
                 > 
               > &references, 
               std::vector<StateSequence_t> &state_sequences) {
    initial_probabilities_.clear();
    // comment. It is possible that if we do not use the SetAll(0.0)
    // and just zero out all the nonzero elements, we can use 
    // the same code for sparse and dense
    transition_matrix_->SetAll(0.0);
    // check all the first states to update the initial probabilities
    for(size_t i=0; i<state_sequences.size(); ++i) {
      if (initial_probabilities_.count(state_sequences[i][0].first)) {
        initial_probabilities_[state_sequences[i][0].first]=0;
      }
      initial_probabilities_[state_sequences[i][0].first]+=1;
    }
    // normalize and log
    std::cout<<"initial_probabilities=";
    for(std::map<index_t, double>::iterator it=initial_probabilities_.begin();
        it!=initial_probabilities_.end(); ++it) {
      it->second=log(it->second / initial_probabilities_.size());
      std::cout<<"("<<it->first<<","<<it->second<<"),";
    }
    std::cout<<std::endl;
    // count the total number of states
    std::vector<index_t> state_sums(n_states_);
    std::fill(state_sums.begin(), 
        state_sums.end(), 0);
    // reset the distributions
    for(size_t i=0; i<distributions_.size(); ++i) {
      distributions_[i].ResetData();
    }
    for(size_t i=0; i<state_sequences.size(); ++i) {
      Table_t *table=references[i].get();
      typename Table_t::Point_t rpoint;
      index_t state1=0;
      index_t state2=0;
      for(size_t j=0; j<state_sequences[i].size()-1; ++j) {
        state1=state_sequences[i][j].first;
        state2=state_sequences[i][j+1].first;
        transition_matrix_->UpdatePlus(
            state1, 
            state2, 
            1.0);   
        state_sums[state1]+=1;
        table->get(j, &rpoint);        
        distributions_[state1].AddPoint(rpoint);
      }
      table->get(state_sequences[i].size()-1, &rpoint);
      distributions_[state2].AddPoint(rpoint);
      state_sums[state2]+=1;
    }
    typename TransitionTable_t::Point_t point;
    for(index_t i=0; i<transition_matrix_->n_entries(); ++i) {
      transition_matrix_->get(i, &point);
      LogTransform transform(state_sums[i]);
      point.Transform(transform);
    }     
    // now train the distributions
    for(size_t i=0; i<distributions_.size(); ++i) {
      distributions_[i].Train();
    }
    for(index_t i=0; i<transition_matrix_->n_entries(); ++i) {
      typename TransitionTable_t::Point_t tpoint;
      transition_matrix_->get(i, &tpoint);
      for(typename TransitionTable_t::Point_t::iterator it=tpoint.begin();
          it!=tpoint.end(); ++it) {
        std::cout<<"("<<i<<","<<it.attribute()<<","<<it.value()<<"),";
      }
      std::cout<<std::endl;
    }
    std::cout<<"distribution_count=";
    for(index_t i=0; i<distributions_.size(); ++i) {
      std::cout<<"("<<i<<","<<distributions_[i].count()<<"), ";
    }
    std::cout<<std::endl;
  }

  template<typename HmmArgsType>
  typename Hmm<HmmArgsType>::StateSequence_t 
      Hmm<HmmArgsType>::ComputeMostLikelyStateSequence(Table_t &table) {
    
    index_t total_time=table.n_entries();
    std::vector<StateSequence_t> delta(n_states_);
    for(size_t i=0; i<delta.size(); ++i) {
      delta[i].resize(total_time);
      std::fill(delta[i].begin(), 
          delta[i].end(), 
          std::make_pair(0, -std::numeric_limits<double>::max()));
    }
    Point_t point;
    table.get(0, &point);
    for(std::map<index_t, double>::iterator it=initial_probabilities_.begin(); 
        it!=initial_probabilities_.end(); ++it) {
      delta[it->first][0].second=it->second+distributions_[it->first].LogDensity(point);
      delta[it->first][0].first=0;
      //std::cout<<"("<<it->first<<","<<it->second<<","<<distributions_[it->first].LogDensity(point)<<"), ";
    }
    //std::cout<<std::endl;

    for(index_t t=1; t<total_time; ++t) {
      table.get(t, &point);
      //std::cout<<"emission_likelihood=";
      for(int32 i=0; i<n_states_; ++i) {
        double emission_likelihood=distributions_[i].LogDensity(point);
        //std::cout<<emission_likelihood<<", ";
        typename TransitionTable_t::Point_t tpoint;
        transition_matrix_->get(i, &tpoint);
        for(typename TransitionTable_t::Point_t::iterator it=tpoint.begin();
            it!=tpoint.end(); ++it) {
          int32 j=it.attribute();
          double log_transition_prob=it.value();
          double new_delta=delta[j][t-1].second
          //warning did you check if transition matrix is in loglikelihoods?
          +log_transition_prob
          +emission_likelihood;
          //std::cout<<log_transition_prob<<" "<<emission_likelihood<<" "<< delta[j][t].second<<std::endl;
          if (new_delta>delta[j][t].second) {
            delta[j][t].first=i;
            delta[j][t].second=new_delta;
          }
          //std::cout<<"*"<<delta[j][t].first<<" "<<delta[j][t].second<<std::endl;
        }
      }
      //std::cout<<std::endl;
    }
    //scan the final time to find the best  sequence 
    index_t best_state=-1;
    double best_value=-std::numeric_limits<double>::max();
    for(size_t i=0; i<n_states_; ++i) {
      if (delta[i][total_time-1].second>best_value) {
        best_state=i;
        best_value=delta[i][total_time-1].second;
      }
    }
    if (best_state==-1) {
      for(size_t i=0; i<n_states_; ++i) {
        std::cout<<delta[i][total_time-1].first<<","<<delta[i][total_time-1].second<<" ";
      }
      std::cout<<std::endl;
      for(std::map<index_t, double>::iterator it=initial_probabilities_.begin(); 
          it!=initial_probabilities_.end(); ++it) {
        std::cout<<"("<<it->first<<","<<it->second<<"),";
      }
      std::cout<<std::endl;
    }
    DEBUG_ASSERT_MSG(best_state!=-1, "Something is going wrong in the decoding");
    // now backtrack to get the best path
    StateSequence_t best_path(total_time);
    best_path[total_time-1].first=best_state;
    best_path[total_time-1].second=best_value;
    for(index_t t=total_time-2; t>=0; --t) {
      best_path[t].first = delta[best_state][t+1].first;
      best_path[t].second = delta[best_state][t+1].second;
      best_state=delta[best_state][t+1].first;
      best_value=delta[best_state][t+1].second;
    }
    return best_path;
  }

  template<typename HmmArgsType>
  boost::shared_ptr<typename HmmArgsType::TransitionTableType> 
  Hmm<HmmArgsType>::transition_matrix() {
    return transition_matrix_;       
  }

  template<typename HmmArgsType>
  void Hmm<HmmArgsType>::set_transition_matrix(
      boost::shared_ptr<TransitionTable_t> transition_matrix) {
    transition_matrix_=transition_matrix;       
  }
 
  template<typename HmmArgsType> 
  std::vector<typename Hmm<HmmArgsType>::Distribution_t>
  &Hmm<HmmArgsType>::distributions() {
    return distributions_;
  }

  template<typename HmmArgsType>
  std::map<index_t, double> &Hmm<HmmArgsType>::initial_probabilities() {
    return initial_probabilities_;    
  }

  template<typename HmmArgsType>
  template<typename PointType>
  void Hmm<HmmArgsType>::set_initial_probabilities(PointType &point) {
    initial_probabilities_.clear();
    for(typename PointType::iterator it=point.begin();
        it!=point.end(); ++it) {
      initial_probabilities_[it.attribute()]=it.value();
    }
  }

  template<typename HmmArgsType>
  int32 Hmm<HmmArgsType>::n_states() {
    return n_states_;    
  }

  template<typename WorkSpaceType1>
  template<typename TableType>
  void Hmm<boost::mpl::void_>::Core<WorkSpaceType1>::operator()(
      TableType&) {
    FL_SCOPED_LOG(Hmm);
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "references_in",
      boost::program_options::value<std::string>(),
      "the reference data comma separated. Each file is a time series"
      " eg --references_in=myfile,yourfile,hisfile"
    )(
      "references_prefix_in",
      boost::program_options::value<std::string>(),
      "the reference data prefix. The program will look for a file "
      "sequence with this prefix and a number as a suffix, ranging from "
      "0 to --references_num_in. For example:\n"
      "--references_prefix_in=file --references_num_in=13 will look for "
      " file0 file1 file2 ... file12"
    )(
      "references_num_in",
      boost::program_options::value<int32>(),
      "number of references file with the prefix defined above"
    )(
      "task",
      boost::program_options::value<std::string>(),
      "valid tasks are:\n"
      "  train   :\n"
      "  eval    :\n"
      "  generate:\n"
    )(
      "transition_matrix_type",
      boost::program_options::value<std::string>(),
      "This option must be used only when --task=train, it defines "
      "the structure of the transition matrix. Available options are:\n"
      "feedforward: Only the upper triangle of the matrix can have nonzero values\n"
      "full       : All transitions are allowed"
    )(
      "n_states",
      boost::program_options::value<int32>(),
      "number of states for the transition matrix"
    )(
      "transition_matrix_in",
      boost::program_options::value<std::string>(),
      "the transition matrix to be imported for the model"  
    )(
      "iterations",
      boost::program_options::value<int32>()->default_value(10),
      "Number of iterations for running the training loop" 
    )(
      "transition_matrix_out",
      boost::program_options::value<std::string>(),
      "the transition matrix to be exported"  
    )(
      "initial_probabilities_in",
      boost::program_options::value<std::string>(),
      "table that contains the probabilities of each state "
      "being the initial state. Use this option to import them "
      "when working in eval mode"  
    )(
      "initial_probabilities_out",
      boost::program_options::value<std::string>(),
      "table that contains the probabilities of each state "
      "being the initial state. Use this option to export them "
      "when working in train mode" 
    )(
      "distribution_type",
      boost::program_options::value<std::string>()->default_value("kde"),
      "The distributions to be used for the states. Available options are:\n"
      "discrete: the references are one dimensional discrete data\n"
      "kde     : uses kernel density estimation\n"
      "gmm     : used gaussian mixture models"     
    )(
      "export_distributions_prefix",
      boost::program_options::value<std::string>(),
      "This prefix will contain all tha arguments the distribution needs"
      "to export the distribution"
    )(
      "import_distributions_prefix",
      boost::program_options::value<std::string>(),
      "This prefix will contain all the arguments the distribution needs to be imported" 
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
    fl::ws::RequiredOrArgs(vm, "references_in,references_prefix_in");
    fl::ws::RequiredArgs(vm, "task");
    std::vector<index_t> dense_sizes;
    std::vector<index_t> sparse_sizes; 
    boost::shared_ptr<typename WorkSpaceType1::DefaultTable_t> full_transition_matrix;
    boost::shared_ptr<typename WorkSpaceType1::DefaultSparseDoubleTable_t> ff_transition_matrix;
    int32 n_states=0;

    if (vm.count("transition_matrix_in")==true) {
      if (vm.count("transition_matrix_type")==true) {
        if (vm["transition_matrix_type"].as<std::string>()=="full") {
          ws_->Attach(vm["transition_matrix_in"].as<std::string>(),
              &full_transition_matrix);
        } else {
          if (vm["transition_matrix_type"].as<std::string>()=="feedforward") {
            ws_->Attach(vm["transition_matrix_in"].as<std::string>(),
                &ff_transition_matrix);
          }
        }
      } else {
        bool success=false;
        try {
          ws_->Attach(vm["transition_matrix_in"].as<std::string>(),
              &full_transition_matrix);
          success=true;
        } 
        catch(...) {           
        }
        if (success==false) {
          try {
            ws_->Attach(vm["transition_matrix_in"].as<std::string>(),
                &full_transition_matrix);
            success=true;
          }
          catch(...) {
            
          }
        }
        if (success==false) {
          fl::logger->Die()<<"Failed to attach --transition_matrix_in "
            "something is wrong with the type. It can be either "
              "dense double or sparse double";
        }
      }
    } else {
      if (vm.count("transition_matrix_type")==true) {
        if (vm.count("n_states")==true) {
          n_states=vm["n_states"].as<int32>();
        } else {
          fl::logger->Die()<<"If you do not provide --transition_matrix_in"
              "you need to set --n_states"; 
        }
        if (vm["transition_matrix_type"].as<std::string>()=="full") {
          ws_->Attach(ws_->GiveTempVarName(),
              std::vector<index_t>(1, n_states),
              std::vector<index_t>(),
              n_states,
              &full_transition_matrix);
          typename WorkSpaceType1::DefaultTable_t::Point_t point;
          for(index_t i=0; i<full_transition_matrix->n_entries(); ++i) {
            full_transition_matrix->get(i, &point); 
            point.SetRandom(0, 1);
          }
          ws_->Purge(full_transition_matrix->filename());
          ws_->Detach(full_transition_matrix->filename());
        } else {
          if (vm["transition_matrix_type"].as<std::string>()=="feedforward") {
            ws_->Attach(ws_->GiveTempVarName(),
                std::vector<index_t>(),
                std::vector<index_t>(1, n_states),
                n_states,
                &ff_transition_matrix);
          typename WorkSpaceType1::DefaultSparseDoubleTable_t::Point_t point;
          for(index_t i=0; i<ff_transition_matrix->n_entries(); ++i) {
            ff_transition_matrix->get(i, &point); 
              point.set(i, 1.0);
              point.set(std::min(i+1, 
                    ff_transition_matrix->n_entries()-1), 3.0);
          }
            ws_->Purge(ff_transition_matrix->filename());
            ws_->Detach(ff_transition_matrix->filename());
          }
        }
      } else {
        fl::logger->Die()<<"You have to define "
          "the --transition_matrix_type";
      }   
    }
    
    index_t transition_matrix_rank=0;
    if (full_transition_matrix.get()!=NULL) {
      transition_matrix_rank=full_transition_matrix->n_entries();
      if (full_transition_matrix->n_entries()!=full_transition_matrix->n_attributes()) {
        fl::logger->Die()<<"Transition matrix must be square";
      }
    } else {
      transition_matrix_rank=ff_transition_matrix->n_entries();
      if (ff_transition_matrix->n_entries()!=ff_transition_matrix->n_attributes()) {
        fl::logger->Die()<<"Transition matrix must be square";
      }
    }
    
    if (vm.count("n_states")) {
      n_states=vm["n_states"].as<int32>();
      if (n_states!=transition_matrix_rank && transition_matrix_rank==0) {
        fl::logger->Die()<<"--n_states must agree with the dimensions of "
          "--transition_matrix_in if --transition_matrix_in is set";
      }
    } else  {
      if (transition_matrix_rank==0) {
        fl::logger->Die()<<"either --n_states or --transition_matrix_in "
            "must be set (or both can be set but the dimensions must agree)";
      }
      n_states=transition_matrix_rank;
    }
    boost::shared_ptr<typename WorkSpaceType1::DefaultSparseDoubleTable_t> initial_probabilities;
    typename WorkSpaceType1::DefaultSparseDoubleTable_t::Point_t
          initial_probabilities_point;

    if (vm.count("initial_probabilities_in")) {
      ws_->Attach(vm["initial_probabilities_in"].as<std::string>(), 
          &initial_probabilities);
      initial_probabilities->get(0, &initial_probabilities_point);
    } else {
      ws_->Attach(ws_->GiveTempVarName(), 
          std::vector<index_t>(),
          std::vector<index_t>(1, n_states), 
          1,
          &initial_probabilities);
      initial_probabilities->get(0, &initial_probabilities_point);
      initial_probabilities_point.SetRandom(1e-10, 1,0.0);
      for(typename WorkSpaceType1::DefaultSparseDoubleTable_t::Point_t::iterator it=
          initial_probabilities_point.begin(); it!=initial_probabilities_point.end(); ++it) {
        it.value()=log(it.value());
      }
      ws_->Purge(initial_probabilities->filename());
      ws_->Detach(initial_probabilities->filename());
    }

    if (vm["task"].as<std::string>()=="train") {
      int32 iterations=vm["iterations"].as<int32>();
      std::vector<boost::shared_ptr<TableType> > references;
      if (vm.count("references_in")==true) {
        std::vector<std::string> reference_names=fl::SplitString(
            vm["references_in"].as<std::string>(),",");
        references.resize(reference_names.size());
        for(size_t i=0; i<reference_names.size(); ++i) {
          ws_->Attach(reference_names[i], &references[i]);
        } 
      } else {
        std::string reference_prefix=vm["references_prefix_in"].as<std::string>();
        int32 reference_num=vm["references_num_in"].as<int32>();
        references.resize(reference_num);
        fl::logger->Message()<<"Loaded references"<<std::endl;
        references.resize(reference_num);
        for(size_t i=0; i<references.size(); ++i) {
          ws_->Attach(ws_->GiveFilenameFromSequence(
              reference_prefix, i),
            &references[i]);
        }
      }
      dense_sizes=references[0]->dense_sizes();
      sparse_sizes=references[0]->sparse_sizes();
      if (vm["distribution_type"].as<std::string>()=="discrete") {
        typedef DiscreteDistribution<TableType> DiscreteDistribution_t;
        std::vector<std::string> distribution_args=
          fl::ws::MakeArgsFromPrefix(args_, "discrete");
        if (vm["transition_matrix_type"].as<std::string>()=="feedforward") {
          Hmm<HmmArgsDiscrete1<TableType> > engine;
          fl::logger->Message()<<"Initializing hmm engine"<<std::endl;
          fl::logger->SuspendLogging();
          engine.Init(n_states,
                      distribution_args,
                      references);
          fl::logger->ResumeLogging();
          if (ff_transition_matrix.get()!=NULL) {
            engine.set_transition_matrix(ff_transition_matrix);
          }
          if (initial_probabilities.get()!=NULL) {
            engine.set_initial_probabilities(initial_probabilities_point);
          }
          engine.Train(iterations,
                       references);
          Export(vm, engine);        
        } 
      } else {
        if (vm["distribution_type"].as<std::string>()=="kde") {
          std::vector<std::string> distribution_args=
              fl::ws::MakeArgsFromPrefix(args_, "kde");
          if (vm["transition_matrix_type"].as<std::string>()=="feedforward") {
            Hmm<HmmArgsKde1<TableType> > engine;
            fl::logger->Message()<<"Initializing hmm engine"<<std::endl;
            fl::logger->SuspendLogging();
            engine.Init(n_states,
                      distribution_args,
                      references);
            fl::logger->ResumeLogging();
            if (ff_transition_matrix.get()!=NULL) {
              engine.set_transition_matrix(ff_transition_matrix);
            } 
            if (initial_probabilities.get()!=NULL) {
              engine.set_initial_probabilities(initial_probabilities_point);
            }
            engine.Train(iterations,
                        references);
            Export(vm, engine);
          } else {
            if (vm["transition_matrix_type"].as<std::string>()=="full") {
              Hmm<HmmArgsKde2<TableType> > engine;
              fl::logger->SuspendLogging();
              engine.Init(n_states,
                      distribution_args,
                      references);
              fl::logger->ResumeLogging();
              if (full_transition_matrix.get()!=NULL) {
                engine.set_transition_matrix(full_transition_matrix);
              }
              if (ff_transition_matrix.get()!=NULL) {
                engine.set_initial_probabilities(initial_probabilities_point);
              }
              engine.Train(iterations,
                       references);
              Export(vm, engine);
            }
          }
        } else {
          std::vector<std::string> distribution_args=
              fl::ws::MakeArgsFromPrefix(args_, "gmm");
           if (vm["transition_matrix_type"].as<std::string>()=="feedforward") {
              Hmm<HmmArgsGmm1<TableType> > engine;
              fl::logger->SuspendLogging();
              engine.Init(n_states,
                      distribution_args,
                      references);
              fl::logger->ResumeLogging();
              if (ff_transition_matrix.get()!=NULL) {
                engine.set_transition_matrix(ff_transition_matrix);
              }
              if (initial_probabilities.get()!=NULL) {
                engine.set_initial_probabilities(initial_probabilities_point);
              }
              engine.Train(iterations,
                       references);
              Export(vm, engine);
           } else {
             if (vm["transition_matrix_type"].as<std::string>()=="full") {
               Hmm<HmmArgsGmm2<TableType> > engine;
               fl::logger->SuspendLogging();
               engine.Init(n_states,
                      distribution_args,
                      references);
               fl::logger->ResumeLogging();
               if (full_transition_matrix.get()!=NULL) {
                 engine.set_transition_matrix(full_transition_matrix); 
               }
               if (ff_transition_matrix.get()!=NULL) {
                 engine.set_initial_probabilities(initial_probabilities_point);
               }
               engine.Train(iterations,
                       references);
               Export(vm, engine);
            }
          }
        }
      }
      fl::logger->Message()<<"Finished Trainging HMM"<<std::endl;
    } else {
      if (vm["task"].as<std::string>()=="eval"
          || vm["task"].as<std::string>()=="generate") {
        std::vector<std::string> init_args;
        std::vector<std::string> exec_args;
        std::string distribution_type=vm["distribution_type"].as<std::string>();
       Hmm<HmmArgsDiscrete1<TableType> > engine1;
       Hmm<HmmArgsDiscrete2<TableType> > engine2;
       LoadHmmParams(
           full_transition_matrix,
           ff_transition_matrix,
           initial_probabilities_point,
           engine1,
           engine2,
           n_states,
           init_args,
           exec_args);
        if (distribution_type=="kde") {
          Hmm<HmmArgsKde1<TableType> > engine1;
          Hmm<HmmArgsKde2<TableType> > engine2;
          LoadHmmParams(
             full_transition_matrix,
             ff_transition_matrix,
             initial_probabilities_point,
             engine1,
             engine2,
             n_states,
             init_args,
             exec_args);
          if (vm["task"].as<std::string>()=="generate") {
            std::vector<std::string> sequences_out;
            if (vm.count("sequence_out")) {
                 
                 
             }
             //engine.Generate(ws_, )  
          
          }
        } 
      }
    } 
  }

  template<typename WorkSpaceType>  
  template<
           typename FullTransitionTableType,
           typename SparseTransitionTableType,
           typename Engine1Type,
           typename Engine2Type,
           typename InitialProbType
          >
  void Hmm<boost::mpl::void_>::Core<WorkSpaceType>::LoadHmmParams(
      FullTransitionTableType full_transition_matrix,
      SparseTransitionTableType ff_transition_matrix,
      InitialProbType &initial_probabilities_point,
      Engine1Type &engine1,
      Engine2Type &engine2,
      int32 n_states,
      const std::vector<std::string> &init_args,
      const std::vector<std::string> &exec_args) {
    if (ff_transition_matrix.get()==NULL) {
      if (ff_transition_matrix.get()!=NULL) {
        engine1.set_transition_matrix(ff_transition_matrix);
      }
      if (initial_probabilities_point.size()!=0) {
        engine2.set_initial_probabilities(initial_probabilities_point);
      }
      std::vector<typename Engine1Type::Distribution_t> &distributions
          =engine1.distributions();
      distributions.resize(n_states);
      for(size_t i=0; i<distributions.size(); ++i) {
        distributions[i].Import(init_args, exec_args, ws_, i);
      }           
    } else {
      if (full_transition_matrix.get()==NULL) {
        if (full_transition_matrix.get()!=NULL) {
          engine2.set_transition_matrix(full_transition_matrix);
        }
        if (initial_probabilities_point.size()!=0) {
          engine2.set_initial_probabilities(initial_probabilities_point);
        }
        std::vector<typename Engine2Type::Distribution_t> &distributions
            = engine2.distributions();
        distributions.resize(n_states);
        for(size_t i=0; i<distributions.size(); ++i) {
          distributions[i].Import(init_args, exec_args, ws_, i);
        }    
      }
    }
  }

  template<typename WorkSpaceType>
  int Hmm<boost::mpl::void_>::Run(
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
  Hmm<boost::mpl::void_>::Core<WorkSpaceType>::Core(
     WorkSpaceType *ws, const std::vector<std::string> &args) :
   ws_(ws), args_(args)  {}

  template<typename WorkSpaceType>
  template<typename EngineType>
  void Hmm<boost::mpl::void_>::Core<WorkSpaceType>::Export(
    boost::program_options::variables_map &vm,
    EngineType &engine) {
    // Loading transition  matrix to the workspace so that
    // we can export it
    if (vm.count("transition_matrix_out")) {
      ws_->LoadTable(vm["transition_matrix_out"].as<std::string>(), 
          engine.transition_matrix());
    }
    if (vm.count("initial_probabilities_out")) {
      boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t> 
          initial_prob_table;
      ws_->Attach(vm["initial_probabilities_out"].as<std::string>(),
          std::vector<index_t>(),
          std::vector<index_t>(1, engine.n_states()),
          1,
          &initial_prob_table);
      typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t point;
      initial_prob_table->get(0, &point);
      point.template sparse_point<double>().Load(
        engine.initial_probabilities().begin(),
        engine.initial_probabilities().end()
      );
      ws_->Purge(initial_prob_table->filename());
      ws_->Detach(initial_prob_table->filename());
    }
    if (vm.count("export_distributions_prefix")) {
      for(size_t i=0; i<engine.distributions().size(); ++i) {
        std::vector<std::string> args=
            fl::ws::MakeArgsFromPrefix(args_,
                vm["export_distributions_prefix"].as<std::string>());
        engine.distributions()[i].Export(args, ws_);
      };
    }
  }
}}
#endif
