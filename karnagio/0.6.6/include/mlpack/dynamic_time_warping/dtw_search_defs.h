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

#ifndef PAPERBOAT_MLPACK_DYNAMIC_TIME_WARPING_DTW_SEARCH_DEFS_H_
#define PAPERBOAT_MLPACK_DYNAMIC_TIME_WARPING_DTW_SEARCH_DEFS_H_
#include <map>
#include "dynamic_time_warping_defs.h"
#include "dtw_search.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/util/string_utils.h"

namespace fl {namespace ml{

  template<typename WorkSpaceType>
  template<typename TableType>
  void DtwSearch<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
    FL_SCOPED_LOG(DtwSearch);
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "references_in",
      boost::program_options::value<std::string>(),
      "the reference data comma separated"
    )(
      "queries_in",
      boost::program_options::value<std::string>(),
      "the query data" 
    )(
      "k_neighbors",
      boost::program_options::value<int32>()->default_value(3),
      "number of neighbors to return"
    )(
      "indices_out",
      boost::program_options::value<std::string>(),
      "file containing the indices of nearest neighbors"
    )(
      "distances_out",
      boost::program_options::value<std::string>(),
      "file containing the distances of the neighbors"
    )(
      "auto_detect_boundaries",
      boost::program_options::value<bool>()->default_value(true),
      "if your data is imported as dense, then all of them will eventually "
      "have equal length. That means your waveforms are padded with zeros "
      "left and right. You are advised to set this option to true so that "
      "zeros will be removed and the waveforms will be copied to more appropriate "
      "and fast structures. If your data is in sparse format you can leave this "
      "flag set to false. Setting it true will use more memory but it will be faster"
    )(
      "scaling_factor",
      boost::program_options::value<double>()->default_value(1.2),
      "the scaling factor for computing dynamic warping with scaling. It must be "
      "greater than one"
    )(
      "horizon",
      boost::program_options::value<index_t>()->default_value(5),
      "the restricted horizon for the dynamic time warping search"  
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
    fl::ws::RequiredOrArgs(vm, "references_in");
    boost::shared_ptr<TableType> references_table;
    ws_->Attach(vm["references_in"].as<std::string>(), &references_table);
    boost::shared_ptr<TableType> queries_table;
    if (vm.count("queries_in")!=0) {
      ws_->Attach(vm["queries_in"].as<std::string>(), &queries_table);
    } else {
      queries_table=references_table;
    }
    boost::shared_ptr<typename WorkSpaceType::UIntegerTable_t> indices;
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> distances;
    ws_->Attach(
        vm.count("indices_out")==0?ws_->GiveTempVarName():vm["indices_out"].as<std::string>(),
        std::vector<index_t>(1, vm["k_neighbors"].as<int32>()),
        std::vector<index_t>(),
        queries_table->n_entries(),
        &indices);  
    ws_->Attach(
        vm.count("distances_out")==0?ws_->GiveTempVarName():vm["distances_out"].as<std::string>(),
        std::vector<index_t>(1, vm["k_neighbors"].as<int32>()),
        std::vector<index_t>(),
        queries_table->n_entries(),
        &distances);  

    double scaling_factor=vm["scaling_factor"].as<double>();
    index_t horizon=vm["horizon"].as<index_t>();
    int32 k_neighbors=vm["k_neighbors"].as<int32>();
    if (vm["auto_detect_boundaries"].as<bool>()) {
      fl::logger->Message()<<"Autodetecting boundaries"<<std::endl;
      boost::shared_ptr<std::vector<std::vector<double> > > references(new std::vector<std::vector<double> >());
      boost::shared_ptr<std::vector<std::vector<double> > > queries(new std::vector<std::vector<double> >());
      TransformQueriesReferences(*references_table,
          *queries_table, 
          references.get(), 
          queries.get());
      fl::logger->Message()<<"Computing all nearest neighbors"<<std::endl;
      ComputeNearestNeighbors(
          *references, 
          *queries, 
          k_neighbors,
          scaling_factor,
          horizon,
          indices.get(),
          distances.get());
    } else {
      fl::logger->Message()<<"Computing all nearest neighbors"<<std::endl;
      ComputeNearestNeighbors(
          *references_table, 
          *queries_table, 
          k_neighbors,
          scaling_factor,
          horizon,
          indices.get(), 
          distances.get());
    }
    fl::logger->Message()<<"Exporting the results"<<std::endl;
    ws_->Purge(indices->filename());
    ws_->Detach(indices->filename());
    ws_->Purge(distances->filename());
    ws_->Detach(distances->filename());
    fl::logger->Message()<<"Done"<<std::endl;
  }


  template<typename WorkSpaceType1>
  template<typename TableType, 
    typename IndicesTableType, 
    typename DistancesTableType>
  void DtwSearch<boost::mpl::void_>::Core<WorkSpaceType1>::
      ComputeNearestNeighbors(
        TableType &references_table, 
        TableType &queries_table, 
        int32 k_neighbors,
        double scaling_factor,
        index_t horizon,
        IndicesTableType *indices, 
        DistancesTableType *distances) {
    index_t q_entries=GetEntries(queries_table);
    index_t r_entries=GetEntries(references_table);
    std::multimap<double, index_t, std::greater<double> > best_scores;
    int chunk_id=0;
    index_t num_of_prunes=0;
    for(index_t i=0; i<q_entries; ++i) {
      best_scores.clear();
      for(int32 k=0; k<k_neighbors; ++k) {
        best_scores.insert(std::make_pair(std::numeric_limits<double>::max(), -1));
      }
      for(index_t j=0; j<r_entries; ++j) {
         double dist_lower_bound=DistanceLB(queries_table, i, references_table, j, scaling_factor, horizon);
         // prune
         if (dist_lower_bound>best_scores.begin()->first) {
           num_of_prunes++;
           continue;
         }
         double dist=Distance(queries_table, 
             i, 
             references_table, 
             j,
             scaling_factor,
             horizon);
         best_scores.insert(std::make_pair(dist,j));
         best_scores.erase(best_scores.begin());
      }
      index_t k=0;
      for(std::multimap<double, index_t, std::greater<double> >::const_reverse_iterator it=best_scores.rbegin();
          it!=best_scores.rend(); ++it) {
        indices->set(i, k, it->second);
        distances->set(i, k, it->first);
        ++k;
      }
      if (10*i/q_entries>=chunk_id) {
        fl::logger->Message()<<"Computed "<<k_neighbors<<"-neighbors "
          "for "<<index_t(100.0*i/q_entries)<<"\% of the query_set"<<std::endl;
        chunk_id++;
        fl::logger->Message()<<"Pruned "<<100.0*num_of_prunes/((i+1)*r_entries)
          <<"\% of the distances so far"<<std::endl;
      }
    } 
    fl::logger->Message()<<"Pruned "<<100.0*num_of_prunes/(q_entries*r_entries)
      <<"\% of the distances"<<std::endl;
  }

  template<typename WorkSpaceType1>
  template<typename TableType>
  index_t DtwSearch<boost::mpl::void_>::Core<WorkSpaceType1>::GetEntries(const TableType &x) {
    return x.n_entries();
  }

  template<typename WorkSpaceType1> 
  index_t DtwSearch<boost::mpl::void_>::Core<WorkSpaceType1>::GetEntries(
      const std::vector<std::vector<double> > &x) {
    return x.size();
  }

  template<typename WorkSpaceType1>   
  template<typename TableType>
  void  DtwSearch<boost::mpl::void_>::Core<WorkSpaceType1>::TransformQueriesReferences(
              TableType &references_table,
              TableType &queries_table, 
              std::vector<std::vector<double> > *references, 
              std::vector<std::vector<double> > *queries) {
    references->resize(references_table.n_entries());
    queries->resize(queries_table.n_entries());
    typename TableType::Point_t point;
    for(index_t i=0; i<references_table.n_entries(); ++i) {
      references_table.get(i, &point);
      for(typename TableType::Point_t::iterator it=point.begin(); it!=point.end(); ++it) {
        if (it.value()!=std::numeric_limits<double>::max()) { 
          (*references)[i].push_back(it.value());
        }
      }
    }   
    for(index_t i=0; i<queries_table.n_entries(); ++i) {
      queries_table.get(i, &point);
      for(typename TableType::Point_t::iterator it=point.begin(); it!=point.end(); ++it) {
        if (it.value()!=std::numeric_limits<double>::max()) { 
          (*queries)[i].push_back(it.value());
        }
      }
    }  
  }
   
  template<typename WorkSpaceType1>
  template<typename TableType>
  double DtwSearch<boost::mpl::void_>::Core<WorkSpaceType1>::DistanceLB(
      TableType &queries, 
      index_t i, 
      TableType &references, 
      index_t j,
      index_t scaling_factor,
      index_t horizon) {
    typename TableType::Point_t q_point, r_point;
    queries.get(i, &q_point);
    references.get(j, &r_point);
    if (q_point.size()<r_point.size()) {
      return ScalingTimeWarping::ScalingTimeWarpingLowerBound(
          q_point, 
          r_point,
          scaling_factor,
          horizon);
    } else {
      return ScalingTimeWarping::ScalingTimeWarpingLowerBound(
          r_point, 
          q_point,
          scaling_factor,
          horizon);

    }
  }
 
  template<typename WorkSpaceType1>
  double DtwSearch<boost::mpl::void_>::Core<WorkSpaceType1>::DistanceLB(
      std::vector<std::vector<double> > &queries, 
      index_t i, 
      std::vector<std::vector<double> > &references, 
      index_t j,
      index_t scaling_factor,
      index_t horizon) {
    if (queries[i].size()<references[j].size()) {
      return ScalingTimeWarping::ScalingTimeWarpingLowerBound(
          queries[i], 
          references[j],
          scaling_factor,
          horizon);
    } else {
       return ScalingTimeWarping::ScalingTimeWarpingLowerBound(
          references[j], 
          queries[i],
          scaling_factor,
          horizon);
   
    }
  } 

  template<typename WorkSpaceType1>
  template<typename TableType>
  double DtwSearch<boost::mpl::void_>::Core<WorkSpaceType1>::Distance(
      TableType &queries, 
      index_t i, 
      TableType &references, 
      index_t j, 
      double scaling_factor,
      index_t horizon) {
    typename TableType::Point_t point1, point2;
    queries.get(i, &point1);
    references.get(j, &point2);
    if (point1.size()<point2.size()) {
      return ScalingTimeWarping::Compute(point1, 
          point2, 
          scaling_factor, 
          horizon);
    } else {
      return ScalingTimeWarping::Compute(point2, 
          point1, 
          scaling_factor, 
          horizon);

    }
  }

  template<typename WorkSpaceType1>
  double DtwSearch<boost::mpl::void_>::Core<WorkSpaceType1>::Distance(
      std::vector<std::vector<double> > &queries, 
      index_t i, 
      std::vector<std::vector<double> > &references, 
      index_t j, 
      double scaling_factor, 
      index_t horizon) {
    if (queries[i].size()<references[j].size()) {
      return ScalingTimeWarping::Compute(
          queries[i], 
          references[j],
          scaling_factor,
          horizon);
    } else {
       return ScalingTimeWarping::Compute(
          references[j], 
          queries[i],
          scaling_factor,
          horizon);
   
    }
  }

  template<typename WorkSpaceType>
  int DtwSearch<boost::mpl::void_>::Run(
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
  DtwSearch<boost::mpl::void_>::Core<WorkSpaceType>::Core(
     WorkSpaceType *ws, const std::vector<std::string> &args) :
   ws_(ws), args_(args)  {}


}}
#endif
