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

#ifndef PAPERBOAT_KARNAGIO_INCLUDE_MLPACK_MULTI_TIME_SERIES_PREDICTION_DEFS_H_
#define PAPERBOAT_KARNAGIO_INCLUDE_MLPACK_MULTI_TIME_SERIES_PREDICTION_DEFS_H_
#include "multi_time_series_predictor.h"
#include "mlpack/svd/svd.h"
#include "mlpack/nmf/nmf.h"
#include "mlpack/nonparametric_regression/nonparametric_regression.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/util/string_utils.h"
#include "fastlib/workspace/based_on_table_run.h"
#include "fastlib/table/linear_algebra.h"


  template<typename WorkSpaceType>
  template<typename TableType>
  void fl::ml::MultiTsPredictor<WorkSpaceType>::PrepareNprQueryData(
      WorkSpaceType *ws,
      TableType &references_table,
      const std::vector<index_t> reference_breaks,
      typename WorkSpaceType::DefaultTable_t &low_dim_references_table,
      typename WorkSpaceType::DefaultTable_t &query_augment_table, 
      index_t timestamp_attribute,
      int32 time_lag,
      const std::string &query_table_name,
      boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *query_table) {
 
    index_t dim=low_dim_references_table.n_entries()==0?0:
      low_dim_references_table.n_attributes();
    // add as extra dimensions the lagged values
    ws->Attach(query_table_name,
        std::vector<index_t>(1, query_augment_table.n_attributes()+dim
          +time_lag),
        std::vector<index_t>(), 
        0, 
        query_table);
    typename TableType::Point_t r_point;
    typename WorkSpaceType::DefaultTable_t::Point_t low_dim_point;
    typename WorkSpaceType::DefaultTable_t::Point_t aug_point;
    const std::vector<index_t> sizes(1, (*query_table)->n_attributes());
    if (timestamp_attribute<0 || 
        timestamp_attribute>=(*query_table)->n_attributes()) {
      fl::logger->Die()<<"timestamp_attribute  must be between "
        <<"0 and "<<(*query_table)->n_attributes()-1;
    }
    for(index_t j=0; j<references_table.n_attributes(); ++j) {
      for(index_t i=0; i<query_augment_table.n_entries(); ++i) {
        query_augment_table.get(i, &aug_point);
        typename WorkSpaceType::DefaultTable_t::Point_t new_point;
        new_point.Init(sizes);
        new_point.meta_data().template get<1>()=0.0;
        if (dim!=0) {
          low_dim_references_table.get(j, &low_dim_point);
          memcpy(new_point.template dense_point<double>().ptr(), 
              low_dim_point.template dense_point<double>().ptr(),
              low_dim_point.size()*sizeof(double));
          memcpy(new_point.template dense_point<double>().ptr()
                            +low_dim_point.size(), 
              aug_point.template dense_point<double>().ptr(),
              aug_point.size()*sizeof(double));
        } else {
          memcpy(new_point.template dense_point<double>().ptr(), 
                aug_point.template dense_point<double>().ptr(),
                aug_point.size()*sizeof(double));
        }
        // add the time lag values
        for(int32 k=0; k<time_lag; ++k) {
          if (reference_breaks.size()<k+1) {
            new_point.set(query_augment_table.n_attributes()
               +dim+k,  0);
          } else {
            new_point.set(query_augment_table.n_attributes()
               +dim+k,  references_table.get(reference_breaks[
                   reference_breaks.size()-k-1]+new_point[timestamp_attribute-1], j));
          }
        }
        (*query_table)->push_back(new_point);
      }
    }
    ws->Purge((*query_table)->filename());
    ws->Detach((*query_table)->filename());
  }


  /**
   * @brief Takes the low dimensional representation of columns augments them
   *        with the augment_table. Adds the the rwo_columns value as a metadata
   */
  template<typename WorkSpaceType>
  template<typename TableType>
  void fl::ml::MultiTsPredictor<WorkSpaceType>::PrepareNprData(
      WorkSpaceType *ws,
      TableType &references_table,
      const std::string &new_table,
      typename WorkSpaceType::DefaultTable_t &low_dim_references_table,
      typename WorkSpaceType::DefaultTable_t &augment_table, 
      index_t timestamp_attribute,
      int32 time_lag,
      std::vector<index_t> *timestamp_breakpoints,
      std::vector<index_t> *reference_breaks) {
 
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> new_references;
    typename WorkSpaceType::DefaultTable_t *new_references_ptr;
    index_t dim=low_dim_references_table.n_entries()==0?0:
      low_dim_references_table.n_attributes();
    // add as extra dimensions the lagged values
    ws->Attach(new_table,
        std::vector<index_t>(1, augment_table.n_attributes()+dim
          +time_lag),
        std::vector<index_t>(), 
        0, 
        &new_references);
    new_references_ptr=new_references.get();
    typename TableType::Point_t r_point;
    typename WorkSpaceType::DefaultTable_t::Point_t low_dim_point;
    typename WorkSpaceType::DefaultTable_t::Point_t aug_point;
    const std::vector<index_t> sizes(1, new_references_ptr->n_attributes());
    double current_timestamp=std::numeric_limits<double>::max();
    if (timestamp_attribute<0 || 
        timestamp_attribute>=new_references_ptr->n_attributes()) {
      fl::logger->Die()<<"timestamp_attribute  must be between "
        <<"0 and "<<new_references_ptr->n_attributes()-1;
    }
    reference_breaks->push_back(0);
    for(index_t i=0; i<references_table.n_entries(); ++i) {
      references_table.get(i, &r_point);
      augment_table.get(i, &aug_point);
      for(typename TableType::Point_t::iterator it=r_point.begin();
          it!=r_point.end(); ++it) {
        typename WorkSpaceType::DefaultTable_t::Point_t new_point;
        new_point.Init(sizes);
        new_point.meta_data().template get<1>()=it.value();
        if (dim!=0) {
          low_dim_references_table.get(it.attribute(), &low_dim_point);
          memcpy(new_point.template dense_point<double>().ptr(), 
              low_dim_point.template dense_point<double>().ptr(),
              low_dim_point.size()*sizeof(double));
          memcpy(new_point.template dense_point<double>().ptr()
                            +low_dim_point.size(), 
              aug_point.template dense_point<double>().ptr(),
              aug_point.size()*sizeof(double));
        } else {
          augment_table.get(i, &aug_point);
          memcpy(new_point.template dense_point<double>().ptr(), 
              aug_point.template dense_point<double>().ptr(),
              aug_point.size()*sizeof(double));
        }
        // add the time lag values
        for(int32 k=0; k<time_lag; ++k) {
          if (reference_breaks->size()<k+1) {
            new_point.set(augment_table.n_attributes()
               +dim+k,  it.value());
          } else {
            new_point.set(augment_table.n_attributes()
               +dim+k,  references_table.get((*reference_breaks)[
                   reference_breaks->size()-k-1]+new_point[timestamp_attribute-1], it.attribute()));
          }
        }
        new_references_ptr->push_back(new_point);
        double timestamp=new_point[timestamp_attribute];
        if (timestamp!=current_timestamp) {
          timestamp_breakpoints->push_back(
              new_references_ptr->n_entries()-1);
          current_timestamp=timestamp;
          reference_breaks->push_back(i);
        }
      }
    }
    ws->Purge(new_references->filename());
    ws->Detach(new_references->filename());
  }


  template<typename WorkSpaceType>
  fl::ml::MultiTsPredictor<boost::mpl::void_>::Core<WorkSpaceType>::Core(WorkSpaceType *ws, const std::vector<std::string> &args) :
  ws_(ws), args_(args){
  
  }

  template<typename WorkSpaceType>
  template<typename TableType>
  void fl::ml::MultiTsPredictor<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
    FL_SCOPED_LOG(MultiTsPredictor);
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "references_in",
      boost::program_options::value<std::string>(),
      "the reference data "
    )(
      "references_augmented_data_in",
      boost::program_options::value<std::string>(),
      "Extra columns to be appended in the references_in table"
    )(
      "queries_augmented_data_in",
      boost::program_options::value<std::string>(),
      "Extra columns to be used for prediction, they contain the timestamps "
      "for the prediction"
    )("summary",
      boost::program_options::value<std::string>()->default_value("svd"),
      "Your data is high dimensional and you need to reduce its dimension "
      "you need to use a dimensionality reduction method to summarize your "
      "data in a lower dimensional space. Available options are: \n"
      "  svd :  uses svd to reduce the dimensionality\n"
      "  nmf :  uses nmf to reduce the dimensionality\n"
      "  none:  does not use any summarization method"
    )(
      "timestamp_attribute",
      boost::program_options::value<index_t>(),
      "In the table augmented one attribute must be the timestamp"
      "you need to inform the program by setting this option" 
    )(
      "time_lag",
      boost::program_options::value<int32>()->default_value(0),
      "Every point for regression must also include --time_lagg values "
    )(
      "window",
      boost::program_options::value<index_t>()->default_value(4),
      "this is the rolling time window to use for prediction"  
    )(
      "fixed_bw",
      boost::program_options::value<std::string>(),
      "if you want some bandwidths to be fixed, then provide "
      "a comma separated list like 0:0.33,1:4.33 etc"
    )(
      "run_mode",
      boost::program_options::value<std::string>()->default_value("train"),
      "You can choose either train or predict. The predict will compute the "
      "predicted values of the last timestamp"
    )(
      "predictions_out",
      boost::program_options::value<std::string>(),
      "this is the table to store the predicted values when using --run_mode=predict"  
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

    if (vm.count("references_in")==0) {
      fl::logger->Die()<<"You must set the --references_in flag";
    }
    index_t timestamp_attribute=0;
    if (vm.count("timestamp_attribute")==0) {
      fl::logger->Die()<<"You need to set --timestamp_attribute option";
    } else {
      timestamp_attribute=vm["timestamp_attribute"].as<index_t>();
    }

    std::string references_in=vm["references_in"].as<std::string>();
    boost::shared_ptr<TableType> references_table;
    ws_->Attach(references_in, &references_table);
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> low_dim_references_table;
    const std::string summary=vm["summary"].as<std::string>();
    if (summary=="svd") {
      std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args_, "svd");
      args1.push_back("--references_in="+references_in);
      args1.push_back("--rsv_out=rsv");
      args1.push_back("--sv_out=sv");
      args1.push_back("--col_mean_normalize=0");
      fl::ml::Svd<boost::mpl::void_>::Run(ws_, args1);
      boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> rsv_table;
      ws_->Attach("rsv", &rsv_table);
      boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> sv_table;
      ws_->Attach("sv", &sv_table);
      low_dim_references_table.reset(new typename WorkSpaceType::DefaultTable_t());
      typename WorkSpaceType::DefaultTable_t sv_table1;
      sv_table1.Init("",
          std::vector<index_t>(1, sv_table->n_entries()),
          std::vector<index_t>(),
          1); 
      typename WorkSpaceType::DefaultTable_t::Point_t point;
      sv_table1.get(0, &point);
      for(index_t i=0; i<point.size(); ++i) {
        point.set(i, sv_table->get(i, index_t(0)));
      }
      fl::table::Scale(*rsv_table,
                       sv_table1,
                       low_dim_references_table.get());

    } else {
      if (summary=="nmf") {
        std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args_, "nmf");
        args1.push_back("--h_table_out=h_table");
        fl::ml::Nmf::Run(ws_, args1);
        ws_->Attach("h_table", &low_dim_references_table);
      } else {
        if (summary=="none") {
          low_dim_references_table.reset(new typename WorkSpaceType::DefaultTable_t());
          fl::logger->Message()<<"No summarization of the data"
            <<"no cross correlation will be used";     
        } else {
          fl::logger->Die()<<"This --summary="<<summary<<" is not supported";
        }
      }
    }
    const std::string new_references_name=ws_->GiveTempVarName();
    std::vector<index_t> timestamp_breakpoints;
    std::vector<index_t> reference_breakpoints;
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> augmented_table;
    if (vm.count("references_augmented_data_in")) {
      ws_->Attach(vm["references_augmented_data_in"].as<std::string>(), 
          &augmented_table);
      if (augmented_table->n_entries()!=references_table->n_entries()) {
        fl::logger->Die()<<"--references_augmented_data_in="<<vm["references_augmented_data_in"].as<std::string>()
         <<" must have the same number of entries with "
         <<"--references_in="<<references_in;
      }
      fl::logger->Message()<<"After summarization preparing the data for "
        "Nonparametric regression"<<std::endl;
      int32 time_lag=vm["time_lag"].as<int32>();
      MultiTsPredictor<WorkSpaceType>::PrepareNprData(ws_,
          *references_table,
          new_references_name,
          *low_dim_references_table,
          *augmented_table,
          timestamp_attribute,
          time_lag,
          &timestamp_breakpoints,
          &reference_breakpoints);
      fl::logger->Message()<<"Data ready for nonparametric regression"<<std::endl;
    } else {
      fl::logger->Die()<<"You need to set --augmented_data_in. "
          "You always have to provide --augmented_data_in that will contain "
          "the timestamp info"<<std::endl;
    }
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> npr_ready_table;
    ws_->Attach(new_references_name, &npr_ready_table);
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> npr_reference_table;
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> npr_query_table;
    index_t window=vm["window"].as<index_t>();
    // We will get a window of points for training 
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> bandwidths_so_far;
    ws_->Attach(ws_->GiveTempVarName(), 
        std::vector<index_t>(1, npr_ready_table->n_attributes()),
        std::vector<index_t>(),
        1,
        &bandwidths_so_far);
    std::vector<double> means;
    std::vector<double> variances;
    fl::logger->Message()<<"Computing attribute statistics on the NPR "
      "table"<<std::endl;
    npr_ready_table->AttributeStatistics(&means, &variances);
    std::string result;
    for(size_t i=0; i<means.size(); ++i) {
      result+="("+boost::lexical_cast<std::string>(i)
                 +", "+boost::lexical_cast<std::string>(means[i])
                 +", "+boost::lexical_cast<std::string>(variances[i])
                 +") ";
    }
    fl::logger->Message()<<"(attribute, means, variances)="
        <<result<<std::endl;
    typename WorkSpaceType::DefaultTable_t::Point_t point;
    bandwidths_so_far->get(0, &point);
    for(index_t i=0; i<point.size(); ++i) {
      point.set(i, sqrt(variances[i]));
    }
    if (vm.count("fixed_bw")) {
      std::vector<std::string> bws=fl::SplitString(
          vm["fixed_bw"].as<std::string>(), ",");
      for(size_t i=0; i<bws.size(); ++i) {
        std::vector<std::string> tokens=fl::SplitString(bws[i], ":");
        if (tokens.size()!=2) {
          fl::logger->Die()<<"--fixed_bw ("<<bws[i]<<") "
            "has wrong format";
        }
        try {
          int ind=boost::lexical_cast<int>(tokens[0]);
          double val=boost::lexical_cast<double>(tokens[1]);
          point.set(ind, val);
        } 
        catch(...) {
          fl::logger->Die()<<"--fixed_bw ("<<bws[i]<<") "
            "has wrong format";

        }
      }
    }
    fl::logger->Message()<<"timestamp_breakpoints.size()="
      << timestamp_breakpoints.size()<<std::endl;
    if (vm["run_mode"].as<std::string>()=="train") {
      if (vm.count("predictions_out")>0) {
        fl::logger->Die()<<"Setting --predictions_out when "
          <<"--run_mode=train is illegal";
      }
      for(size_t i=0; i<0*(timestamp_breakpoints.size()-window-1); ++i) {
        fl::logger->Message()<<"Training for window="<<i<<std::endl;
        boost::shared_ptr<WorkSpaceType> local_ws(new WorkSpaceType()); 
        local_ws->set_schedule_mode(2);
        const std::string npr_reference_name=ws_->GiveTempVarName();
        const std::string npr_query_name=ws_->GiveTempVarName();
        local_ws->Attach(npr_reference_name,
            std::vector<index_t>(1, npr_ready_table->n_attributes()),
            std::vector<index_t>(), 
            0, 
            &npr_reference_table);
        local_ws->Attach(npr_query_name,
            std::vector<index_t>(1, npr_ready_table->n_attributes()),
            std::vector<index_t>(), 
            0, &npr_query_table);
  
        index_t counter1=timestamp_breakpoints[i];
        index_t counter2=timestamp_breakpoints[i+window];
        typename WorkSpaceType::DefaultTable_t::Point_t point;
        for(index_t j=counter1; j<counter2; ++j) {
          npr_ready_table->get(j, &point);
          npr_reference_table->push_back(point);
        }
        index_t counter3=timestamp_breakpoints[i+window+1];
        for(index_t j=counter2; j<counter3; ++j) {
          npr_ready_table->get(j, &point);
          npr_query_table->push_back(point);
        }
        fl::logger->Message()<<"Query points="<<npr_query_table->n_entries()
          <<", Reference points="<<npr_reference_table->n_entries()<<std::endl;
        local_ws->Purge(npr_reference_name);
        local_ws->Detach(npr_reference_name);
        local_ws->Purge(npr_query_name);
        local_ws->Detach(npr_query_name);
        std::vector<std::string> npr_args;
        npr_args.push_back("--references_in="+npr_reference_name);
        npr_args.push_back("--queries_in="+npr_query_name);
        npr_args.push_back("--bandwidths_out=bandwidths");
        npr_args.push_back("--ref_split_factor=-1");
        npr_args.push_back("--query_split_factor=1e-5");
        npr_args.push_back("--mse_out=mse");
        npr_args.push_back("--train_algorithm=lbfgs");
        npr_args.push_back("--iterations=10");
        npr_args.push_back("--lbfgs_rank=4");
        //npr_args.push_back("--bandwidths_init=1");
        local_ws->IndexAllReferencesQueries(npr_args);
        // here we create a table that has the average bandwidths so far
        // in order to seed the next training
        boost::shared_ptr<
          typename WorkSpaceType::DefaultTable_t> current_bandwidths;
        local_ws->Attach(local_ws->GiveTempVarName(),
            std::vector<index_t>(1, bandwidths_so_far->n_attributes()),
            std::vector<index_t>(),
            1,
            &current_bandwidths);
        typename WorkSpaceType::DefaultTable_t::Point_t point1, point2;
        current_bandwidths->get(0, &point1);
        bandwidths_so_far->get(0, &point2);
        for(index_t k=0; k<point1.size(); ++k) {
          point1.set(k, point2[k]);
        }
        local_ws->Purge(current_bandwidths->filename());
        local_ws->Detach(current_bandwidths->filename());
        npr_args.push_back("--bandwidths_in="+current_bandwidths->filename());
        fl::logger->SuspendLogging();
        fl::ml::NonParametricRegression<boost::mpl::void_>::Run(
            local_ws.get(), npr_args);
        fl::logger->ResumeLogging();
        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> mse_table;
        local_ws->Attach("mse", &mse_table);
        fl::logger->Message()<<"Mean square error="
          <<mse_table->get(index_t(0),index_t(0))<<"%"<<std::endl;
        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> new_bandwidths;
        local_ws->Attach("bandwidths", &new_bandwidths);
        bandwidths_so_far->get(0, &point1);
        new_bandwidths->get(0, &point2);
        for(index_t i=0; i<point1.size(); ++i) {
          point1.set(i, point2[i]);
        }
      }
      // now we compute the total error
      fl::logger->Message()<<"Finished training, evaluating training error"
        <<std::endl;
      double total_error=0;
      index_t total_points=0;
      for(size_t i=0; i<timestamp_breakpoints.size()-window-1; ++i) {
        fl::logger->Message()<<"Evaluating for window="<<i<<std::endl;
        boost::shared_ptr<WorkSpaceType> local_ws(new WorkSpaceType()); 
        local_ws->set_schedule_mode(2);
        const std::string npr_reference_name=local_ws->GiveTempVarName();
        const std::string npr_query_name=local_ws->GiveTempVarName();
        local_ws->Attach(npr_reference_name,
            std::vector<index_t>(1, npr_ready_table->n_attributes()),
            std::vector<index_t>(), 
            0, 
            &npr_reference_table);
        local_ws->Attach(npr_query_name,
            std::vector<index_t>(1, npr_ready_table->n_attributes()),
            std::vector<index_t>(), 
            0, &npr_query_table);
  
        index_t counter1=timestamp_breakpoints[i];
        index_t counter2=timestamp_breakpoints[i+window];
        typename WorkSpaceType::DefaultTable_t::Point_t point;
        for(index_t j=counter1; j<counter2; ++j) {
          npr_ready_table->get(j, &point);
          npr_reference_table->push_back(point);
        }
        index_t counter3=timestamp_breakpoints[i+window+1];
        for(index_t j=counter2; j<counter3; ++j) {
          npr_ready_table->get(j, &point);
          npr_query_table->push_back(point);
        }
        local_ws->Purge(npr_reference_name);
        local_ws->Detach(npr_reference_name);
        local_ws->Purge(npr_query_name);
        local_ws->Detach(npr_query_name);
        std::vector<std::string> npr_args;
        npr_args.push_back("--references_in="+npr_reference_name);
        npr_args.push_back("--queries_in="+npr_query_name);
        npr_args.push_back("--mse_out=mse");
        npr_args.push_back("--run_mode=eval");
        npr_args.push_back("--relative_error=0.01");
        // here we create a table that has the average bandwidths so far
        // in order to seed the next training
        boost::shared_ptr<
          typename WorkSpaceType::DefaultTable_t> current_bandwidths;
        local_ws->Attach(local_ws->GiveTempVarName(),
            std::vector<index_t>(1, bandwidths_so_far->n_attributes()),
            std::vector<index_t>(),
            1,
            &current_bandwidths);
        typename WorkSpaceType::DefaultTable_t::Point_t point1, point2;
        current_bandwidths->get(0, &point1);
        bandwidths_so_far->get(0, &point2);
        for(index_t k=0; k<point1.size(); ++k) {
          point1.set(k, point2[k]);
        }
        local_ws->Purge(current_bandwidths->filename());
        local_ws->Detach(current_bandwidths->filename());
        npr_args.push_back("--bandwidths_in="+current_bandwidths->filename());
        fl::logger->SuspendLogging();
        fl::ml::NonParametricRegression<boost::mpl::void_>::Run(
            local_ws.get(), npr_args);
        fl::logger->ResumeLogging();
        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> mse_table;
        local_ws->Attach("mse", &mse_table);
        double mse=mse_table->get(index_t(0),index_t(0));
        fl::logger->Message()<<"Mean square error="
          <<mse<<"%"<<std::endl;
        total_error+=(mse/100)*(mse/100)*npr_query_table->n_entries();
        total_points+=npr_query_table->n_entries();
      }
      fl::logger->Message()<<"Training error="<<
        100*sqrt(total_error/total_points)<<std::endl;
    } else {
      if (vm["run_mode"].as<std::string>()=="predict") {
        fl::logger->Message()<<"Predicting values"
          <<std::endl;
        std::vector<std::string> npr_args;
        if (vm.count("predictions_out")==0) {
           fl::logger->Die()<<"When running in --run_mode=predict "
             <<"you must set the --predictions_out flag";
        }
        if (vm.count("queries_augmented_data_in")==0) {
          fl::logger->Die();
        }
        const std::string query_table_name=ws_->GiveTempVarName();
        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> query_table;
        int32 time_lag=vm["time_lag"].as<int32>();
        fl::logger->Message()<<"Preparing the data"<<std::endl;
        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> query_augmented_table;
        ws_->Attach(vm["queries_augmented_data_in"].as<std::string>(), 
            &query_augmented_table);
        MultiTsPredictor<WorkSpaceType>::PrepareNprQueryData(
            ws_,
            *references_table,
            reference_breakpoints,
            *low_dim_references_table,
            *query_augmented_table, 
            timestamp_attribute,
            time_lag,
            query_table_name,
            &query_table);
        fl::logger->Message()<<"Data ready for NPR"<<std::endl;
        std::string predictions_out=vm["predictions_out"].as<std::string>();
        npr_args.push_back("--references_in="+npr_ready_table->filename());
        npr_args.push_back("--queries_in="+query_table->filename());
        npr_args.push_back("--predictions_out="+predictions_out);
        npr_args.push_back("--run_mode=eval");
        npr_args.push_back("--relative_error=0.01");
        // here we create a table that has the average bandwidths so far
        // in order to seed the next training
        boost::shared_ptr<
          typename WorkSpaceType::DefaultTable_t> current_bandwidths;
        ws_->Attach(ws_->GiveTempVarName(),
            std::vector<index_t>(1, bandwidths_so_far->n_attributes()),
            std::vector<index_t>(),
            1,
            &current_bandwidths);
        typename WorkSpaceType::DefaultTable_t::Point_t point1, point2;
        current_bandwidths->get(0, &point1);
        bandwidths_so_far->get(0, &point2);
        for(index_t k=0; k<point1.size(); ++k) {
          point1.set(k, point2[k]);
        }
        ws_->Purge(current_bandwidths->filename());
        ws_->Detach(current_bandwidths->filename());
        npr_args.push_back("--bandwidths_in="+current_bandwidths->filename());
        fl::logger->Message()<<"Ready to run NPR"<<std::endl;
        fl::logger->Message()<<"Reference Table num_of_points"<<npr_ready_table->n_entries()<<std::endl;
        fl::logger->Message()<<"Query Table num_of_points="<<query_table->n_entries()<<std::endl;
        fl::logger->SuspendLogging();
        fl::ml::NonParametricRegression<boost::mpl::void_>::Run(
            ws_, npr_args);
        fl::logger->ResumeLogging();
        fl::logger->Message()<<"NPR done"<<std::endl;
      
      } else {
        fl::logger->Die()<<"The flag --run_mode="
            <<vm["run_mode"].as<std::string>()
            <<" is wrong, it can either be train or predict";
      }
    }
  }

  template<typename WorkSpaceType>
  int fl::ml::MultiTsPredictor<boost::mpl::void_>::Run(
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
#endif
