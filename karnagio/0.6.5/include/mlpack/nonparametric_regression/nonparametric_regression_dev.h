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
#ifndef PAPER_BOAT_INCLUDE_MLPACK_NONPARAMETRIC_REGRESSION_DEV_H_
#define PAPER_BOAT_INCLUDE_MLPACK_NONPARAMETRIC_REGRESSION_DEV_H_

#include "nonparametric_regression.h"
#include "boost/program_options.hpp"
#include "fastlib/metric_kernel/weighted_lmetric_dev.h"
#include "fastlib/optimization/lbfgs/lbfgs_dev.h"
#include "fastlib/math/fl_math.h"
#include "fastlib/data/linear_algebra.h"
#include "mlpack/kde/kde_dev.h"
#include "mlpack/kde/dualtree_dfs_dev.h"
#include "fastlib/workspace/task.h"

namespace fl { namespace ml {
  template<typename TableType>
  NonParametricRegression<TableType>::Trainer::Trainer() {
    bandwidths_.SetAll(1.0);
    references_=NULL;
    queries_=NULL;
    lbfgs_rank_=5;
    num_of_line_searches_=5;
    iterations_=10;
    iteration_chunks_=2;
    error_=0;
    eta0_=1;
  }
  
  template<typename TableType>
  void NonParametricRegression<TableType>::Trainer::InitBandwidthsFromData() {
    WPoint_t mean;
    mean.Init(references_->n_attributes());
    mean.SetAll(0);
    Point_t point;
    for(index_t i=0; i<references_->n_entries(); ++i) {
      references_->get(i, &point);
      for(typename Point_t::iterator it=point.begin(); it!=point.end(); ++it) {
        mean.set(it.attribute(), mean[it.attribute()]+it.value());
      }
    }
    for(index_t i=0; i<mean.size(); ++i) {
      mean[i]=mean[i]/references_->n_entries();
    }
    // now compute variance;
    bandwidths_.Init(references_->n_attributes());
    bandwidths_.SetAll(0);
    for(index_t i=0; i<queries_->n_entries(); ++i) {
      queries_->get(i, &point);
      for(typename Point_t::iterator it=point.begin(); it!=point.end(); ++it) {
        bandwidths_.set(it.attribute(), bandwidths_[it.attribute()]
            +fl::math::Pow<double,2,1>(mean[it.attribute()]-it.value()));
      }
    }
    for(index_t i=0; i<bandwidths_.size(); ++i) {
      bandwidths_[i]=fl::math::Pow<double,1,2>(bandwidths_[i])/queries_->n_entries();
    }
    bandwidths_.Print(std::cout, ",");
    std::cout<<std::endl;
  }

  template<typename TableType>
  double NonParametricRegression<TableType>::Trainer::StochasticTrain() {
    typedef fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double>  > WLMetric_t;
    FL_SCOPED_LOG(Sgd);
    double qnorm=0;
    index_t dim=queries_->n_attributes();
    Point_t qpoint;
    Point_t rpoint;  
    for(index_t i=0; i<queries_->n_entries(); ++i) {
      queries_->get(i, &qpoint);
      //double value=qpoint.meta_data().template get<1>();
      double qvalue=qpoint.meta_data().template get<1>();
      qnorm+=qvalue*qvalue;
    }
    if (qnorm==0) {
      fl::logger->Die()<<"The data does not contain regression values"; 
    }
    // set this allias for bandwidths_ so that it is more handy
    WPoint_t &x=bandwidths_;
    bandwidths_.Print(std::cout, ",");
        std::cout<<std::endl;
    double old_objective=Evaluate(x);
    fl::logger->Message()<<"objective="<<old_objective
      <<", error="<<100*fl::math::Pow<double, 1,2>(old_objective/qnorm)<<"%"
      <<std::endl;
      for(index_t it=1; it<iterations_+1; ++it) {
      double eta=eta0_/it;
      index_t non_contributing_points=0;
      for(index_t i=0; i<queries_->n_entries(); ++i) {
        std::vector<index_t> dimension(1, dim);
        WPoint_t x2, x3;
        x2.Init(dimension);
        x3.Init(dimension);
        for(unsigned int k=0; k<dim; ++k) {
          double temp=x[k]*x[k];
          x2[k]=2*temp;
          x3[k]=temp*x[k];
        }
        queries_->get(i, &qpoint);
        //double value=qpoint.meta_data().template get<1>();
        std::vector<double> d_numerator(dim, 0);
        std::vector<double> d_denominator(dim, 0);
        double numerator=0;
        double denominator=0;
        double qvalue=qpoint.meta_data().template get<1>();
        std::vector<double> elements(dim);
        WPoint_t df_dx;
        df_dx.Init(dim);
        df_dx.SetAll(0);
        for(index_t j=0; j<references_->n_entries(); ++j) {
          references_->get(j, &rpoint);
          double rvalue=rpoint.meta_data().template get<1>();
          double val1, val2;
          double distance=0;
          for(unsigned int k=0; k<dim; ++k) {
            val1=qpoint[k];
            val2=rpoint[k];
            double square=(val1-val2)*(val1-val2);
            distance+=square/x2[k];
            elements[k]=square/x3[k];
          }
         // std::cout<<it<<" "<<distance<<" "<<" "<<denominator<<" "<<i<<std::endl;
          if (distance==0) {
            continue;
          }
          double expvalue=exp(-distance);
          numerator+=rvalue*expvalue;
          denominator+=expvalue;
          for(int k=0; k<dim; ++k) {
            d_numerator[k]+=rvalue*expvalue*elements[k];
            d_denominator[k]+=expvalue*elements[k];
          }
        }
        if (denominator>1e-40) {
          for(int k=0;k<dim; ++k) {
            double derivative=(qvalue-(numerator/denominator))*
              (denominator*d_numerator[k]-numerator*d_denominator[k])/
              (denominator*denominator);
            df_dx.set(k, df_dx[k]+derivative);
          }   
          WPoint_t new_x;
          new_x.Init(dim);
          for(unsigned int k=0; k<dim; ++k) {
            new_x[k] = x[k] + eta * 2*df_dx[k];
          }  
          double new_objective=Evaluate(new_x);
          //std::cout<<new_objective<<std::endl;
          //x.Print(std::cout, ",");
          //std::cout<<std::endl;
          if (new_objective<old_objective) {
            x.CopyValues(new_x);
            old_objective=new_objective;
            fl::logger->Message()<<"iteration="<<it
                <<", eta="<<eta
                <<", objective="<<new_objective
                <<", error="<<100*fl::math::Pow<double, 1,2>(new_objective/qnorm)<<"%"
                <<std::endl;
          } else {
            non_contributing_points++;
            //fl::logger->Debug()<<"Point is not contributing in error minimization"<<std::endl;
          }
        } else {
          non_contributing_points++;
          //fl::logger->Debug()<<"Point rejected because of low confidence"<<std::endl;
        }
      }
      fl::logger->Message()<<100.0*non_contributing_points/queries_->n_entries()<<"%"
        " of quering points did not contribute in this iteration"<<std::endl;
    }
    return 100*fl::math::Pow<double, 1,2>(old_objective/qnorm);
  }
  template<typename TableType>
  double NonParametricRegression<TableType>::Trainer::LbfgsTrain() {
    fl::ml::Lbfgs<Trainer> optimizer;
    optimizer.Init(*this, lbfgs_rank_);
    optimizer.set_max_num_line_searches(num_of_line_searches_);
    double qnorm=0;
    Point_t qpoint;
    for(index_t i=0; i<queries_->n_entries(); ++i) {
      queries_->get(i, &qpoint);
      //double value=qpoint.meta_data().template get<1>();
      double qvalue=qpoint.meta_data().template get<1>();
      qnorm+=qvalue*qvalue;
    }
    if (qnorm==0) {
      fl::logger->Die()<<"The data does not contain regression values"; 
    }
    double old_error=std::numeric_limits<double>::max();
    error_=old_error;
    WPoint_t old_bandwidths=bandwidths_;
    
    for(index_t i=0; i<=iterations_/iteration_chunks_; ++i) {
      bool success=false;
      success=optimizer.Optimize(iteration_chunks_, &old_bandwidths);
      std::pair<WPoint_t, double > result=optimizer.min_point_iterate();
      old_error= fl::math::Pow<double, 1, 2>(result.second/qnorm)*100;
      if (old_error<error_) {
        error_=old_error;
        bandwidths_.Copy(old_bandwidths);
        bandwidths_.Print(std::cout, ",");
        std::cout<<std::endl;
      }
      fl::logger->Message()<<"iteration="<<i*iteration_chunks_<<"-"<<(i+1)*iteration_chunks_
        <<", error="<<old_error<<"%"<<std::endl;
      if (success==false && error_>0.1) {
        fl::logger->Debug()<<"readjusting bandwidths"<<std::endl;
        fl::la::SelfScale(0.1/(i+1), &old_bandwidths);
      }
    }
    return error_;
  }

  template<typename TableType>
  void NonParametricRegression<TableType>::Trainer::ComputeGradient(Table_t &refereces, Table_t &queries,
      WPoint_t &x, WPoint_t *df_dx) {
   
   typedef fl::math::WeightedLMetric<2, fl::data::MonolithicPoint<double>  > WLMetric_t;
   int dim=queries.n_attributes();
   df_dx->SetAll(0);
   Point_t qpoint;
   Point_t rpoint;
   std::vector<index_t> dimension(1, dim);
   WPoint_t x2, x3;
   x2.Init(dimension);
   x3.Init(dimension);
   for(unsigned int i=0; i<dim; ++i) {
     double temp=x[i]*x[i];
     x2[i]=2*temp;
     x3[i]=temp*x[i];
   }

   for(index_t i=0; i<queries.n_entries(); ++i) {
     queries.get(i, &qpoint);
     //double value=qpoint.meta_data().template get<1>();
     std::vector<double> d_numerator(dim, 0);
     std::vector<double> d_denominator(dim, 0);
     double numerator=0;
     double denominator=0;
     double qvalue=qpoint.meta_data().template get<1>();
     std::vector<double> elements(dim);
     for(index_t j=0; j<references_->n_entries(); ++j) {
       references_->get(j, &rpoint);
       double rvalue=rpoint.meta_data().template get<1>();
       double val1, val2;
       double distance=0;
       for(unsigned int k=0; k<dim; ++k) {
         val1=qpoint[k];
         val2=rpoint[k];
         double square=(val1-val2)*(val1-val2);
         distance+=square/x2[k];
         elements[k]=square/x3[k];
       }
       if (distance==0) {
         continue;
       }
       double expvalue=exp(-distance);
       numerator+=rvalue*expvalue;
       denominator+=expvalue;
       for(int k=0; k<dim; ++k) {
         d_numerator[k]+=rvalue*expvalue*elements[k];
         d_denominator[k]+=expvalue*elements[k];
       }
     }
     if (denominator>1e-20) {
       for(int k=0;k<dim; ++k) {
         double derivative=(qvalue-(numerator/denominator))*
           (denominator*d_numerator[k]-numerator*d_denominator[k])/
           (denominator*denominator);
         df_dx->set(k, df_dx->operator[](k)+derivative);
       }   
     } else {
       // do nothing just reject this point
     }
   }
   for(unsigned int k=0; k<dim; ++k) {
     df_dx->set(k, -2*df_dx->operator[](k));
   }
 }

  template<typename TableType>
  void NonParametricRegression<TableType>::Trainer::Gradient(WPoint_t &x, WPoint_t *df_dx) {
    ComputeGradient(*references_, *queries_, x, df_dx);
  }
  
  template<typename TableType>
  double NonParametricRegression<TableType>::Trainer::Evaluate(const WPoint_t &x) {
    int dim=queries_->n_attributes();
    Point_t qpoint;
    Point_t rpoint;
    double error=0; 
    index_t low_confidence_points=0;
    for(index_t i=0; i<queries_->n_entries(); ++i) {
      queries_->get(i, &qpoint);
      double numerator=0;
      double denominator=0;
      double qvalue=qpoint.meta_data().template get<1>();
      for(index_t j=0; j<references_->n_entries(); ++j) {
        references_->get(j, &rpoint);
        double rvalue=rpoint.meta_data().template get<1>();
        double val1, val2;
        double distance=0;
        for(int k=0; k<dim; ++k) {
          val1=qpoint[k];
          val2=rpoint[k];
          double square=(val1-val2)*(val1-val2);
          distance+=square/(2*x[k]*x[k]);
        }
        if (distance==0) {
          continue;
        }
        double expvalue=exp(-distance);
        numerator+=rvalue*expvalue;
        denominator+=expvalue;
      }
      // we need this step to avoid overflows
      if (denominator<1e-40){
        low_confidence_points++;
      } else {
        error+=fl::math::Pow<double,2,1>(qvalue-numerator/denominator);
      }
    }
    if (low_confidence_points>0.1*queries_->n_entries()){
      return std::numeric_limits<double>::max();
    } else {
      error+=low_confidence_points*(error/(queries_->n_entries()-low_confidence_points));
    }
    return error;
  }
  
  template<typename TableType>
  index_t NonParametricRegression<TableType>::Trainer::num_dimensions() {
    return references_->n_attributes();
  }
  
  template<typename TableType>
  typename NonParametricRegression<TableType>::Trainer::WPoint_t 
  NonParametricRegression<TableType>::Trainer::get_bandwidths() {
    return bandwidths_;  
  }
  
  template<typename TableType>
  double NonParametricRegression<TableType>::Trainer::get_error() {
    return error_;
  }
  
  template<typename TableType>
  void NonParametricRegression<TableType>::Trainer::set_references(
      Table_t *references) {
    references_=references;
  }

  template<typename TableType>
  void NonParametricRegression<TableType>::Trainer::set_queries(Table_t *queries) {
    queries_=queries;
  }

  template<typename TableType>
  void NonParametricRegression<TableType>::Trainer::set_lbfgs_rank(int lbfgs_rank) {
    lbfgs_rank_=lbfgs_rank;
  }

  template<typename TableType>
  void NonParametricRegression<TableType>::Trainer::set_num_of_line_searches(int num_of_line_searches) {
    num_of_line_searches_=num_of_line_searches; 
  }

  template<typename TableType>
  void NonParametricRegression<TableType>::Trainer::set_iterations(int iterations) {
    iterations_=iterations; 
  }

  template<typename TableType>
  void NonParametricRegression<TableType>::Trainer::set_iteration_chunks(
      int iteration_chunks) {
    iteration_chunks_=iteration_chunks; 
  }

  template<typename TableType>
  void NonParametricRegression<TableType>::Trainer::set_bandwidths(
      fl::data::MonolithicPoint<double> &bandwidths) {
    bandwidths_.Copy(bandwidths);  
  }

  template<typename TableType>
  void NonParametricRegression<TableType>::Trainer::set_eta0(double eta0) {
    eta0_=eta0;
  }

  template<typename TableType>
  NonParametricRegression<TableType>::Predictor::Predictor() {
    probability_=1;  
    references_=NULL;
    relative_error_=0.1;
  } 

  template<typename TableType>
  void NonParametricRegression<TableType>::Predictor::set_references(
      Table_t *references) {
    references_=references;
  }
 
  template<typename TableType>
  void NonParametricRegression<TableType>::Predictor::set_relative_error(
      double relative_error) {
    relative_error_=relative_error;
  }

  template<typename TableType>
  void NonParametricRegression<TableType>::Predictor::set_probability(
      double probability) {
    probability_=probability;
  }
  template<typename TableType>
  void NonParametricRegression<TableType>::Predictor::Predict(
      Table_t *queries,
      NprResult<std::vector<double> > *result) {

    Npr_t npr_instance;
    double bandwidth=1.0;
    npr_instance.Init(references_,
                      queries, 
                      bandwidth,
                      relative_error_, 
                      probability_);
    fl::ml::DualtreeDfs<Npr_t> dualtree_engine;
    dualtree_engine.Init(npr_instance);
    dualtree_engine.Compute(fl::math::LMetric<2>(), result);
  }

  template<typename TableType>
  template<typename WorkSpaceType>
  int NonParametricRegression<boost::mpl::void_>::Core<TableType>::Main(
      WorkSpaceType *ws,
      boost::program_options::variables_map &vm) {
    FL_SCOPED_LOG(NPRegression);
    boost::shared_ptr<TableType> references;
    if (vm.count("references_in")==0) {
      fl::logger->Die()<<"You need to set --references_in option";
    }
    fl::logger->Message()<<"Loading data from "<< vm["references_in"].as<std::string>()<<std::endl;
    ws->Attach(vm["references_in"].as<std::string>(),
        &references);
    if (vm.count("targets_in")) {
      boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> targets;
      ws->Attach(vm["targets_in"].as<std::string>(), &targets);
      boost::shared_ptr<TableType> references1;
      ws->template TieLabels<1>(references, targets, ws->GiveTempVarName(), &references1);
      typename TableType::template IndexArgs<fl::math::LMetric<2> > index_args;
      // at some point we should change that and do the indexing
      // according to the diameter. This is implemented but not tested yet
      index_args.leaf_size=20;
      references1->IndexData(index_args);
      references=references1;
    }
    std::string task=vm["run_mode"].as<std::string>();
    if (task=="train") {
      if (vm.count("predictions_out")>0) {
        fl::logger->Die()<<"--predictions_out cannot be present in train mode";
      }
      if (vm.count("reliabilties_out")>0) {
        fl::logger->Die()<<"--reliabilities_out cannot be present in train mode";
      }
      double ref_split_factor=vm["ref_split_factor"].as<double>();
      std::vector<boost::shared_ptr<TableType> > reference_tables;
      std::vector<boost::shared_ptr<TableType> > query_tables;

      if (ref_split_factor<=0 || ref_split_factor>1) {
        fl::logger->Message()<<"Using all reference for training"<<std::endl;
        reference_tables.push_back(references);
      } else {
        fl::logger->Message()<<"Spliting reference data for training"<<std::endl;
        SplitTable(references, 1, ref_split_factor*references->n_entries(),&reference_tables);
      }
      double query_split_factor=vm["query_split_factor"].as<double>();
      if (vm.count("queries_in")) {
        boost::shared_ptr<TableType> queries;
        ws->Attach(vm["queries_in"].as<std::string>(), &queries);
        if (query_split_factor>=0 && query_split_factor<=1) {
          SplitTable(queries, 1, query_split_factor*queries->n_entries(), &query_tables);   
        } else {
          fl::logger->Message()<<"Using all queries for training"<<std::endl;
          query_tables.push_back(queries);
        } 
      } else {
        if (query_split_factor<0 || query_split_factor>1) {
          fl::logger->Die()<<"If no --queries_in is given the 0<= --query_split_factor <=1";
        } else {
          SplitTable(references, 1, query_split_factor*references->n_entries(), &query_tables);   
        }
      }
      if (query_split_factor>=0 && query_split_factor<=1) {
        SplitTable(references, 1, query_split_factor*references->n_entries(), &query_tables);   
      } else {
        if (vm.count("queries_in")==0) {
          fl::logger->Die()<<"--query_split_factor(="<<query_split_factor
            <<") must be between 0 and 1";
        } else {
          query_tables.resize(1);
                 }
      }

      fl::logger->Message()<<"Training the nonparametric regression"<<std::endl;
      fl::data::MonolithicPoint<double> bandwidths;
      bandwidths.Init(references->n_attributes());
      double initial_bandwidth=vm["bandwidths_init"].as<double>();
      if (initial_bandwidth!=-1) {
        bandwidths.SetAll(initial_bandwidth);
      }
      if (vm.count("bandwidths_in")) {
        // bad design
        // we need to set it to a value different to -1 so that 
        // the trainer.InitBandwidthsFromData();
        // will not be used
        initial_bandwidth=0;
        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> bandwidths_in;
        ws->Attach(vm["bandwidths_in"].as<std::string>(), &bandwidths_in);
        typename WorkSpaceType::DefaultTable_t::Point_t point;
        bandwidths_in->get(0, &point);
        bandwidths.Copy(point.template dense_point<double>());
      }      

      double mse=0;
      for(index_t i=0; i<query_tables.size(); ++i) {
        fl::logger->Message()<<"Training subset "<<i
          <<", references="<< reference_tables[i]->n_entries()
          <<", queries="<<query_tables[i]->n_entries()<<std::endl;
        typename NonParametricRegression<TableType>::Trainer trainer;
        trainer.set_references(reference_tables[i].get());
        trainer.set_queries(query_tables[i].get());
        trainer.set_bandwidths(bandwidths);
        if (initial_bandwidth==-1) {
          fl::logger->Message()<<"Initializing bandwidths from data"<<std::endl;
          trainer.InitBandwidthsFromData();
        }
        trainer.set_lbfgs_rank(vm["lbfgs_rank"].as<int>());
        trainer.set_num_of_line_searches(vm["max_num_line_searches"].as<int>());
        trainer.set_iterations(vm["iterations"].as<int>());
        trainer.set_iteration_chunks(vm["iteration_chunks"].as<int>());
        trainer.set_eta0(vm["eta0"].as<double>());
        std::string train_algorithm=vm["train_algorithm"].as<std::string>();
        if (train_algorithm=="lbfgs") {
          fl::logger->Message()<<"Started Training with LBFGS method"<<std::endl;
          mse=trainer.LbfgsTrain();
        } else {
          if (train_algorithm=="stoc") {
            fl::logger->Message()<<"Started Training with SGD method"<<std::endl;
            mse=trainer.StochasticTrain();
          }
        }
        bandwidths.Copy(trainer.get_bandwidths());
      }
      mse/=query_tables.size();
      if (vm.count("mse_out")!=0) {
        fl::logger->Message()<<"Exporting Mean Square Error"<<std::endl;
        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> table;
        ws->Attach(vm["mse_out"].as<std::string>(),
            std::vector<index_t>(1,1),
            std::vector<index_t>(),
            1,
            &table);
        typename WorkSpaceType::DefaultTable_t::Point_t point;
        table->get(0, &point);
        point.set(0, mse);
        ws->Purge(table->filename());
        ws->Detach(table->filename());
      }
      fl::logger->Message()<<"Exporting bandwidths"<<std::endl;      
      if (vm["bandwidths_out"].as<std::string>()!="") {
        fl::logger->Message()<<"Exporting bandwidths to "<<
          vm["bandwidths_out"].as<std::string>()<<std::endl;
        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> table_out;
        ws->Attach(vm["bandwidths_out"].as<std::string>(),
            std::vector<index_t>(1,bandwidths.size()),
            std::vector<index_t>(),
            1,
            &table_out);
        typename WorkSpaceType::DefaultTable_t::Point_t point;
        table_out->get(0, &point);
        point.template dense_point<double>().CopyValues(bandwidths);
        // We need to do this scaling, because the paper that 
        // describes nonparametric regression uses as 
        // bandwidth 2*h^2, but when we do the evaluation
        // we use our kde that uses \sigma^2
        fl::la::SelfScale(fl::math::Pow<double, 1,2>(2), &point);
        ws->Purge(vm["bandwidths_out"].as<std::string>());
        ws->Detach(vm["bandwidths_out"].as<std::string>());
      }
    } else {
      if (task=="eval") {
        if (vm["data_table_out"].as<std::string>()!="") {
          fl::logger->Die()<<"--data_table_out cannot be present in eval mode";
        }
        if (vm["queries_in"].as<std::string>()=="") {
          fl::logger->Die()<<"You must provide a query set through option "
            "--queries_in"<<std::endl;
        }
        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> bandwidths_table;
        std::string bandwidths_in=vm["bandwidths_in"].as<std::string>();
        if (bandwidths_in=="") {
          fl::logger->Die()<<"You must provide --bandwidths_in in eval mode";
        }
        ws->Attach(bandwidths_in, &bandwidths_table);
        typename WorkSpaceType::DefaultTable_t::Point_t bandwidths1, bandwidths;
        bandwidths_table->get(0, &bandwidths1);
        bandwidths.Copy(bandwidths1);
        for(index_t i=0; i<bandwidths.size(); ++i) {
          if (bandwidths[i]==0) {
            fl::logger->Die()<<"Dimension ("<<i<<
              ") of bandwidths is 0";
          }
          if (bandwidths[i]<1e-10) {
            fl::logger->Warning()<<"Bandwidth too low ("
             <<bandwidths[i]<< ") possible numerical instabillities"<<std::endl;
          }
          bandwidths.set(i, 1.0/bandwidths[i]);          
        }

        boost::shared_ptr<TableType> queries;
        ws->Attach(vm["queries_in"].as<std::string>(), &queries);
        boost::shared_ptr<TableType> scaled_queries;
        std::string scaled_queries_name=ws->GiveTempVarName();
        ws->Attach(scaled_queries_name,
            queries->dense_sizes(),
            queries->sparse_sizes(),
            0,
            &scaled_queries);
        for(index_t i=0; i< queries->n_entries(); ++i) {
          typename TableType::Point_t point1, point2;
          queries->get(i, &point1);
          fl::la::DotMul<fl::la::Init>(bandwidths.template dense_point<double>(), point1, &point2);
          scaled_queries->push_back(point2);
        }
        ws->Purge(scaled_queries_name);
        ws->Detach(scaled_queries_name);
        ws->IndexTable(scaled_queries_name, 
                       "l2",
                       "",
                       20);
        boost::shared_ptr<TableType> references;
        ws->Attach(vm["references_in"].as<std::string>(), &references);        

        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> targets;
        bool is_targets=false;
        if (vm.count("targets_in")) {
          ws->Attach(vm["targets_in"].as<std::string>(), &targets);
          is_targets=true;
          if (targets->n_entries()!=references->n_entries()) {
            fl::logger->Die()<<"--targets_in and --references_in must have the same "
              "number of points";
          }
        };
        
        boost::shared_ptr<TableType> scaled_references;
        std::string scaled_references_name=ws->GiveTempVarName();
        ws->Attach(scaled_references_name,
            references->dense_sizes(),
            references->sparse_sizes(),
            0,
            &scaled_references);
        
        for(index_t i=0; i< queries->n_entries(); ++i) {
          typename TableType::Point_t point1, point2;
          references->get(i, &point1);
          fl::la::DotMul<fl::la::Init>(bandwidths.template dense_point<double>(), point1, &point2);
          if (is_targets) {
            point2.meta_data(). template get<1>()=targets->get(i, (index_t)0);
          } 
          scaled_references->push_back(point2);
        }
        ws->Purge(scaled_references_name);
        ws->Detach(scaled_references_name);
        ws->IndexTable(scaled_references_name, 
                       "l2",
                       "",
                       20);
        typename NonParametricRegression<TableType>::Predictor predictor;
        std::vector<double> predictions;
        std::vector<double> reliabilities;
        predictor.set_references(scaled_references.get()); 
        NprResult<std::vector<double> > result;
        predictor.set_relative_error(vm["relative_error"].as<double>());
        predictor.Predict(scaled_queries.get(), &result); 
        if (vm.count("predictions_out")) {
          std::string predictions_out=vm["predictions_out"].as<std::string>();
          boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> 
            predictions_table;
          ws->Attach(predictions_out,
              std::vector<index_t>(1,1),
              std::vector<index_t>(),
              queries->n_entries(),
              &predictions_table);
          result.GetPredictions<1>(predictions_table.get()); 
          ws->Purge(predictions_out);
          ws->Detach(predictions_out);
        }
        if (vm.count("mse_out")) {
          typename WorkSpaceType::DefaultTable_t predictions_table;
           predictions_table.Init("",
              std::vector<index_t>(1,1),
              std::vector<index_t>(),
              queries->n_entries());
          result.GetPredictions<1>(&predictions_table); 
          double mse=0;
          double norm=0;
          typename TableType::Point_t q_point;
          for(index_t i=0; i<predictions_table.n_entries(); ++i) {
            queries->get(i, &q_point);
            double val=q_point.meta_data(). template get<1>();
            double error=predictions_table.get(i, index_t(0))-val;
            mse+= error*error;
            norm+=val*val;
          }
          boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> mse_table;
          ws->Attach(vm["mse_out"].as<std::string>(),
              std::vector<index_t>(1, 1),
              std::vector<index_t>(),
              1,
              &mse_table);
          if (norm==0) {
            fl::logger->Warning()<<"Query data do not have target values so "
              "Mean square error cannot be computed";
          } else {
            mse=100*sqrt(mse/norm);
            mse_table->set(0, 0, mse);
          }
          ws->Purge(mse_table->filename());
          ws->Detach(mse_table->filename());
        }

        if (vm.count("reliabilities_out")>0) {
          std::string reliabilities_out=vm["reliabilities_out"].as<std::string>();
          boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> 
            reliabilities_table; 
          ws->Attach(reliabilities_out,
              std::vector<index_t>(1,1),
              std::vector<index_t>(),
              queries->n_entries(),
              &reliabilities_table);
          fl::logger->Message()<<"Exporting reliabilities to: "<<reliabilities_out<<std::endl;
          result.GetDensities<1>(reliabilities_table.get());
          fl::logger->Message()<<"Finished exporting reliabilities"<<std::endl;
          ws->Purge(reliabilities_out);
          ws->Detach(reliabilities_out); 
        }
      } else {
        fl::logger->Die()<<"This task ("<<task<<") is not supported";
      }
    }
    return 1;
  }
  
  template<typename TableType>
  void NonParametricRegression<boost::mpl::void_>::Core<TableType>::SplitTable(
      boost::shared_ptr<TableType> table,
      index_t n_splits, 
      index_t new_table_max_size,
      std::vector<boost::shared_ptr<TableType> > *tables) {
    
    if (new_table_max_size>table->n_entries()) {
      fl::logger->Die()<<"You want to split a table ("
        << table->n_entries() <<" elements) to smaller ones"
       << " but instead you requested the new size to be (" 
       << new_table_max_size
       << " entries)"<<std::endl;
    }
    tables->resize(n_splits);
    std::vector<index_t> nodes;
    std::vector<index_t> leafs;
    std::vector<double> node_diameters;
    std::vector<double> leaf_diameters; 
    table->ComputeStatsUpToLevel(&nodes, &leafs, &node_diameters, &leaf_diameters);
    index_t best_level=-1;
    index_t leafs_so_far=0;
    double sample_points_per_node=1;
    for(unsigned int i=0; i<nodes.size(); ++i) {
      leafs_so_far+=leafs[i];
      if (leafs_so_far+nodes[i]>=new_table_max_size) {
        best_level=i;
        break;
      }
    }
    // in case we went all the way to the bottom of the tree 
    // then we have to find out how many points we have to sample from every
    // node
    if (best_level==-1) {
      best_level=nodes.size();
      sample_points_per_node=new_table_max_size/leafs_so_far;
    }
    for(unsigned int i=0; i<n_splits; ++i) {
      tables->operator[](i).reset(new TableType());
      table->RestrictTableToSamplesUpToLevel(best_level,
            sample_points_per_node, tables->operator[](i).get()); 
    }
  }

  template <typename WorkSpaceType, typename BranchType>
  int NonParametricRegression<boost::mpl::void_>::Main(
      WorkSpaceType *ws, const std::vector<std::string> &args) {
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "references_in",
      boost::program_options::value<std::string>(),
      "the reference data "
    )(
      "targets_in",
      boost::program_options::value<std::string>(),
      "if your target values (dependent variable) is not encoded in the "
      "--references_in then you should provide them with this option"
    )(
      "queries_in",
      boost::program_options::value<std::string>()->default_value(""),
      "the set of query points for making predictions or training"
    )(
      "bandwidths_out", 
      boost::program_options::value<std::string>()->default_value(""),
      "the bandwidths for scaling the data"
    )(
      "bandwidths_in",
      boost::program_options::value<std::string>(),
      "the bandwidths for evaluation"
    )(
      "bandwidths_init",
      boost::program_options::value<double>()->default_value(-1),
      "the initial bandwidth for all variables. if you set it to -1, then "
      "it will initialize the bandwidths from the maximum  "
      "absolute value in every dimension"
    )(
      "data_table_out",
      boost::program_options::value<std::string>()->default_value(""),
      "the reference table scaled after reaching the optimal"
    )(
      "predictions_out",
      boost::program_options::value<std::string>(),
      "the predictions of the query points"
    )(
      "reliabilities_out",
      boost::program_options::value<std::string>(),
      "table that contains the reliabilties of the predictions"
    )(
      "lbfgs_rank", 
      boost::program_options::value<int>()->default_value(5),
      "number of basis vectors (history) for the lbfgs"
    )(
      "max_num_line_searches",
      boost::program_options::value<int>()->default_value(10),
      "maximum number of lbfgs line searches"
    )(
      "iterations",
      boost::program_options::value<int>()->default_value(100),
      "number of iterations for lbfgs training"
    )(
      "iteration_chunks",
      boost::program_options::value<int>()->default_value(1),
      "iteration intervals for reporting error"
    )(
      "train_algorithm",
      boost::program_options::value<std::string>()->default_value("stoc"),
      "this option can be either stoc for stochastic gradient descent or "
      "lbfgs for LBFGS"
    )(
      "eta0",
      boost::program_options::value<double>()->default_value(1.0),
      "in case you choose stochastic gradient descent for training, you can "
      "set the initial eta0 by setting this option"
    )(
      "relative_error",
      boost::program_options::value<double>()->default_value(0.1),
      "relative error for computing reliability"  
    )(
      "mse_out",
      boost::program_options::value<std::string>(),
      "table to export the normalized mean square error"
    )(
      "ref_split_factor",
      boost::program_options::value<double>()->default_value(0.8),
      "percentage of the reference data to use for training as reference"
    )(
      "query_split_factor",
      boost::program_options::value<double>()->default_value(0.1),
      "percentage of the reference data to use for training as query"  
    )(
      "run_mode", 
      boost::program_options::value<std::string>()->default_value("train"),
      "run mode can be either train (for training the regression) or "
      "eval (for predicting the value of a point)." 
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
      return true;
    }

    return BranchType::template BranchOnTable<NonParametricRegression<boost::mpl::void_>, 
           WorkSpaceType>(ws, vm);
  }
  template<typename WorkSpaceType>
  void NonParametricRegression<boost::mpl::void_>::Run(
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

}} // namespace fl ml

#endif
