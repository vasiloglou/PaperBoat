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

#ifndef PAPERBOAT_MLPACK_GROUSE_GROUSE_DEFS_H_
#define PAPERBOAT_MLPACK_GROUSE_GROUSE_DEFS_H_
#include "grouse.h"
#include "boost/shared_ptr.hpp"
#include "fastlib/dense/matrix.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/table/default_table.h"
#include "fastlib/table/linear_algebra.h"
#include "fastlib/workspace/workspace.h"
#include "fastlib/dense/linear_algebra.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/util/string_utils.h"
#include "fastlib/workspace/based_on_table_run.h"

namespace fl {namespace ml {

  template<typename TableType>
  template<typename MatrixTableType>
  bool Grouse<TableType>::Update(double eta,
      const typename TableType::Point_t &point, 
      boost::shared_ptr<MatrixTableType> *low_rank_table,
      double *delta_magnitude) {
    
    boost::shared_ptr<MatrixTableType> &low_rank=*low_rank_table;
    index_t k_rank=low_rank->n_attributes();
    fl::dense::Matrix<double> a_matrix;
    a_matrix.Init(k_rank, point.nnz());
    fl::dense::Matrix<double, true> x_vector;
    fl::dense::Matrix<double, true> b_vector;
    b_vector.Init(point.nnz());
    index_t counter=0;
    for(typename TableType::Point_t::iterator it=point.begin();
        it!=point.end(); ++it) {
      typename MatrixTableType::Point_t lpoint;
      low_rank->get(it.attribute(), &lpoint);
      b_vector[counter]=it.value();
      fl::dense::Matrix<double, true> vec;
      a_matrix.MakeColumnVector(counter, &vec);
      vec.CopyValues(lpoint.template dense_point<double>());
      counter++;
    }   
    success_t success;
    fl::dense::ops::LeastSquareFitTrans<fl::la::Init>(
        b_vector, a_matrix, &x_vector,
        &success);
    if (success==SUCCESS_FAIL) {
      return false;
    }
    double x_norm=fl::la::LengthEuclidean(x_vector);
    fl::dense::Matrix<double, true> p_vector;
    p_vector.Init(low_rank->n_entries());
    // we need to make a multiplication function
    // this one is going to be a little slow
    typename MatrixTableType::Point_t p;
    for(index_t i=0; i<low_rank->n_entries(); ++i) {
      low_rank->get(i, &p);
      p_vector.set(i,
          fl::la::Dot(p, x_vector));
    }
//  fl::la::Mul<fl::la::Init>(low_rank, x_vector, &p_vector);    
    std::vector<std::pair<index_t, double> > residual;
    double residual_norm=0;
    double p_norm=fl::la::LengthEuclidean(p_vector);
    counter=0;
    for(typename TableType::Point_t::iterator it=point.begin(); 
        it!=point.end(); ++it) {
      double res=it.value()-p_vector[counter];
      residual.push_back(std::make_pair(it.attribute(), res));
      residual_norm+=res*res;
      counter++;
    }   
    residual_norm=sqrt(residual_norm);
    
    double sigma=residual_norm * p_norm;
    if (sigma<1e-6) {
      return true;
    }
    double w1=(cos(sigma*eta)-1)/(p_norm*x_norm);
    double w2=sin(sigma*eta)/(residual_norm*x_norm);
    fl::la::SelfScale(w1 ,&p_vector);
    for(std::vector<std::pair<index_t, double> >::iterator it=residual.begin();
          it!=residual.end(); ++it) {
       p_vector[it->first]+=w2*it->second;  
    } 
    double subspace_magnitude=0;
    if (delta_magnitude!=NULL) {
      *delta_magnitude=0;
    }
    for(index_t i=0; i<low_rank->n_entries(); ++i) {
      typename fl::ws::WorkSpace::MatrixTable_t::Point_t lpoint;
      low_rank->get(i, &lpoint);
      for(size_t k=0; k<lpoint.size(); ++k) {
        double value=x_vector[k]*p_vector[i];
        subspace_magnitude+=lpoint[k]*lpoint[k];
        lpoint.set(k, lpoint[k]+value);
        if (delta_magnitude!=NULL) {
          *delta_magnitude+=value*value;
        }
      }
    }
    *delta_magnitude=sqrt(*delta_magnitude);
    return true;
  }

  template<typename WorkSpaceType>
  template<typename TableType>
  void Grouse<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
    FL_SCOPED_LOG(Grouse);
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "references_in",
      boost::program_options::value<std::string>(),
      "a csv list of input files"
    )(
      "references_prefix_in",
      boost::program_options::value<std::string>(),
      "the reference data prefix"
    )(
      "references_num_in",
      boost::program_options::value<index_t>(),  
      "number of references file with the prefix defined above"
    )(
      "k_rank",
      boost::program_options::value<int32>(),
      "the rank of the factorization"
    )(
      "low_rank_out",
      boost::program_options::value<std::string>(),
      "the low rank matrix generated by the algorithm"
    )(
      "iterations",
      boost::program_options::value<int32>()->default_value(5),
      "number of iterations over the data"  
    )(
      "eta0",
      boost::program_options::value<double>()->default_value(1.0),
      "the initial eta (stepsize) )of the gradient descent"
    )(
      "stepsize_amortization",
      boost::program_options::value<std::string>()->default_value("constant"),
      "constant: keeps stepsize (eta) to its initial value (eta0), ideal for changing subspaces\n"
      "one_over_n: reduces eta as 1/n, where n is the iteration number"
    )(
      "deltas_out",
      boost::program_options::value<std::string>(),
      "logs the magnitude of every update on the subspace"  
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
    int32 k_rank=0;
    if (vm.count("k_rank")==0) {
      fl::logger->Die()<<"--k_rank is required";
    }
    k_rank=vm["k_rank"].as<int32>();
    index_t iterations=vm["iterations"].as<int32>();
    std::vector<std::string> filenames
      =fl::ws::GetFileSequence("references", vm);
    boost::shared_ptr<fl::ws::WorkSpace::DefaultTable_t> low_rank;
    boost::shared_ptr<TableType> dummy_table;
    index_t dimension=0;
    ws_->GetTableInfo(filenames[0],
        NULL, 
        &dimension,
        NULL,
        NULL);
    fl::ws::WorkSpace::DefaultTable_t random_table;
    fl::logger->Message()<<"Creating an orthonormal initial matrix";
    random_table.Init("", 
        std::vector<index_t>(1, k_rank),
        std::vector<index_t>(),
        dimension);
    typename fl::ws::WorkSpace::DefaultTable_t::Point_t p;
    for(index_t i=0; i<random_table.n_entries(); ++i) {
      random_table.get(i, &p);
      p.SetRandom(0,1);
    }
    fl::ws::WorkSpace::DefaultTable_t r_table;
    ws_->Attach(
        vm.count("low_rank_out")==0?
        ws_->GiveTempVarName():vm["low_rank_out"].as<std::string>(),
        std::vector<index_t>(1, k_rank),
        std::vector<index_t>(),
        dimension,
        &low_rank);
    low_rank->SetAll(0.0);
    fl::logger->Message()<<"QR orthonormalization in progress";
    fl::table::QR(random_table, low_rank.get(), &r_table);
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> deltas_table;
    //find out how many deltas you are going to output
    index_t total_deltas=0;
    for(size_t i=0; i<filenames.size(); ++i) {
      index_t n_entries=0;
      ws_->GetTableInfo(filenames[i],
        &n_entries, 
        NULL,
        NULL,
        NULL);
      total_deltas+=n_entries;
    }
    total_deltas*=iterations;
    if (vm.count("deltas_out")>0) {
      ws_->Attach(vm["deltas_out"].as<std::string>(),
          std::vector<index_t>(1,1),
          std::vector<index_t>(),
          total_deltas,
          &deltas_table);
    }
    fl::logger->Message()<<"Initial orthonormal basis created";
    double eta0=vm["eta0"].as<double>();
    double eta=eta0;
    std::string amortization_schedule=vm["stepsize_amortization"].as<std::string>();
    index_t processed_points=0;
    index_t ignored_points=0;
    index_t counter=0;
    for(int32 i=0; i<iterations; ++i) {
      if (amortization_schedule=="constant") {
        
      } else {
        if (amortization_schedule=="one_over_n") {
          eta=eta0/fl::math::Pow<double, 2,3>(1+i);
        } else {
          fl::logger->Die()<<"Do not know how to handle --stepsize_amortization="
            +amortization_schedule;
        }
      }
      fl::logger->Message()<<"iteration="<<i<<" in progress";
      for(size_t i=0; i<filenames.size(); ++i) {
        boost::shared_ptr<TableType> references;
        ws_->Attach(filenames[i], &references);
        typename TableType::Point_t point;
        for(index_t j=0; i<references->n_entries(); ++j) {
          double delta=0;
          references->get(j, &point);
          ignored_points+=Grouse<TableType>::Update(eta, 
              point, &low_rank, &delta)?0:1;
          if (deltas_table.get()!=NULL) {
            deltas_table->set(counter, 0, delta);
            counter++;
          }
        }
        processed_points+=references->n_entries();
        fl::logger->Message()<<"Processed "
          <<processed_points<<" so far, ignored "<<ignored_points;
        ws_->Purge(filenames[i]);
        ws_->Detach(filenames[i]);
      }
    }
    fl::logger->Message()<<"Finished iterations";
    if (vm.count("low_rank")>0) {
      fl::logger->Message()<<"Exporting low_rank table in ("
          <<vm["low_rank"].as<std::string>()<<")";
    }
    ws_->Purge(low_rank->filename());
    ws_->Detach(low_rank->filename());
    if (deltas_table.get()!=NULL) {
      ws_->Purge(deltas_table->filename());
      ws_->Detach(deltas_table->filename());
    }
  }

  template<typename WorkSpaceType>
  int Grouse<boost::mpl::void_>::Run(
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
  Grouse<boost::mpl::void_>::Core<WorkSpaceType>::Core(
     WorkSpaceType *ws, const std::vector<std::string> &args) :
   ws_(ws), args_(args)  {}

}}
#endif

