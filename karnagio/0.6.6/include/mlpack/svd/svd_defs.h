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
/**
 * @file svd_defs.cc
 *
 * This file implements command line interface for the QUIC-SVD
 * method. It approximate the original matrix by another matrix
 * with smaller dimension to a certain accuracy degree specified by the
 * user and then make SVD decomposition in the projected supspace.
 *
 * Run with --help for more usage.
 *
 * @see svd.h
 */

#ifndef FL_LITE_MLPACK_SVD_SVD_DEFS_H
#define FL_LITE_MLPACK_SVD_SVD_DEFS_H

#include <string>
#include "fastlib/util/string_utils.h"
#include "boost/program_options.hpp"
#include "fastlib/dense/matrix.h"
#include "fastlib/data/multi_dataset.h"
#include "svd.h"
#include "fastlib/workspace/task.h"
#include "fastlib/workspace/based_on_table_run.h"
#include "fastlib/table/linear_algebra.h"
#include "fastlib/util/timer.h"
#include "mlpack/random_projections/random_projections_defs.h"


template<typename WorkSpaceType>
template<typename TableType1>
void fl::ml::Svd<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType1&) { 
  FL_SCOPED_LOG(Svd);
  ////////// READING PARAMETERS AND LOADING DATA /////////////////////
  boost::program_options::options_description desc("Available options");
  desc.add_options()
  ("help", "Display help on SVD")
  ("references_prefix_in",
   boost::program_options::value<std::string>(),
   "the reference data prefix"
  )
  ( "references_num_in",
     boost::program_options::value<int32>(),
     "number of references file with the prefix defined above"
  )
  ("references_in",
    boost::program_options::value<std::string>(),
    "the reference data comma separated. Each file is a time series"
    " eg --references_in=myfile,yourfile,hisfile"
  )
  ("algorithm", boost::program_options::value<std::string>()->default_value("covariance"),
   "covariance : LAPACK implementation on the covariance matrix\n"
   "sgdl       : compute low rank factorization with stochastic gradient descent\n"
   "             only the right factor will be orthogonal. \n"
   "sgdr       : compute low rank factorization with stochastic gradient descent\n"
   "             only the left factor will be orthogonal. \n"
   "lbfgs      : same as before but with lbfgs\n"
   "randomized : use a projection to a gaussian random matrix and then do svd\n"
   "concept    : uses the concept decomposition from Dhillon \"Concept Decompositions "
   "             for Large Sparse Text Data using Clustering\""
  )
  ("step0", 
   boost::program_options::value<double>()->default_value(1.0),
   "This is the step0 that sgdl, sgdr, the stochastic gradient decent"
  )
  ("randomize",
   boost::program_options::value<bool>()->default_value(true),
   "The stochastic gradient descent requires the data to be randomized. "
   "If you think that your data are not randomized then set this flag to true. "
   "It might slow down the performance but it will improve the results."
  )
  ("l2normalize", 
   boost::program_options::value<bool>()->default_value(true),
   "For svd through concept vector decomposition you need to have your data "
   "L2 mormalized. If your data is not L2 normalized then set this flag true"
  )("col_mean_normalize",
    boost::program_options::value<bool>()->default_value(false),
    "if you set this flag to true it will normalize the columns so that "
    "they have zero mean. In that case SVD is equivalent to PCA"
  )("n_epochs",
   boost::program_options::value<index_t>()->default_value(5),
   "number of epochs of stochastic gradient descent. "
   " each epoch finishes after n_iterations are completed"
  )
  ("n_iterations",
   boost::program_options::value<index_t>()->default_value(10),
   "number of iterations. Each iteration finishes after a pass over all "
   "reference table data."
  )
  ("rec_error", 
   boost::program_options::value<bool>()->default_value(false),
   "If you set this flag true then the reconstruction error will be computed"   
  )
  ("svd_rank", boost::program_options::value<int>()->default_value(5),
   "The algorithm will find up to the svd_rank first components")
  ("smoothing_p", boost::program_options::value<int>()->default_value(2),
   "when doing randomized svd you need to smooth the matrix by "
   "mutliplying it with XX' p times")
  ("lsv_out",
   boost::program_options::value<std::string>(),
   "The output file for the left singular vectors (each column is a singular vector).")
  ("lsv_prefix_out",
   boost::program_options::value<std::string>(),
   "If you have multiple references files you need to have multiple lsv files")
  ("lsv_num_out", 
   boost::program_options::value<int32>(),
   "the number of lesv files")
  ("sv_out",
   boost::program_options::value<std::string>(),
   "The output file for the singular values.")
  ("rsv_out",
   boost::program_options::value<std::string>(),
   "The output file for the transposed right singular vectors (each row is a singular vector).");


  boost::program_options::variables_map vm;
  boost::program_options::command_line_parser clp(
      fl::ws::MakeArgsFromPrefix(args_, ""));
  clp.style(boost::program_options::command_line_style::default_style
            ^ boost::program_options::command_line_style::allow_guessing);
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


  typedef TableType1 Table_t;
  // The reference data file is a required parameter.
  fl::logger->Message() << "Loading reference tables from "<< std::endl;
  boost::shared_ptr<TableType1> references_table; 
  std::vector<std::string> references_filenames=fl::ws::GetFileSequence("references", vm);

  if (vm["col_mean_normalize"].as<bool>()==true) {
    std::vector<std::string> new_filenames;
    fl::logger->Message()<<"Performing zero mean normalization over the columns"
      <<std::endl;
    for(size_t k=0; k<references_filenames.size(); ++k) {
      std::string filename=ws_->GiveTempVarName();
      boost::shared_ptr<TableType1> new_references_table; 
      ws_->Attach(references_filenames[k], &references_table);
      ws_->Attach(filename,
          references_table->dense_sizes(),
          references_table->sparse_sizes(),
          0,
          &new_references_table);
      std::vector<double> means, variances;
      references_table->AttributeStatistics(&means, &variances);
      typename TableType1::Point_t point, point1;
      for(index_t i=0; i<references_table->n_entries(); ++i) {
        references_table->get(i, &point);
        point1.Copy(point);
        for(size_t j=0; j<means.size(); ++j) {
          point1.set(j, point[j]-means[j]);
        }
        new_references_table->push_back(point1);
      }
      new_filenames.push_back(filename);
      ws_->Purge(references_filenames[k]);
      ws_->Detach(references_filenames[k]);
      ws_->Purge(filename);
      ws_->Detach(filename);
    }
    references_filenames=new_filenames;
  }
  fl::logger->Message() << "Loading completed" << std::endl;

  boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> left_table;
  boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> right_trans_table;
  boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> sv_table;

  std::vector<std::string> lsv_filenames;
  if (vm.count("lsv_out") || vm.count("lsv_prefix_out")) {
    lsv_filenames=fl::ws::GetFileSequence("lsv", vm); 
  } else {
    for(size_t i=0; i<references_filenames.size(); ++i) {
      lsv_filenames.push_back(ws_->GiveTempVarName());
    }
  }
  std::string sv_file;
  if (vm.count("sv_out")) {
    sv_file=vm["sv_out"].as<std::string>();
  } else {
    sv_file=ws_->GiveTempVarName();
  }
  std::string rsv_trans_file;
  if (vm.count("rsv_out")) { 
    rsv_trans_file=vm["rsv_out"].as<std::string>();
  } else {
    rsv_trans_file=ws_->GiveTempVarName(); 
  }

  int svd_rank = vm["svd_rank"].as<int>();
  index_t n_attributes=0;
  ws_->GetTableInfo(references_filenames[0], NULL, &n_attributes, NULL, NULL);
  if (svd_rank>n_attributes) {
    fl::logger->Die()<<"--svd_rank ("<< svd_rank <<") must be less "
      "or equal to the --references_in attributes ("
      <<n_attributes<<")";
  }

  Svd<TableType1> engine;
  fl::util::Timer timer;
  timer.Start();
  if (vm["algorithm"].as<std::string>() == "covariance") {
    fl::logger->Message()<<"Computing SVD with LAPACK "
      "on the covariance matrix"<<std::endl;
    engine.template ComputeFull<WorkSpaceType, TableType1, typename WorkSpaceType::MatrixTable_t>(
                ws_,
                svd_rank,
                references_filenames,
                &sv_file,
                &lsv_filenames,
                &rsv_trans_file);
    fl::logger->Message() << "Finished computing SVD" << std::endl;
  } else {
    if (StringStartsWith(vm["algorithm"].as<std::string>(), "sgd")) {
      double step0=vm["step0"].as<double>();
      index_t n_epochs=vm["n_epochs"].as<index_t>();
      index_t n_iterations=vm["n_iterations"].as<index_t>();
      bool randomize=vm["randomize"].as<bool>();
      boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> temp_left,
               temp_right_trans;
      ws_->Attach(ws_->GiveTempVarName(),
          std::vector<index_t>(1, svd_rank),
          std::vector<index_t>(),
          references_table->n_entries(),
          &temp_left);
      ws_->Attach(ws_->GiveTempVarName(),
          std::vector<index_t>(1, svd_rank),
          std::vector<index_t>(),
          references_table->n_attributes(),
          &temp_right_trans);

      engine.ComputeLowRankSgd(*references_table,
                               step0,
                               n_epochs,
                               n_iterations,
                               randomize,
                               temp_left.get(),
                               temp_right_trans.get()); 
      if (vm["algorithm"].as<std::string>()=="sgdl") {
        boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> temp_left1;
        ws_->Attach(ws_->GiveTempVarName(),
            std::vector<index_t>(1, temp_right_trans->n_attributes()),
            std::vector<index_t>(),
            temp_right_trans->n_attributes(),
            &temp_left1);
        engine.template ComputeFull<WorkSpaceType, 
          typename WorkSpaceType::MatrixTable_t, 
          typename WorkSpaceType::MatrixTable_t>(
                   ws_, 
                   svd_rank,
                   std::vector<std::string>(1, temp_right_trans->filename()),
                   &sv_file,
                   &lsv_filenames,
                   &rsv_trans_file); 
        fl::table::Mul<fl::la::NoTrans, 
          fl::la::NoTrans>(*temp_left, *temp_left1, left_table.get());
        ws_->Purge(temp_left->filename());
        ws_->Purge(temp_right_trans->filename());
        ws_->Purge(temp_left1->filename());
      } else {
        if (vm["algorithm"].as<std::string>()=="sgdr") {
          boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> temp_right_trans1;
          std::string temp_right_trans1_filename=ws_->GiveTempVarName();

          engine.template ComputeFull<WorkSpaceType, 
              typename WorkSpaceType::MatrixTable_t, 
              typename WorkSpaceType::MatrixTable_t>(
                  ws_,
                  svd_rank,
                  std::vector<std::string>(1, temp_left->filename()),
                  &sv_file,
                  &lsv_filenames,
                  &temp_right_trans1_filename); 
          ws_->Attach(temp_right_trans1_filename, &temp_right_trans1);
          fl::table::Mul<fl::la::NoTrans, fl::la::NoTrans>(
              *temp_right_trans,
              *temp_right_trans1, 
              right_trans_table.get());
        } else {
          fl::logger->Die()<<"This option "
            <<vm["algorithm"].as<std::string>() 
            <<" is not supported";
        }
      }    
    } else {
      if (vm["algorithm"].as<std::string>() == "lbfgs") {
        fl::logger->Die()<<"Not supported yet"; 
      } else {
        if (vm["algorithm"].as<std::string>() == "randomized") {
          int smoothing_p=vm["smoothing_p"].as<int>();
          fl::logger->Message()<<"Computing randomized svd"<<std::endl;
          std::vector<std::string> random_projection_args=fl::ws::MakeArgsFromPrefix(args_, "randproj");
          auto arg_map=fl::ws::GetArgumentPairs(random_projection_args);
          if (arg_map.count("--references_in") || arg_map.count("--references_prefix_in")) {
            fl::logger->Die()<<"Please do not define --references_in, leave it to svd";
          } else {
            std::string references_in="--references_in=";
            for(size_t i=0; i<references_filenames.size()-1; ++i) {
              references_in+=references_filenames[i]+",";
            }
            references_in+=references_filenames[references_filenames.size()-1];
            random_projection_args.push_back(references_in);
          }
          std::vector<std::string> projected_references_filenames;
          if (arg_map.count("--projected_out")>0 || 
              arg_map.count("--projected_prefix_out")>0) {
            projected_references_filenames=fl::ws::GetFileSequence("projected", arg_map);
          } else {
            std::string projected_out="--projected_out=";
            for(size_t i=0; i<references_filenames.size(); ++i) {
              projected_references_filenames.push_back(ws_->GiveTempVarName()); 
              projected_out+=projected_references_filenames.back()+",";
            }
            random_projection_args.push_back(projected_out.substr(0, 
                  std::string::npos-1));
          }
          if (arg_map.count("--projection_rank")==0) {
            random_projection_args.push_back("--projection_rank="
                +boost::lexical_cast<std::string>(svd_rank));
          } else {
            fl::logger->Die()<<"You cannot define --projection_rank at this stage";
            if (boost::lexical_cast<int32>(arg_map["--projection_rank"])>n_attributes) {
              fl::logger->Die()<<"--projection_rank ("<< arg_map["--projection_rank"] <<") cannot be more than n_attributes ("
                <<n_attributes<<")";
            }
          }
          fl::ml::RandomProjections<boost::mpl::void_>::Run(ws_, random_projection_args);
          std::vector<std::string> dummy(1, rsv_trans_file);
          engine.template ComputeRandomizedSvd<
              WorkSpaceType, 
              typename WorkSpaceType::MatrixTable_t, 
              typename WorkSpaceType::MatrixTable_t>(
                               ws_,
                               svd_rank,
                               references_filenames,
                               projected_references_filenames,
                               smoothing_p,
                               &sv_file,
                               &lsv_filenames,
                               &dummy);

        } else {
          if (vm["algorithm"].as<std::string>() == "concept") {
            std::vector<double> l2norms(references_table->n_entries());
            if (vm["l2normalize"].as<bool>()==true) {
              fl::logger->Message()<<"L2 normalization of the input "
                "data"<<std::endl;
              typename Table_t::Point_t point;
              for(index_t i=0; i<references_table->n_entries(); ++i) {
                references_table->get(i, &point);
                double norm=fl::la::LengthEuclidean(point);
                l2norms[i]=norm;   
              }
            }
            fl::logger->Message()<<"Computing SVD with Concept "
              "decomposition"<<std::endl;
//            index_t n_iterations=vm["n_iterations"].as<index_t>();
//            double error_change=0;
//            engine.ComputeConceptSvd(*references_table,
//                                     l2norms,
//                                     n_iterations,
//                                     error_change,
//                                     sv_table.get(),
//                                     left_table.get(),
//                                     right_trans_table.get());
          } else {
            fl::logger->Die()<<"This algorithm ("<<
                vm["algorithm"].as<std::string>()
                << ") is not supported";
          }
        }     
      }
    }
  }
  timer.End();
  fl::logger->Message()<<"Svd computation took ("
    <<timer.GetTotalElapsedTimeString()<<") seconds";

  if( vm["rec_error"].as<bool>()==true) {
    double error;
    fl::logger->Message()<<"Computing reconstruction error"<<std::endl;
    fl::logger->Message()<<"referece_table="<<references_table->n_entries()
      <<"x"<<references_table->n_attributes();
    fl::logger->Message()<<"sv_table="<<sv_table->n_entries()
      <<"x"<<sv_table->n_attributes();
    fl::logger->Message()<<"left_table="<<left_table->n_entries()
      <<"x"<<left_table->n_attributes();
    fl::logger->Message()<<"right_trans_table="<<right_trans_table->n_entries()
      <<"x"<<right_trans_table->n_attributes();
    engine.ComputeRecError(*references_table,
                           *sv_table,
                           *left_table,
                           *right_trans_table,
                           &error);
    int truncated_error=static_cast<int>(error*10000);
    fl::logger->Message()<<"The reconstruction error is: "
      <<truncated_error/100.0<<"%"<<std::endl;

  }
  fl::logger->Message() << "Exporting the results of SVD " << std::endl;
  // check if there are point ids on the refences_table
  {
    ws_->Attach(references_filenames[0], &references_table);
    typename TableType1::Point_t p1, p2;
    if (references_table->n_entries()>=2) {
      references_table->get(0, &p1);
      references_table->get(1, &p2);
      if (p1.meta_data(). template get<2>()!=p2.meta_data(). template get<2>()) {
        typename WorkSpaceType::DefaultTable_t::Point_t lpoint;
        typename TableType1::Point_t ref_point;
        for(size_t j=0; j<references_filenames.size(); ++j) {
          ws_->Attach(references_filenames[j], &references_table);
          ws_->Attach(lsv_filenames[j], &left_table);
          for(index_t i=0; i<references_table->n_entries(); ++i) {
            left_table->get(i, &lpoint);
            references_table->get(i, &ref_point);
            lpoint.meta_data().template get<2>()=ref_point.meta_data().template get<2>();
          }
          ws_->Purge(lsv_filenames[j], true);
          ws_->Purge(lsv_filenames[j]);
          ws_->Purge(references_filenames[j]);
          ws_->Detach(references_filenames[j]);
        }
      } else {
        ws_->Purge(references_filenames[0]);
        ws_->Detach(references_filenames[0]);
      }
    }
  }
  fl::logger->Message() << "Finished exporting the results of SVD" << std::endl;
  return ;
}


template<typename WorkSpaceType>
int fl::ml::Svd<boost::mpl::void_>::Run(
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
fl::ml::Svd<boost::mpl::void_>::Core<WorkSpaceType>::Core(
   WorkSpaceType *ws, const std::vector<std::string> &args) :
 ws_(ws), args_(args)  {}


#endif

