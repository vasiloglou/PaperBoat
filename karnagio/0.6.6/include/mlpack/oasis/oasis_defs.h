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

#ifndef PAPERBOAT_MLPACK_OASIS_OASIS_DEFS_H_
#define PAPERBOAT_MLPACK_OASIS_OASIS_DEFS_H_
#include "oasis.h"
#include "boost/shared_ptr.hpp"
#include "boost/tuple/tuple.hpp"
#include "fastlib/dense/matrix.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/table/default_table.h"
#include "fastlib/table/linear_algebra.h"
#include "fastlib/workspace/workspace.h"
#include "fastlib/dense/linear_algebra.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/util/string_utils.h"
#include "fastlib/workspace/based_on_table_run.h"

namespace fl { namespace ml {

  template<typename TableType>
  template<typename MatrixTableType>
  double Oasis<TableType>::L_w(
      const typename TableType::Point_t &point, 
      const typename TableType::Point_t &point_plus, 
      const typename TableType::Point_t &point_minus, 
      const MatrixTableType &w_matrix) {
    double dot_prod_plus=0;
    double dot_prod_minus=0;
    typename MatrixTableType::Point_t w_point;
    for(auto it=point.begin(); it!=point.end(); ++it) {
      w_matrix.get(it.attribute(), &w_point);
      dot_prod_plus+=it.value()*fl::la::Dot(w_point, point_plus);
      dot_prod_minus+=it.value()*fl::la::Dot(w_point, point_minus);
    }
    return std::max(0.0, 1-dot_prod_plus+dot_prod_minus);
  }
  template<typename TableType>
  template<typename MatrixTableType>
  bool Oasis<TableType>::IsItCorrect(
      const typename TableType::Point_t &point, 
      const typename TableType::Point_t &point_plus, 
      const typename TableType::Point_t &point_minus, 
      const MatrixTableType &w_matrix,
      typename MatrixTableType::Point_t *score_point) {
    double dot_prod_plus=0;
    double dot_prod_minus=0;
    typename MatrixTableType::Point_t w_point;
    for(auto it=point.begin(); it!=point.end(); ++it) {
      w_matrix.get(it.attribute(), &w_point);
      dot_prod_plus+=it.value()*fl::la::Dot(w_point, point_plus);
      dot_prod_minus+=it.value()*fl::la::Dot(w_point, point_minus);
    }
    if (score_point!=NULL) {
      score_point->set(0, dot_prod_plus);
      score_point->set(1, dot_prod_minus);
    }
    return dot_prod_plus>dot_prod_minus;
  }

  template<typename TableType>
  template<typename MatrixTableType, typename IntegerTableType>
  void Oasis<TableType>::ComputeTrainPrecision(
        IntegerTableType &triplets_table, 
        TableType &reference_table,
        MatrixTableType &weight_matrix,
        double *score,
        index_t *num_of_triplets, 
        MatrixTableType *score_table) {
    *score=0;
    *num_of_triplets=0;
    typename IntegerTableType::Point_t index_point;
    typename TableType::Point_t point, point_plus, point_minus;
    typename MatrixTableType::Point_t score_point;
    for(index_t j=0; j<triplets_table.n_entries(); ++j) {
      triplets_table.get(j, &index_point);
      if (index_point[0]<0 || index_point[0]>=reference_table.n_entries()) {
        fl::logger->Warning()<<"Triplet ("<<j<<") contains invalid indices ("
          <<index_point[0]<<","
          <<index_point[1]<<","
          <<index_point[2]<<") is invalid, skipping it ";
        continue;
      }
      reference_table.get(index_point[0], &point);
      if (index_point[1]<0 || index_point[1]>=reference_table.n_entries()) {
        fl::logger->Warning()<<"Triplet ("<<j<<") contains invalid indices ("
          <<index_point[0]<<","
          <<index_point[1]<<","
          <<index_point[2]<<") is invalid, skipping it ";
        continue;
      }
      reference_table.get(index_point[1], &point_plus);
      if (index_point[2]<0 || index_point[2]>=reference_table.n_entries()) {
        fl::logger->Warning()<<"Triplet ("<<j<<") contains invalid indices ("
          <<index_point[0]<<","
          <<index_point[1]<<","
          <<index_point[2]<<") is invalid, skipping it ";
        continue;
      }
      reference_table.get(index_point[2], &point_minus);
      if (score_table!=NULL) {
        score_table->get(j, &score_point);
      }
      (*score)+=Oasis<TableType>::IsItCorrect(
             point, 
             point_plus, 
             point_minus, 
             weight_matrix, 
             score_table==NULL?NULL:&score_point);
       (*num_of_triplets)+=1;
    }
  }   

  template<typename TableType>
  template<typename MatrixTableType>
  void Oasis<TableType>::Update(
      const typename TableType::Point_t &point,
      const typename TableType::Point_t &point_plus,
      const typename TableType::Point_t &point_minus, 
      double regularizer_c,
      bool is_nonnegative,
      MatrixTableType *metric_matrix) {
    // compute gradient
    double gradient_l2=0;
    double tau=regularizer_c;
    typename TableType::Point_t point_diff;
    fl::la::Sub<fl::la::Init>(point_minus, point_plus, &point_diff); 
    // check if the difference is dense
    // this test needs to be revised, it is too slow
    if ( false && 1.0*point_diff.nnz()*point.nnz()/(point_diff.size()*point.size())>1e-2) {
      MatrixTableType gradient;
      gradient.Init("", 
         std::vector<index_t>(1, metric_matrix->n_attributes()),
         std::vector<index_t>(),
         metric_matrix->n_entries());
      for(index_t j=0; j<metric_matrix->n_entries(); ++j) {
        auto diff=point_plus[j]-point_minus[j];
        for(index_t i=0; i<metric_matrix->n_entries(); ++i) {
          double value=point[i]*diff;
          // according to the formula we need the transpose
          gradient.set(j, i, value);
          gradient_l2+=value*value;
        }
      }
      tau=std::min(regularizer_c, 
        L_w(point, point_plus, point_minus, *metric_matrix)/gradient_l2);
      typename MatrixTableType::Point_t matrix_point1;
      typename MatrixTableType::Point_t matrix_point2;
      for(index_t i=0; i<metric_matrix->n_entries(); ++i) {
        metric_matrix->get(i, &matrix_point1);
        gradient.get(i, &matrix_point2);
        fl::la::AddExpert(tau, matrix_point2, &matrix_point1);
      }
    } else {
      std::vector<boost::tuple<index_t, index_t, double> > gradient;
      for(auto it1=point.begin(); it1!=point.end(); ++it1) {
        for(auto it2=point_diff.begin(); it2!=point_diff.end(); ++it2) {
          double value=it1.value()*it2.value();
          gradient.push_back(boost::make_tuple(it2.attribute(), it1.attribute(), value));
          gradient_l2+=value*value;
        }
      }
      tau=std::min(regularizer_c, 
        L_w(point, point_plus, point_minus, *metric_matrix)/gradient_l2);
      if (is_nonnegative==false) {
        for(size_t i=0; i<gradient.size(); ++i) {
          auto row=gradient[i].get<0>();
          auto col=gradient[i].get<1>();
          auto val=gradient[i].get<2>();
          auto new_val=tau*val+metric_matrix->get(row, col);
          metric_matrix->set(row, col, new_val);
        }
      } else {
        for(size_t i=0; i<gradient.size(); ++i) {
          auto row=gradient[i].get<0>();
          auto col=gradient[i].get<1>();
          auto val=gradient[i].get<2>();
          auto new_val=tau*val+metric_matrix->get(row, col);
          if (new_val<0) {
            new_val=0;
          }
          metric_matrix->set(row, col, new_val);
        }
      }
    }
  }

  template<typename TableType>
  template<typename MatrixTableType>
  void Oasis<TableType>::RegularizeAndProject(
      double eta,
      bool is_nonnegative,
      double lambda_regularization,
      MatrixTableType *metric_matrix) {
    for(index_t i=0; i<metric_matrix->n_entries(); ++i) {
      for(index_t j=0; j<metric_matrix->n_attributes(); ++j) {
        auto new_value=metric_matrix->get(i, j)/(1+lambda_regularization);
        if (is_nonnegative && new_value<0) {
          new_value=0;
        }
        metric_matrix->set(i, j, new_value);
      }
    }
  }


  template<typename TableType>
  template<typename MatrixTableType>
  double OasisLowRank<TableType>::L_w(
      const typename TableType::Point_t &point, 
      const typename TableType::Point_t &point_plus, 
      const typename TableType::Point_t &point_minus, 
      const MatrixTableType &w_matrix) {
    fl::data::MonolithicPoint<double> w_point;
    fl::data::MonolithicPoint<double> w_point_plus;
    fl::data::MonolithicPoint<double> w_point_minus;
    fl::table::MulPoint(w_matrix, point, &w_point);
    fl::table::MulPoint(w_matrix, point_plus, &w_point_plus);
    fl::table::MulPoint(w_matrix, point_minus, &w_point_minus);
    double dot_prod_plus=fl::la::Dot(w_point, w_point_plus);
    double dot_prod_minus=fl::la::Dot(w_point, w_point_minus);
        
    return std::max(0.0, 1-dot_prod_plus+dot_prod_minus);
  }
  template<typename TableType>
  template<typename MatrixTableType>
  bool OasisLowRank<TableType>::IsItCorrect(
      const typename TableType::Point_t &point, 
      const typename TableType::Point_t &point_plus, 
      const typename TableType::Point_t &point_minus, 
      const MatrixTableType &w_matrix,
      typename MatrixTableType::Point_t *score_point) {
    fl::data::MonolithicPoint<double> w_point;
    fl::data::MonolithicPoint<double> w_point_plus;
    fl::data::MonolithicPoint<double> w_point_minus;
    fl::table::MulPoint(w_matrix, point, &w_point);
    fl::table::MulPoint(w_matrix, point_plus, &w_point_plus);
    fl::table::MulPoint(w_matrix, point_minus, &w_point_minus);
    double dot_prod_plus=fl::la::Dot(w_point, w_point_plus);
    double dot_prod_minus=fl::la::Dot(w_point, w_point_minus);
    if (score_point!=NULL) {
      score_point->set(0, dot_prod_plus);
      score_point->set(1, dot_prod_minus);
    }
    return dot_prod_plus>dot_prod_minus;
  }

  template<typename TableType>
  template<typename MatrixTableType, typename IntegerTableType>
  void OasisLowRank<TableType>::ComputeTrainPrecision(
        IntegerTableType &triplets_table, 
        TableType &reference_table,
        MatrixTableType &weight_matrix,
        double *score,
        index_t *num_of_triplets,
        MatrixTableType *score_table) {
    *score=0;
    *num_of_triplets=0;
    typename IntegerTableType::Point_t index_point;
    typename TableType::Point_t point, point_plus, point_minus;
    typename MatrixTableType::Point_t score_point;
    for(index_t j=0; j<triplets_table.n_entries(); ++j) {
      triplets_table.get(j, &index_point);
      if (index_point[0]<0 || index_point[0]>=reference_table.n_entries()) {
        fl::logger->Warning()<<"Triplet ("<<j<<") contains invalid indices ("
          <<index_point[0]<<","
          <<index_point[1]<<","
          <<index_point[2]<<") is invalid, skipping it ";
        continue;
      }
      reference_table.get(index_point[0], &point);
      if (index_point[1]<0 || index_point[1]>=reference_table.n_entries()) {
        fl::logger->Warning()<<"Triplet ("<<j<<") contains invalid indices ("
          <<index_point[0]<<","
          <<index_point[1]<<","
          <<index_point[2]<<") is invalid, skipping it ";
        continue;
      }
      reference_table.get(index_point[1], &point_plus);
      if (index_point[2]<0 || index_point[2]>=reference_table.n_entries()) {
        fl::logger->Warning()<<"Triplet ("<<j<<") contains invalid indices ("
          <<index_point[0]<<","
          <<index_point[1]<<","
          <<index_point[2]<<") is invalid, skipping it ";
        continue;
      }
      reference_table.get(index_point[2], &point_minus);
      if (score_table !=NULL) {
        score_table->get(j, &score_point);
      }
      (*score)+=OasisLowRank<TableType>::IsItCorrect(
             point, 
             point_plus, 
             point_minus, 
             weight_matrix, 
             score_table==NULL?NULL:&score_point);
          (*num_of_triplets)+=1;
    }
  }   

  template<typename TableType>
  template<typename MatrixTableType>
  void OasisLowRank<TableType>::Update(
      const typename TableType::Point_t &point,
      const typename TableType::Point_t &point_plus,
      const typename TableType::Point_t &point_minus, 
      double regularizer_c,
      bool is_nonnegative,
      MatrixTableType *metric_matrix) {
    // compute gradient
    double gradient_l2=0;
    double tau=regularizer_c;
    typename TableType::Point_t point_diff;
    fl::la::Sub<fl::la::Init>(point_minus, point_plus, &point_diff); 
    typename fl::data::MonolithicPoint<double> w_point;
    fl::table::MulPoint(*metric_matrix, point, &w_point);

    // check if the diff is dense
    // There is something wrong here we have to review it
    if (false && 1.0*point_diff.nnz()*point.nnz()/(point_diff.size()*point.size())>1e-2) {
      MatrixTableType gradient;
      gradient.Init("", 
         std::vector<index_t>(1, metric_matrix->n_attributes()),
         std::vector<index_t>(),
         metric_matrix->n_entries());
      
      for(index_t j=0; j<metric_matrix->n_entries(); ++j) {
        for(auto it=point_diff.begin(); it!=point_diff.end(); ++it) {
          double value=w_point[j]*it.value();
          gradient.set(j, it.attribute(), value);
          gradient_l2+=value*value;
        } 
      }
      tau=std::min(regularizer_c, 
        L_w(point, point_plus, point_minus, *metric_matrix)/gradient_l2);
      typename MatrixTableType::Point_t matrix_point1;
      typename MatrixTableType::Point_t matrix_point2;
      for(index_t i=0; i<metric_matrix->n_entries(); ++i) {
        metric_matrix->get(i, &matrix_point1);
        gradient.get(i, &matrix_point2);
        fl::la::AddExpert(tau, matrix_point2, &matrix_point1);
      }
    } else {
      std::vector<boost::tuple<index_t, index_t, double> > gradient;
      for(index_t j=0; j<w_point.size(); ++j)  {
        for(auto it2=point_diff.begin(); it2!=point_diff.end(); ++it2) {
          double value=w_point[j]*it2.value();
          gradient.push_back(boost::make_tuple(j, it2.attribute(), value));
          gradient_l2+=value*value;
        }
      }
      tau=std::min(regularizer_c, 
        L_w(point, point_plus, point_minus, *metric_matrix)/gradient_l2);
      if (is_nonnegative==false) {
        for(size_t i=0; i<gradient.size(); ++i) {
          auto row=gradient[i].get<0>();
          auto col=gradient[i].get<1>();
          auto val=gradient[i].get<2>();
          auto new_val=tau*val+metric_matrix->get(row, col);
          metric_matrix->set(row, col, new_val);
        }
      } else {
        for(size_t i=0; i<gradient.size(); ++i) {
          auto row=gradient[i].get<0>();
          auto col=gradient[i].get<1>();
          auto val=gradient[i].get<2>();
          auto new_val=tau*val+metric_matrix->get(row, col);
          if (new_val<0) {
            new_val=0;
          }
          metric_matrix->set(row, col, new_val);
        }
      }
    }
  }

  template<typename TableType>
  template<typename MatrixTableType>
  void OasisLowRank<TableType>::RegularizeAndProject(
      double eta,
      bool is_nonnegative,
      double lambda_regularization,
      MatrixTableType *metric_matrix) {
    for(index_t i=0; i<metric_matrix->n_entries(); ++i) {
      for(index_t j=0; j<metric_matrix->n_attributes(); ++j) {
        auto new_value=metric_matrix->get(i, j)/(1+lambda_regularization);
        if (is_nonnegative && new_value<0) {
          new_value=0;
        }
        metric_matrix->set(i, j, new_value);
      }
    }
  
  }

  template<typename WorkSpaceType>
  template<typename TableType>
  void Oasis<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
    FL_SCOPED_LOG(Oasis);
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
      boost::program_options::value<int32>(),  
      "number of references file with the prefix defined above"
    )(
      "triplets_in",
      boost::program_options::value<std::string>(),
      "triplets of points, 3 numbers, the index of the test point, "
      "the index of a positive point, the index of a negative point"
    )(
      "triplets_prefix_in",
      boost::program_options::value<std::string>(),
      "the triplets point prefix"  
    )(
      "triplets_num_in",
      boost::program_options::value<int32>(),
      "number of triplets files with the prefix above"
    )(
      "weight_matrix_in",
      boost::program_options::value<std::string>(),
      "weight matrix W for computing the bilinear simmilarity  x1^T W x2"
    )(
      "weight_matrix_out",
      boost::program_options::value<std::string>(),
      "weight matrix W for computing the bilinear simmilarity  x1^T W x2"
    )(
      "aggressiveness_parameter", 
      boost::program_options::value<double>()->default_value(1.0),
      "the method is using a passive aggressive method. This parameter controls how aggresively to update "
      "the weight matrix given a new triplet"  
    )(
      "iterations",
      boost::program_options::value<int32>()->default_value(1),
      "the number of iterations over the whole data set"  
    )(
      "compute_train_score",
      boost::program_options::value<bool>()->default_value(true),
      "if you want to evaluate the training precision, set this to true"  
    )("randomize_triplets",
      boost::program_options::value<bool>()->default_value(true),
      "if you feel the triplets are not in random order then you should set this flag to true"
    )(
      "weight_matrix_type",
      boost::program_options::value<std::string>()->default_value("asymmetric"),
      "The weight matrix can be different types\n"
      "   asymmetric: just as described in the paper, it is NxN and it grows quadratically to the "
      "dimensionality of the data\n"
      "   symmetric: here again it is NxN and it grows quadratically to the dimensionality of the "
      "data points\n"
      "  psd: is positive semidefinite. In that case W can be written as W = W' W'T where W' is Nxk "
      " so you have to define k through the flag --weight_matrix_rank. In reality the output weight "
      "matrix is W' which grows linearly with regard to the data dimensionality"
    )(
      "weight_matrix_rank",
      boost::program_options::value<int32>(),
      "if you choose to use psd as your weight matrix and you don't provide a seed matrix to start "
      "the training then you need to provide the rank of that matrix"
    )(
      "scores_out",
      boost::program_options::value<std::string>(),
      "if you want to get this actual scores computed you can provide a comma "
      "separated list of files that has to be equal to the number of --references_in"
    )(
      "scores_prefix_out",
      boost::program_options::value<std::string>(),
      "prefix for files to export the scores instead of a comma separated list in scores_out"
    )(
      "scores_num_out",
      boost::program_options::value<int32>(),
      "number of file to be exported with the --scores_prefix_out"
    )(
      "nonnegative_matrix",
      boost::program_options::value<bool>()->default_value(false),
      "set this true if you want your w_matrix to have nonnegative values only"
    )(
      "l2_matrix_regularization",
      boost::program_options::value<double>()->default_value(0.0),
      "l2 regularization parameter, for the weight matrix, for better generalization"
    )(
      "unit_magnitude_scores",
      boost::program_options::value<bool>()->default_value(false),
      "set this true if you want the maximum score of the w_matrix to be one"
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
    std::vector<std::string> references=fl::ws::GetFileSequence("references", vm);
    std::vector<std::string> triplets=fl::ws::GetFileSequence("triplets", vm);
    if (references.size()!=triplets.size()) {
      fl::logger->Die()<<"number of references files (" << references.size() <<") "
        <<" is different from the number of triplets ("<<triplets.size() <<")";
    }
    boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> weight_matrix;
    boost::shared_ptr<TableType> reference_table;
    index_t dimensionality=0;
    ws_->GetTableInfo(references[0], NULL, &dimensionality, NULL, NULL);
    int32 w_matrix_type=0;
    int32 w_matrix_rank=0;
    if (vm.count("weight_matrix_rank")>0) {
      w_matrix_rank=vm["weight_matrix_rank"].as<int32>();
    }
    if (vm["weight_matrix_type"].as<std::string>()=="asymmetric") {
      fl::logger->Message()<<"Using OASIS with asymmetric matrix";
      w_matrix_type=0;
    } else {
      if (vm["weight_matrix_type"].as<std::string>()=="symmetric") {
        w_matrix_type=1;
        fl::logger->Die()<<"--weight_matrix_type=symmetric is not supported yet";
      } else {
        if (vm["weight_matrix_type"].as<std::string>()=="psd") {
          fl::logger->Message()<<"Using OASIS with PSD matrix";
          w_matrix_type=2;
        } else {
          fl::logger->Die()<<"--weight_matrix_type="<<vm["weight_matrix_type"].as<std::string>()
            <<" is not supported";
        }
      }
    }
    auto weight_matrix_string = vm.count("weight_matrix_out")>0?
        vm["weight_matrix_out"].as<std::string>():ws_->GiveTempVarName();

    if (vm.count("weight_matrix_in")==0) {
      if (w_matrix_type==2 && vm.count("weight_matrix_rank")==0) {
        fl::logger->Die()<<"You chose --weight_matrix_type=psd, but "
          "you didn't provide --weight_matrix_in or --weight_matrix_rank, cannot "
          "continue";
      }
      fl::logger->Message()<<"Did not detect --weight_matrix_in, proceeding with random initialization"<<std::endl;
      if (w_matrix_type==2) {
        ws_->Attach(
            weight_matrix_string,
            std::vector<index_t>(1, dimensionality),
            std::vector<index_t>(),
            w_matrix_rank,
            &weight_matrix);
        typename WorkSpaceType::MatrixTable_t::Point_t point;
        for(index_t i=0; i<weight_matrix->n_entries(); ++i) {
          weight_matrix->get(i, &point);
          point.SetRandom(0.0, 1.0);
        }
      } else {
        ws_->Attach(
            weight_matrix_string,
            std::vector<index_t>(1, dimensionality),
            std::vector<index_t>(),
            dimensionality,
            &weight_matrix
          );
        typename WorkSpaceType::MatrixTable_t::Point_t point;
        for(index_t i=0; i<weight_matrix->n_attributes(); ++i) {
          weight_matrix->get(i, &point);
          point.SetAll(0.0);
          point.set(i,1);
        }
      }
    } else {
      fl::logger->Message()<<"Detected --weight_matrix_in importing inital w matrix"<<std::endl;
      boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> weight_matrix1;
      ws_->Attach(vm["weight_matrix_in"].as<std::string>(), &weight_matrix1);
      typename WorkSpaceType::MatrixTable_t::Point_t p1,p2;
      ws_->Attach(
            weight_matrix_string,
            std::vector<index_t>(1, weight_matrix1->n_attributes()),
            std::vector<index_t>(),
            weight_matrix1->n_entries(),
            &weight_matrix
          );
      for(index_t i=0; i<weight_matrix->n_entries(); ++i) {
        weight_matrix->get(i, &p1);
        weight_matrix1->get(i, &p2);
        p1.CopyValues(p2);
      }
      ws_->Purge(vm["weight_matrix_in"].as<std::string>());
      ws_->Detach(vm["weight_matrix_in"].as<std::string>());

      if (weight_matrix->n_entries()!=weight_matrix->n_attributes()) {
        if (w_matrix_type==2) {
          fl::logger->Message()<<"Weight matrix type was set to be PSD and the rank is ("
            <<weight_matrix->n_entries()<<") as inferred by the input matrix";
          w_matrix_rank=weight_matrix->n_entries();
        } else {
          fl::logger->Die()<<"The input matrix is not square which conflicts with the choice "
            "of the --weight_matrix_type="<<vm["weight_matrix_type"].as<std::string>();
        }
      }
    }
    double c_param=vm["aggressiveness_parameter"].as<double>();
    boost::shared_ptr<typename WorkSpaceType::IntegerTable_t> triplets_table;
    int32 iterations=vm["iterations"].as<int32>();
    fl::logger->Message()<<"Starting training";
    double old_score=0;
    boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> backup_weight_matrix;
    double eta=0.1;
    double l2_regularization=vm["l2_matrix_regularization"].as<double>();
    if (l2_regularization>0) {
      eta/=l2_regularization;
    }
    bool is_nonnegative=vm["nonnegative_matrix"].as<bool>();
    for(int32 it=0; it<iterations; ++it) {
      eta=eta/(it+1);
      fl::logger->Message()<<"iteration "<<it<<std::endl;
      backup_weight_matrix.reset(new typename WorkSpaceType::MatrixTable_t());
      weight_matrix->CloneDataOnly(backup_weight_matrix.get());
      for(size_t i=0; i<references.size(); ++i) {
        fl::logger->Message()<<"training with chunk "<<references[i]<<std::endl;
        ws_->Attach(references[i], &reference_table);
        if (reference_table->n_attributes()!=dimensionality) {
          fl::logger->Die()<<"references_in dimensionality detected so far is ("<<dimensionality<<") "
            <<"but this references file ("<<references[i]<<") has dimensionality ("<<reference_table->n_attributes()<<")";
        }
        ws_->Attach(triplets[i], &triplets_table);
        if (triplets_table->n_attributes()!=3) {
          fl::logger->Die()<<"tiplets file ("<<triplets[i]<<") does not have 3 attributes "
            "it has only ("<<triplets_table->n_attributes()<<")";
        }
        typename TableType::Point_t point, point_plus, point_minus;
        typename WorkSpaceType::IntegerTable_t::Point_t index_point;
        std::vector<index_t> shuffled_order;
        bool shuffle=false;
        if (vm["randomize_triplets"].as<bool>()==true) {
          shuffle=true;
          for(size_t k=0; k<triplets_table->n_entries(); ++k) {
            shuffled_order.push_back(k);         
          }
          std::random_shuffle(shuffled_order.begin(), shuffled_order.end());
        }

        for(index_t j=0; j<triplets_table->n_entries(); ++j) {
          triplets_table->get(shuffle?shuffled_order[j]:j, &index_point);
          if (index_point[0]<0 || index_point[0]>=reference_table->n_entries()) {
            fl::logger->Warning()<<"Triplet ("<<j<<") contains invalid indices ("
              <<index_point[0]<<","
              <<index_point[1]<<","
              <<index_point[2]<<") is invalid, skipping it ";
            continue;
          }
          reference_table->get(index_point[0], &point);
          if (index_point[1]<0 || index_point[1]>=reference_table->n_entries()) {
            fl::logger->Warning()<<"Triplet ("<<j<<") contains invalid indices ("
              <<index_point[0]<<","
              <<index_point[1]<<","
              <<index_point[2]<<") is invalid, skipping it ";
            continue;
          }
          reference_table->get(index_point[1], &point_plus);
          if (index_point[2]<0 || index_point[2]>=reference_table->n_entries()) {
            fl::logger->Warning()<<"Triplet ("<<j<<") contains invalid indices ("
              <<index_point[0]<<","
              <<index_point[1]<<","
              <<index_point[2]<<") is invalid, skipping it ";
            continue;
          }
          reference_table->get(index_point[2], &point_minus);
          if (w_matrix_type==0) {
            Oasis<TableType>::Update(point, 
              point_plus, 
              point_minus, 
              c_param, 
              is_nonnegative,
              weight_matrix.get());
          } else {
            if (w_matrix_type==2) {
              OasisLowRank<TableType>::Update(point, 
                point_plus, 
                point_minus, 
                c_param, 
                is_nonnegative,
                weight_matrix.get());
            }
          }
        }
        ws_->Purge(references[i]);
        ws_->Detach(references[i]);
        ws_->Purge(triplets[i]);
        ws_->Detach(triplets[i]);
      }
      if (l2_regularization>0 && w_matrix_type==0) {
         Oasis<TableType>::RegularizeAndProject( 
              eta,
              is_nonnegative,
              l2_regularization,
              weight_matrix.get());
          } else {
            if (l2_regularization>0 && w_matrix_type==2) {
              OasisLowRank<TableType>::RegularizeAndProject(
                eta,
                is_nonnegative,
                l2_regularization,
                weight_matrix.get());
            }
          }

      double total_score=0;
      index_t total_num_of_triplets=0;
      for(size_t i=0; i<references.size(); ++i) {
        double local_score=0;
        index_t local_num_of_triplets=0;
        ws_->Attach(references[i], &reference_table);
        ws_->Attach(triplets[i], &triplets_table);
        if (w_matrix_type==0) {
          Oasis<TableType>::ComputeTrainPrecision(
            *triplets_table, 
            *reference_table,
            *weight_matrix,
            &local_score,
            &local_num_of_triplets,
            (typename WorkSpaceType::MatrixTable_t*)NULL);
        } else {
          if (w_matrix_type==2) {
            OasisLowRank<TableType>::ComputeTrainPrecision(
              *triplets_table, 
              *reference_table,
              *weight_matrix,
              &local_score,
              &local_num_of_triplets, 
              (typename WorkSpaceType::MatrixTable_t*)NULL);
          }
        }
        total_score+=local_score;
        total_num_of_triplets+=local_num_of_triplets;
        ws_->Purge(references[i]);
        ws_->Detach(references[i]);
        ws_->Purge(triplets[i]);
        ws_->Detach(triplets[i]);
      }

      double new_score=total_score/total_num_of_triplets;
      if (new_score<old_score) {
        weight_matrix=backup_weight_matrix;
        fl::logger->Message()<<"Skipped iteration ("<<it<<") accuracy did not increase";
      } else {
        old_score=new_score;
      }
    }
    fl::logger->Message()<<"Finished training"<<std::endl;
    if (vm["compute_train_score"].as<bool>()) {
      std::vector<std::string> score_files;
      if (vm.count("scores_out") || 
          vm.count("scores_prefix_out") ||
          vm.count("scores_num_out")) {
        score_files=fl::ws::GetFileSequence("scores", vm);
        if (score_files.size() != references.size()) {
          fl::logger->Die()<<"Score files must be the same number with references files";
        }
      }
      double total_score=0;
      index_t total_num_of_triplets=0;
      for(size_t i=0; i<references.size(); ++i) {
        double local_score=0;
        index_t local_num_of_triplets=0;
        ws_->Attach(references[i], &reference_table);
        ws_->Attach(triplets[i], &triplets_table);
        boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> score_table;
        if (score_files.size()>0) {
          ws_->Attach(score_files[i],
              std::vector<index_t>(1,2),
              std::vector<index_t>(),
              triplets_table->n_entries(), 
              &score_table);
        }
        if (w_matrix_type==0) {
          Oasis<TableType>::ComputeTrainPrecision(
             *triplets_table, 
             *reference_table,
             *weight_matrix,
             &local_score,
             &local_num_of_triplets,
             score_table.get());
        } else {
          if (w_matrix_type==2) {
             OasisLowRank<TableType>::ComputeTrainPrecision(
             *triplets_table, 
             *reference_table,
             *weight_matrix,
             &local_score,
             &local_num_of_triplets,
             score_table.get());
          }
        }
        if (score_files.size()>0) {
          ws_->Purge(score_files[i]);
          ws_->Detach(score_files[i]);
        }
        total_score+=local_score;
        total_num_of_triplets+=local_num_of_triplets;
        ws_->Purge(references[i]);
        ws_->Detach(references[i]);
        ws_->Purge(triplets[i]);
        ws_->Detach(triplets[i]);
      }
      fl::logger->Message()<<"number of triplets="<<total_num_of_triplets<<" score="<<total_score;
      fl::logger->Message()<<"Trainining accuracy: "<<(100.0*total_score)/total_num_of_triplets<<"%";
    }

    ws_->Purge(weight_matrix_string);
    ws_->Detach(weight_matrix_string); 
  }

  template<typename WorkSpaceType>
  int Oasis<boost::mpl::void_>::Run(
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
  Oasis<boost::mpl::void_>::Core<WorkSpaceType>::Core(
     WorkSpaceType *ws, const std::vector<std::string> &args) :
   ws_(ws), args_(args)  {}

}}
#endif

