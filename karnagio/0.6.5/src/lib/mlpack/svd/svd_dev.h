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
#ifndef FL_LITE_MLPACK_SVD_SVD_DEV_H
#define FL_LITE_MLPACK_SVD_SVD_DEV_H

#include <algorithm>
#include "mlpack/svd/svd.h"
#include "fastlib/table/linear_algebra.h"
#include "fastlib/math/fl_math.h"
#include "fastlib/table/default_sparse_double_table.h"
#include "fastlib/table/default_table.h"

namespace fl {namespace ml {

template<typename TableType>
template<typename TableTypeIn, typename ExportedTableType>
void Svd<TableType>::ComputeFull(TableTypeIn &table,
                         int32 svd_rank,
                         ExportedTableType *sv,
                         ExportedTableType *left,
                         ExportedTableType *right_trans) {
  FL_SCOPED_LOG(lapack); 
  if (table.n_entries()!=left->n_entries() 
      || svd_rank != left->n_attributes()) {
    fl::logger->Die()
      <<"input table ("<<table.n_entries()<<" x "
      << table.n_attributes()<<")"
      <<" but left table is "
      <<"("<<left->n_entries()<<" x "<< left->n_attributes()<< ")"
      <<"  and svd_rank="<<svd_rank;
  }
  if (svd_rank != sv->n_entries()) {
    fl::logger->Die()
      <<"input table ("<<table.n_entries()<<" x "
      << table.n_attributes()<<")"
      <<" but singular value table is "
      <<"("<<sv->n_entries()<<" x "<< sv->n_attributes()<< ")"
      <<"  and svd_rank="<<svd_rank;
  }
  if (table.n_attributes()!=right_trans->n_entries() 
      || svd_rank !=right_trans->n_attributes()) {
    fl::logger->Die()    
      <<"input table ("<<table.n_entries()<<" x "
      << table.n_attributes()<<")"
      <<" but right transpose table is "
      <<"("<<right_trans->n_entries()<<" x "<< right_trans->n_attributes()<< ")"
      <<"  and svd_rank="<<svd_rank;

  }
  fl::dense::Matrix<double, false> covariance;
  covariance.Init(table.n_attributes(), table.n_attributes());
  covariance.SetAll(0);
  fl::logger->Message()<<"Computing covariance matrix"<<std::endl;
  typedef typename TableTypeIn::Point_t Point_t;
  Point_t point1;
  for(index_t i = 0; i < table.n_entries(); ++i) {
    table.get(i, &point1);
    for(typename Point_t::iterator it1=point1.begin(); it1!=point1.end(); ++it1) {
      for(typename Point_t::iterator it2=point1.begin(); it2!=point1.end(); ++it2) {
        covariance.set(it1.attribute(), it2.attribute(), 
            covariance.get(it1.attribute(), it2.attribute())+it1.value()*it2.value());
      }
    }
  }
  fl::logger->Message()<<"Finished covariance computation"<<std::endl;
  
  fl::logger->Message()<<"Starting LAPACK computations"<<std::endl;
  fl::dense::Matrix<double, false> singular_values;
  fl::dense::Matrix<double, false> tmp_left_singular;
  fl::dense::Matrix<double, false> tmp_right_singular_transposed;
  success_t success_flag;
  fl::dense::ops::SVD<fl::la::Init>(
      covariance,
      &singular_values,
      &tmp_left_singular,
      &tmp_right_singular_transposed,
      &success_flag);
    fl::logger->Message()<<"Finished LAPACK computations"<<std::endl;
    if (success_flag!=SUCCESS_PASS) {
      fl::logger->Warning()<<"There was an error in LAPACK SVD computation, problem unstable"
        <<std::endl;
    }
    // The singular value is the square root of the eigenvalues.
    for(index_t i=0; i<sv->n_entries(); ++i) {
      typename ExportedTableType::Point_t sv_point;
      sv->get(i, &sv_point);
      if (i<singular_values.size()) {
        sv_point.set(0, sqrt(singular_values[i]));
      } else {
        sv_point.set(0, 0);
      }
      if (sv->get(i, index_t(0))<1e-10) {
        fl::logger->Warning()<<"Singular value i="<<i
          <<" is less than 1e-10, you might want to reconsider reducing "
          "the rank of svd";
      }
    }

    typename ExportedTableType::Point_t point;
    for(index_t i=0; i<right_trans->n_entries(); ++i) {
      right_trans->get(i, &point);
      for(index_t j=0; j<right_trans->n_attributes(); ++j) {
        point.set(j, tmp_right_singular_transposed.get(i, j));
      }
    }

    // Compute the left singular vectors.
    for(index_t i=0; i<left->n_entries(); ++i) {
      table.get(i, &point1);
      left->get(i, &point);
      point.SetAll(0.0);
      for(index_t j=0; j<point.size(); ++j) {
        for(typename TableTypeIn::Point_t::iterator it=point1.begin(); 
            it!=point1.end();++it) {
          point.set(j, point[j] + it.value() * 
              tmp_right_singular_transposed.get(it.attribute(), j));
        }
        point.set(j, point[j]/sqrt(singular_values[j]));
      }
    }
    fl::logger->Message()<<"Finished SVD computation"<<std::endl;
}

template<typename TableType>
template<typename ExportedTableType>
void Svd<TableType>::ComputeLowRankSgd(Table_t &table,
                        double step0,
                        index_t n_epochs,
                        index_t n_iterations,
                        bool randomize,
                        ExportedTableType *left,
                        ExportedTableType *right_trans) {

  typedef typename Table_t::Point_t Point_t;
  Point_t point;
  double norm=0;
  index_t total_elements=0;
  for(index_t i=0; i<table.n_entries(); ++i) {
    table.get(i, &point);
    norm+=fl::la::Dot(point, point);
    for(typename Point_t::iterator it=point.begin();
        it!=point.end(); ++it) {
      total_elements++;
     // norm+=fl::math::Sqr(it.value());
    }
  }
  norm=sqrt(norm);
  double total_error=0;
  std::vector<index_t> permutations(table.n_entries());
  if (randomize==true) {
    for(index_t i=0; i<permutations.size(); ++i) {
      permutations[i]=i;
    }
    std::random_shuffle(permutations.begin(), permutations.end());
  }
  typename ExportedTableType::Point_t lpoint, rpoint;
  typename ExportedTableType::Point_t new_lpoint, new_rpoint;
  // initialize it
  left->get(0, &lpoint);
  new_lpoint.Copy(lpoint);
  new_rpoint.Copy(lpoint);
  // initialize matrices with random data
  for(index_t i=0; i<left->n_entries(); ++i) {
    left->get(i, &lpoint);
    lpoint.SetRandom(0.0, 1.0);
    double sum=0;
    if (sum==0) {
      continue;
    }
    fl::la::Sum(lpoint, &sum);
    fl::la::SelfScale(1.0/sum, &lpoint);
  }
  for(index_t i=0; i<right_trans->n_entries(); ++i) {
    right_trans->get(i, &rpoint);
    rpoint.SetRandom(0.0, 1.0);
    double sum=0;
    if (sum==0) {
      continue;
    }
    fl::la::Sum(rpoint, &sum);
    fl::la::SelfScale(1.0/sum, &rpoint);
  }
  
  double step=step0;
  double previous_error=0;
  for(index_t i=0; i<table.n_entries(); ++i) {
    table.get(i, &point);
    left->get(i, &lpoint); 
    for(typename Point_t::iterator it=point.begin(); 
        it!=point.end(); ++it) {
      right_trans->get(it.attribute(), &rpoint);
      double new_error=it.value()-fl::la::Dot(lpoint, rpoint);
      previous_error+=new_error*new_error;
    }
  }  
  for(index_t epoch=0; epoch<n_epochs; ++epoch) {

    step= step/(1+epoch);
    if (randomize==true) {
      std::random_shuffle(permutations.begin(), permutations.end());
    }
    for(index_t iteration=0; iteration<n_iterations; ++iteration) {
      total_error=0;
      index_t skips=0;
      for(index_t i=0; i<table.n_entries(); ++i) {
        index_t new_ind;
        if(randomize==true) {
          new_ind=permutations[i];
        } else {
          new_ind=i;
        }
        table.get(new_ind, &point);
        left->get(new_ind, &lpoint); 
        double error=0;
        for(typename Point_t::iterator it=point.begin(); 
            it!=point.end(); ++it) {
          right_trans->get(it.attribute(), &rpoint);
          error = it.value()-fl::la::Dot(lpoint, rpoint); 
          if (boost::math::isnan(error) || boost::math::isinf(error)) {
             rpoint.SetRandom(0.0, 1.0);
             lpoint.SetRandom(0.0, 1.0);
            break;
          }
          new_lpoint.CopyValues(lpoint);
          new_rpoint.CopyValues(rpoint);
          fl::la::AddTo(error*step, rpoint, &new_lpoint);
          fl::la::AddTo(error*step, lpoint, &new_rpoint);
          double new_error=it.value()-fl::la::Dot(new_lpoint, new_rpoint);
          // if the error does not decrease then revert
          //std::cout<<step<<" "<<fabs(new_error)<<" "<<fabs(error)<<std::endl;
          if (fabs(new_error)<fabs(error)) {
            lpoint.CopyValues(new_lpoint);
            rpoint.CopyValues(new_rpoint); 
          } else {
            skips++;
          } 
        }
        if (boost::math::isnan(error) || boost::math::isinf(error)) {
          break;
        }
      }
      total_error=0;
      for(index_t i=0; i<table.n_entries(); ++i) {
        table.get(i, &point);
        left->get(i, &lpoint); 
        for(typename Point_t::iterator it=point.begin(); 
            it!=point.end(); ++it) {
          right_trans->get(it.attribute(), &rpoint);
          double new_error=it.value()-fl::la::Dot(lpoint, rpoint);
          total_error+=new_error*new_error;
        }
      }
      if (previous_error<total_error) {
        for(index_t i=0; i<left->n_entries(); ++i) {
          left->get(i, &lpoint);
          lpoint.SetRandom(0.0, 1.0);
          double sum=0;
          if (sum==0) {
             continue;
          }
          fl::la::Sum(lpoint, &sum);
          fl::la::SelfScale(1.0/sum, &lpoint);
        }
        for(index_t i=0; i<right_trans->n_entries(); ++i) {
          right_trans->get(i, &rpoint);
          rpoint.SetRandom(0.0, 1.0);
          double sum=0;
          if (sum==0) {
            continue;
          }
          fl::la::Sum(rpoint, &sum);
          fl::la::SelfScale(1.0/sum, &rpoint);
        }
      }
      if (1.0*skips/total_elements>0.5) {
        step/=10; 
        epoch++;
      }
      double percentage_skip=static_cast<int>(
          10000.0*skips/total_elements)/100.0;
      double relative_error=static_cast<int>(10000*sqrt(total_error)/norm)/100.00;
      double mean_square_error=sqrt(total_error/total_elements);
      fl::logger->Message()<<"epoch="<<epoch
        <<", iteration="<<iteration
        <<", step="<<step
        <<", invalid_updates="<<percentage_skip<<"%"
        <<", relative_error="
        <<relative_error<<"%"
        <<", mse="<<mean_square_error
        <<", mv="<<norm/sqrt(total_elements)
        <<std::endl;

    }
  }
}

template<typename TableType>
template<typename ExportedTableType, typename ProjectionTableType>
void Svd<TableType>::ComputeRandomizedSvd(Table_t &table,
                           int smoothing_p,
                           ProjectionTableType &projector,
                           ExportedTableType *sv,
                           ExportedTableType *left,
                           ExportedTableType *right_trans) {
  FL_SCOPED_LOG(randomized); 
  ProjectionTableType y_table1, y_table2;
  y_table1.Init("", 
      std::vector<index_t>(1, projector.n_entries()),
      std::vector<index_t>(),
      table.n_entries()); 
  y_table2.Init("", 
      std::vector<index_t>(1, y_table1.n_attributes()),
      std::vector<index_t>(),
      table.n_attributes());
  fl::table::Mul<fl::la::NoTrans, fl::la::Trans>(table, 
      projector, &y_table1);
  fl::logger->Message()<<"Matrix smoothing in progress"<<std::endl;
  for(int i=0; i<smoothing_p; ++i) {
    fl::table::Mul<fl::la::Trans, fl::la::NoTrans>(table, 
      y_table1, &y_table2);
    fl::table::Mul<fl::la::NoTrans, fl::la::NoTrans>(table, 
        y_table2, &y_table1);
  }
  fl::logger->Message()<<"Matrix smoothing done"<<std::endl;
  ProjectionTableType q_table;
  ProjectionTableType r_table;
  fl::logger->Message()<<"Orthonormalization in progress"<<std::endl;
  fl::table::QR(y_table1, &q_table, &r_table);
  fl::logger->Message()<<"Orthonormalization done"<<std::endl;
  ProjectionTableType b_table;
  fl::table::Mul<fl::la::Trans, fl::la::NoTrans>(q_table, table, 
      &b_table);
  ProjectionTableType left1;
  fl::logger->Message()<<"Final low rank Svd in progress"<<std::endl;
  fl::table::SVD(b_table, sv,  &left1, right_trans);
  fl::logger->Message()<<"Final low rank Svd done"<<std::endl;
  fl::table::Mul<fl::la::NoTrans, fl::la::NoTrans>(q_table, left1, left);
  fl::logger->Message()<<"Randomized Svd finished"<<std::endl;
}

template<typename TableType>
template<typename ExportedTableType>
void Svd<TableType>::ComputeConceptSvd(Table_t &table,
                                       const std::vector<double> &l2norms,
                                       int32 n_iterations,
                                       double error_change,
                                       ExportedTableType *sv,
                                       ExportedTableType *left,
                                       ExportedTableType *right_trans) {
  int32 svd_rank=sv->n_entries();
  typedef fl::table::DefaultSparseDoubleTable CentroidTable_t;
  CentroidTable_t centroids;
  typename CentroidTable_t::Point_t centroid;
  centroids.Init("", 
                 std::vector<index_t>(),
                 std::vector<index_t>(1, table.n_attributes()),
                 svd_rank);
  // contrary to what we always think about memberships we use
  // cluster associates. In short for every cluster we keep a
  // list of the associated points
  std::vector<std::vector<index_t> > cluster_associates(svd_rank);
  for(index_t i=0; i<table.n_entries(); ++i) {
    int32 cluster_id=fl::math::Random(int32(0), int32(cluster_associates.size()-1));
    cluster_associates[cluster_id].push_back(i);
  }

  for(int32 i=0; i<centroids.n_entries(); ++i) {
    centroids.get(i, &centroid);
    index_t id=fl::math::Random(index_t(0), table.n_entries()-1);
    typename Table_t::Point_t point;
    table.get(id, &point);
    for(typename Table_t::Point_t::iterator it=point.begin();
        it!=point.end(); ++it) {
      if (l2norms.empty()) {
        centroid.set(it.attribute(), it.value());
      } else {
        centroid.set(it.attribute(), it.value()/l2norms[id]);
      }
    }
    // std::cout<<fl::la::LengthEuclidean(centroid)<<std::endl;
  }

  typename Table_t::Point_t point, scaled_point;
  for(int32 iteration=0; iteration<n_iterations; ++iteration) {
    cluster_associates.clear();
    cluster_associates.resize(centroids.n_entries());
    for(index_t i=0; i<table.n_entries(); ++i) {
      table.get(i, &point);
      double max_dot=-std::numeric_limits<double>::max();
      int32 argmax_dot=-1;
      for(int32 j=0; j<centroids.n_entries(); ++j) {
        centroids.get(j, &centroid);
        double dot=fl::la::Dot(centroid, point);
        if (!l2norms.empty()) {
          dot/=l2norms[i];
        }
        //DEBUG_ASSERT(fabs(dot)<=1);
        if (dot>=max_dot) {
          max_dot=dot;
          argmax_dot=j;
        }
      }
      //std::cout<<std::endl;
      cluster_associates[argmax_dot].push_back(i);
    }
    // compute the centroids
    for(int32 j=0; j<cluster_associates.size(); ++j) {
      std::map<int32, double> new_centroid;
      for(std::vector<index_t>::iterator it=cluster_associates[j].begin(); 
          it!=cluster_associates[j].end(); ++it) {
        typename Table_t::Point_t new_point;
        table.get(*it, &point);
        if (!l2norms.empty()) {
          fl::la::Scale<fl::la::Init>(l2norms[*it], point, &new_point);
          for(typename Table_t::Point_t::iterator pit=new_point.begin();
              pit!=new_point.end(); ++pit) {
            new_centroid[pit.attribute()]+=pit.value();
          }
        } else {
          for(typename Table_t::Point_t::iterator pit=point.begin();
              pit!=point.end(); ++pit) {
            new_centroid[pit.attribute()]+=pit.value();
          }
        }
      }
      double norm=0;
      for(std::map<int32, double>::iterator it=new_centroid.begin();
          it!=new_centroid.end(); ++it) {
        norm+=fl::math::Sqr(it->second);
      }
      norm=sqrt(norm);
      for(std::map<int32, double>::iterator it=new_centroid.begin();
          it!=new_centroid.end(); ++it) {
        it->second/=norm;
      }
      centroids.get(j, &centroid);
      centroid.Load(new_centroid.begin(), new_centroid.end());
      //std::cout<<"## "<<norm<<std::endl;
      //std::cout<< "** "<<fl::la::LengthEuclidean(centroid)<<std::endl;
    }
    fl::logger->Message()<<"iteration="<<iteration<<" completed"<<std::endl;   
  }
  // now it is time for SVD
  ExportedTableType covariance;
  fl::table::Mul<fl::la::NoTrans, fl::la::Trans>(
      centroids,
      centroids,
      &covariance); 
  ExportedTableType temp_left, temp_right_trans;
  fl::table::SVD(covariance, sv, &temp_left, &temp_right_trans);  
  int32 true_rank=-1;
  for(index_t i=0; i<sv->n_entries(); ++i) {
    index_t dummy=0;
    if (sv->get(i, dummy)/sv->get(dummy, dummy)<1e-10) {
      sv->set(i, 0, 0);
      if (true_rank==-1) {
        true_rank=i+1;
      }
    } else {
      true_rank=i+1;
      sv->set(i, index_t(0), sqrt(sv->get(i, dummy)));
    }
  }
  if (true_rank<svd_rank) {
    fl::logger->Warning()<<"Concept vectors have rank ("
      <<true_rank<<") less than ("<< svd_rank<<") you asked"
      <<std::endl;
  }
  right_trans->SetAll(0.0);
  for(index_t i=0; i<centroids.n_entries(); ++i) {
    centroids.get(i, &centroid);
    for(typename CentroidTable_t::Point_t::iterator 
        it=centroid.begin(); it!=centroid.end(); ++it) {
      right_trans->set(it.attribute(), i, it.value());
    }
  }
  //fl::table::Mul<fl::la::Trans, fl::la::NoTrans>(
  //    centroids, temp_left, right_trans);
  ExportedTableType temp1, temp2, scaled_right_trans;
  temp_right_trans.CloneDataOnly(&scaled_right_trans);
  for(index_t i=0; i<scaled_right_trans.n_entries(); ++i) {
    typename ExportedTableType::Point_t point;
    scaled_right_trans.get(i, &point);
    for(index_t j=0; j<point.size(); ++j) {
      if (sv->get(j)/sv->get(0) <1e-10) {
        point.set(j, 0.0);
      } else {
        point.set(j, point[j]/(sv->get(j)));
      }
    }
  }
  fl::table::Mul<fl::la::NoTrans, fl::la::Trans>(
      table,
      centroids,
      &temp1);
  fl::table::Mul<fl::la::NoTrans, fl::la::Trans>(
      temp1,
      scaled_right_trans,
      &temp2);
  fl::table::Mul<fl::la::NoTrans, fl::la::Trans>(
      temp2,
      temp_left,
      left);
}

template<typename TableType>
template<typename ExportedTableType, typename TableVectorType>
void Svd<TableType>::ComputeRecError(Table_t &table,
                      TableVectorType &sv,
                      ExportedTableType &left,
                      ExportedTableType &right_trans,
                      double *error) {
  ExportedTableType table1;
  table1.Init("",
      std::vector<index_t>(1, left.n_attributes()),
      std::vector<index_t>(),
      left.n_entries());
  for(index_t i=0; i<left.n_entries(); ++i) {
    typename ExportedTableType::Point_t p1,p2;
    left.get(i, &p1);
    table1.get(i, &p2);
    for(index_t j=0; j<p2.size(); ++j) {
      index_t dummy=0;
      p2.set(j, p1[j] * sv.get(j, dummy));
    }
  }
  ExportedTableType table2;
  fl::table::Mul<fl::la::NoTrans, fl::la::Trans>(table1, right_trans, &table2);
  double norm=0;
  for(index_t i=0; i<table.n_entries(); ++i) {
    typename Table_t::Point_t point1;
    table.get(i, &point1);
    typename ExportedTableType::Point_t point2;
    table2.get(i, &point2);
    for(index_t j=0; j<point2.size(); ++j) {
      norm+=point1[j]*point1[j];
      *error+=(point1[j]-point2[j])*(point1[j]-point2[j]);
    }
  }
  *error/=norm;
  *error=sqrt(*error);
}

}}

#endif
