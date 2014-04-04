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

#ifndef FL_LITE_MLPACK_SVM_SVM_DEV_H_
#define FL_LITE_MLPACK_SVM_SVM_DEV_H_
#include "svm.h"
#include "mlpack/allkn/allkn_dev.h"
#include "mlpack/allkn/allkn_computations_dev.h"

namespace fl {namespace ml {

template<typename TableType>
template<typename GeometryType>
Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
          typename TableType::CalcPrecision_t, GeometryType> >::Trainer() {
  references_=NULL;
  labels_=NULL;
  regularization_=1.0;
  iterations_=100;
  accuracy_=1e-3;
  bias_=false;
}


template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::set_reference_table(Table_t *reference_table) {
  references_=reference_table;
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::set_labels(std::vector<index_t> *labels) {
  labels_=labels;
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::set_regularization(double regularization) {
  regularization_=regularization;
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::set_iterations(index_t iterations) {
  iterations_=iterations;
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::set_accuracy(double accuracy) {
  accuracy_=accuracy;
}
         
template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::set_kernel(Kernel_t &kernel) {
  kernel_=kernel;
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::set_bandwidth_overload_factor(
        double bandwidth_overload_factor) {
  bandwidth_overload_factor_=bandwidth_overload_factor;
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::set_bias(bool bias) {
  bias_=bias;
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::Train(
  std::map<index_t, double> *support_vectors_full,
  boost::shared_ptr<Table_t> *sv_table) {
  sv_table->reset(new Table_t);
  Table_t &sample_table=*(sv_table->get()); 
  std::vector<std::pair<index_t, double> > support_vectors_sample;
  // first we need to sample the data
  if (kernel_.bandwidth()<0) {
    kernel_.set(references_->get_node_bound(references_->get_tree()).
        MaxDistanceWithinBound()/6);
    fl::logger->Message()<<"Using suggested bandwidth: "<<kernel_.bandwidth()<<std::endl;
  }
  index_t suggested_knns=0;
  fl::logger->Message()<<"Sampling training points"<<std::endl;
  Sample(*references_, bandwidth_overload_factor_, &sample_table, &suggested_knns);
  if (sample_table.n_entries()!=0) {
    fl::logger->Message()<<"Picked "<<sample_table.n_entries()<<" points for training"
      <<std::endl;
  } else {
     fl::logger->Message()<<"Picked "<<references_->n_entries()<<" points for training"
      <<std::endl; 
  }
  fl::logger->Message()<<"Suggested k-nearest neighbors for SMO: "
      << suggested_knns;
  // Then we run SMO on the limitted set
  std::vector<PointInfo> point_info;
  fl::logger->Message() << "Initializing SMO"<<std::endl;
  InitSmo(sample_table.n_entries()==0?*references_:sample_table, 
      suggested_knns,
      &point_info);
  fl::logger->Message() << "Running SMO"<<std::endl;
  RunSmo(sample_table.n_entries()==0?*references_:sample_table,     
    accuracy_,
    regularization_,
    iterations_, 
    &point_info,
    support_vectors_full);
    double error;
    if (sample_table.n_entries()>0) {
      ComputeTrainingError(sample_table, sample_table,  *support_vectors_full, &error);
      fl::logger->Message()<<"Total error(sample)="<<error<<"%"<<std::endl;
    }
    ComputeTrainingError(sample_table.n_entries()>0?sample_table:*references_, *references_,  *support_vectors_full, &error);
    fl::logger->Message()<<"Total error(all data)="<<error<<"%"<<std::endl;

    //  std::vectors<std::pair<index_t, double> > support_vectors_full;
//  // We propagate the support vectors to the other points
//  PropagateSupportVectors(sample_table, *reference_table_, 
//      support_vector_sample, &support_vectors_full);
  // We rerun hot smo from there
//  RunSmo(references_in_, accuracy_, iterations_, &support_vectors_full);
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::InitSmo(
    Table_t &references,
    const index_t knn,
    std::vector<PointInfo> *point_info) {

  // clear the kernel cache
  kernel_cache_.clear();
  typename fl::ml::AllKN<boost::mpl::void_>::Core<TableType>::DefaultAllKNN allknn;
  typename TableType::template IndexArgs<GeometryType> index_args;
  // at some point we should change that and do the indexing
  // according to the diameter. This is implemented but not tested yet
  index_args.leaf_size=20;
  if (references.is_indexed()==false) {
    references.IndexData(index_args);
  }
  allknn.Init(&references, NULL);
  GeometryType metric;
  // compute nearest neighbors
  std::vector<double>  distances;
  std::vector<index_t> indices;
  if (knn>0) {
    fl::logger->Message()<<"Computing "<<knn<<"-nearest neighbors."<<std::endl;
    allknn.ComputeNeighbors("dual", metric, knn, &distances, &indices);
    fl::logger->Debug()<<"Nearest neighbor computation finished"<<std::endl;
  } else {
    fl::logger->Warning()<<"Skipping nearest neighbor computation, data very well "
      "separated";
  }
  point_info->resize(references.n_entries());
  typename Table_t::Point_t point;
  index_t num_of_points=references.n_entries();
  for(index_t i=0; i<references.n_entries(); ++i) {
    references.get(i, &point);
    point_info->at(i).a=0;
    point_info->at(i).y=point.meta_data().template get<0>();
    point_info->at(i).u=0;
    if (knn>0) {
      point_info->at(i).neighbors.resize(knn);
      point_info->at(i).kernel_distances.resize(knn);
      point_info->at(i).distances.resize(knn);
      for(index_t j=0; j<knn; ++j) {  
        point_info->at(i).neighbors[j]=indices[i*knn+j];
        double distance=distances[i*knn+j];
        if (distance==0) {
          distance=std::numeric_limits<double>::max();
          //fl::logger->Warning()<<"Duplicate point detected, please remove it"<<std::endl;
        }
        point_info->at(i).distances[j]=distance;  
        double kernel_prod=kernel_.Dot(distance);
        point_info->at(i).kernel_distances[j]=2*(1-kernel_prod);
        kernel_cache_[get_cache_id(i, indices[i*knn+j], num_of_points)]=kernel_prod;
      }
    } else {
      index_t counter=0;
      for(;;) {  
        typename Table_t::Point_t p;
        index_t ind=fl::math::Random(index_t(0), num_of_points-1);
        references.get(ind, &p);
        if (int(p.meta_data().template get<0>())!=int(point.meta_data().template get<0>())) {
          double distance=metric.DistanceSq(point, p);
          if (distance==0) {
            distance=std::numeric_limits<double>::max();
            //fl::logger->Warning()<<"Duplicate point detected, please remove it"<<std::endl;
          }
          point_info->at(i).neighbors.push_back(ind);
          point_info->at(i).distances.push_back(distance);  
          double kernel_prod=kernel_.Dot(distance);
          point_info->at(i).kernel_distances.push_back(2*(1-kernel_prod));
          kernel_cache_[get_cache_id(i, ind, num_of_points)]=kernel_prod;
          counter++;
          if (counter>10) {
            break;
          }
        } else {
          double distance=metric.DistanceSq(point, p);
          if (distance==0) {
            distance=std::numeric_limits<double>::max();
            //fl::logger->Warning()<<"Duplicate point detected, please remove it"<<std::endl;
          }
          point_info->at(i).neighbors.push_back(ind);
          point_info->at(i).distances.push_back(distance);  
          double kernel_prod=kernel_.Dot(distance);
          point_info->at(i).kernel_distances.push_back(2*(1-kernel_prod));
          kernel_cache_[get_cache_id(i, ind, num_of_points)]=kernel_prod;
        }
      }
    }
  }
  fl::logger->Debug()<<"Initialization of SMO finished"<<std::endl;
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::RunSmo(Table_t &references,
    const double accuracy,
    const double regularization,
    const index_t iterations, 
    std::vector<PointInfo> *point_info,
    std::map<index_t, double> *nonzero_alphas) {
  
  typename Table_t::Point_t point1;
  typename Table_t::Point_t point2;
  index_t num_of_altered=std::numeric_limits<index_t>::max();
  index_t num_of_iterations=0;
  while (num_of_altered>0 && num_of_iterations<iterations_) {
    num_of_altered=0;
    for(unsigned int i=0; i<point_info->size(); ++i) {
      double a1 = point_info->at(i).a;
      index_t i1 = i;
      references.get(i1, &point1);
      int y1=point_info->at(i1).y;
      point_info->at(i1).u=EvaluateSvm(point1,references, *nonzero_alphas);
      double u1=point_info->at(i1).u;
      //double prod=y1*u1;
      double prod1=y1*(u1-y1);
      // check if the KKT conditions are violated
        if ( (prod1<-accuracy_ && a1<regularization_) || (prod1>accuracy_ && a1>0) ) {
//      if (!((fabs(a1)<=accuracy_ &&  prod>=1)
//          || (0<a1 && a1<regularization && 1-accuracy_<=prod && prod<=1+accuracy_)
//          || ((regularization_-accuracy_)<=a1 && a1<=(regularization_+accuracy_) && prod<=1) 
//          )) {
        index_t i2;
        double step=0;
        PickPoint(i1, *point_info, references, *nonzero_alphas, &i2, &step);
        if (step==0) {
          //std::cout<<a1<< " "<< prod << std::endl;
          continue;
        }
        references.get(i2, &point2);
        double y2=point_info->at(i2).y;
        double a2=point_info->at(i2).a;
        double low;
        double hi;
        if (y1!=y2) {
          low=std::max(double(0.0), a2-a1);
          hi=std::min(regularization_, regularization_+(a2-a1));  
        } else {
          low=std::max(double(0.0), a1+a2-regularization_);
          hi=std::min(regularization_, a2+a1);
        }
        if (low==hi) {
          continue;
        }
        double a2_new=a2+y2*step;
        double a2_new_clipped;
        if (a2_new<=low) {
          a2_new_clipped=low;
        } else {
          if (a2_new>=hi) {
            a2_new_clipped=hi;
          } else {
            a2_new_clipped=a2_new;
          }
        }
        if (fabs(a2_new_clipped-a2)<accuracy_*(a2_new_clipped+a2+accuracy_)) {
          continue;
        }
        double s=y1*y2;
        double a1_new=a1+s*(a2-a2_new_clipped);
        if (a1_new<low) {
          a1_new=low;
        } else {
          if (a1_new>hi) {
            a1_new=hi;
          }
        }
        if (a1_new !=0) {
          nonzero_alphas->operator[](i1)=a1_new;
        } else {
          if (nonzero_alphas->count(i1)) {
            nonzero_alphas->erase(i1);  
          }
        }
        if (a2_new_clipped !=0) {
          nonzero_alphas->operator[](i2)=a2_new_clipped;
        } else {
          if (nonzero_alphas->count(i2)) {
            nonzero_alphas->erase(i2);
          }
        }
        BOOST_ASSERT(low<=a1_new && a1_new<=hi);
        BOOST_ASSERT(low<=a2_new_clipped && a2_new_clipped<=hi);
        if (!(low<=a1_new && a1_new<=hi)) {
          fl::logger->Warning()<<a1_new<<std::endl;
        }
        if (!(low<=a2_new_clipped && a2_new_clipped<=hi)) {
          fl::logger->Warning()<<a2_new_clipped;
        }
        point_info->at(i1).a=a1_new;
        point_info->at(i1).u=EvaluateSvm(i1,references, *nonzero_alphas);
        point_info->at(i2).a=a2_new_clipped;
        point_info->at(i2).u=EvaluateSvm(i2, references, *nonzero_alphas);
        num_of_altered++;
      } else {
        //if ((0<a1 && a1<regularization && 1-accuracy_<=prod && prod<=1+accuracy_)
        //    || ((regularization_-accuracy_)<=a1 && a1<=(regularization_+accuracy_))) {
        //  nonzero_alphas->operator[](i)=a1;
        //}
      }
    }
    fl::logger->Debug()<<"Iteration: "<< num_of_iterations<< ", modified points: "
      << num_of_altered
      <<", support vectors: "<<nonzero_alphas->size()
      <<std::endl;
    num_of_iterations++;
  }
}

template<typename TableType>
template<typename GeometryType>
double Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
  typename TableType::CalcPrecision_t, GeometryType>  >::EvaluateSvm(
    const Point_t &point,
    Table_t &references,
    const std::map<index_t, double> &nnz_support_vectors) {
  
  Point_t s_point;
  double result=0;
  for(std::map<index_t, double>::const_iterator it=nnz_support_vectors.begin(); 
      it!=nnz_support_vectors.end(); ++it) {
    references.get(it->first, &s_point);
    double alpha=it->second;
    // we can either use this
    result+=kernel_.Dot(point, s_point)*alpha*int(s_point.meta_data().template get<0>());
  }
  return result;
}

template<typename TableType>
template<typename GeometryType>
double Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
  typename TableType::CalcPrecision_t, GeometryType>  >::EvaluateSvm(
    index_t p1,
    Table_t &references,
    const std::map<index_t, double> &nnz_support_vectors) {
  
  Point_t s_point;
  Point_t point;
  double result=0;
  index_t num_of_points=references.n_entries();
  for(std::map<index_t, double>::const_iterator it=nnz_support_vectors.begin(); 
      it!=nnz_support_vectors.end(); ++it) {
    references.get(it->first, &s_point);
    double dot_prod=0;
    std::map<index_t,double>::const_iterator it1;
    it1=kernel_cache_.find(get_cache_id(p1, it->first, num_of_points));
    if (it1!=kernel_cache_.end()) {
      dot_prod=it->second;
    } else {
      references.get(p1, &point);      
      dot_prod=kernel_.Dot(point, s_point);
    }
    double alpha=it->second;
   
    // we can either use this
    result+=dot_prod*alpha*int(s_point.meta_data().template get<0>());
  }
  return result;
}


template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::PickPoint(
  index_t i1, 
  std::vector<PointInfo> &point_info, 
  Table_t &references,
  std::map<index_t, double> &nnz_support_vectors,
  index_t *i2, double *step) {

  PointInfo &p1=point_info[i1];
  Point_t point1;
  Point_t point2;
  references.get(i1, &point1);
  p1.u=EvaluateSvm(point1, references, nnz_support_vectors);
  *step=0;
  *i2=p1.neighbors[0];
  for(index_t i=0; i<p1.neighbors.size(); ++i) {
    PointInfo &p2=point_info[p1.neighbors[i]];
    references.get(p1.neighbors[i], &point2);
    //p2.u=EvaluateSvm(point2, references, nnz_support_vectors);
    double step1=((p1.u-p1.y)-(p2.u-p2.y))/p1.kernel_distances[i];
    if (fabs(step1)>fabs(*step)) {
      *step=step1;
      *i2=p1.neighbors[i];
    }
  }
  //std::cout<<*step<<std::endl;
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::Sample(
        Table_t &table_in,
        double bandwidth_overload_factor, 
        Table_t *table_out, index_t *suggested_knns) {
  // we need to find the average number of points per diameter
  index_t total_knns=0; // 
  index_t total_nodes=0;     //
  table_out->Init("", 
     table_in.dense_sizes(),
     table_in.sparse_sizes(),
     0);
  double diameter=bandwidth_overload_factor*kernel_.bandwidth();
  SampleRecursion(table_in.get_tree(), diameter, table_in, table_out, 
      &total_knns, &total_nodes);
  if (total_nodes==0) {
    fl::logger->Warning() << "Classes seem to be very well separated or the bandwidth is very small"
      <<std::endl;
    *suggested_knns=-1;
  } else {
    *suggested_knns=total_knns/total_nodes;
    fl::logger->Message()<<"Suggested neighbors to be used is: "<<*suggested_knns;
    if (*suggested_knns>100) {
      fl::logger->Message()<<"Your overload factor is too high. Clipping nearest neighbors to: 100"
        <<std::endl;
      *suggested_knns=100;  
    }
  }
}



template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::SampleRecursion(
  typename Table_t::Tree_t *node,
  const double diameter,
  Table_t &table_in, 
  Table_t *table_out,
  index_t *total_knns,
  index_t *total_nodes) {

  double current_diameter = table_in.get_node_bound(node).MaxDistanceWithinBound();
  Point_t point;
  if (table_in.node_is_leaf(node) || current_diameter<=diameter) {
    typename TableType::TreeIterator it=table_in.get_node_iterator(node);
    typename TableType::Point_t point;
    index_t point_id;
    index_t label=0;
    bool mixed_node=false;
    it.Next(&point, &point_id);
    if (it.HasNext()) {
      label=int(point.meta_data().template get<0>());
    }
    while(it.HasNext()) {
      it.Next(&point, &point_id);  
      if ((label<0 && int(point.meta_data().template get<0>()) >0) 
          || (label>0 && int(point.meta_data().template get<0>()) <0)) {
        mixed_node=true;
        break;      
      } 
    }
    if (mixed_node==true) {
      // start copying
      while(it.HasNext()) {
        it.Next(&point, &point_id);
        point.meta_data().template get<1>()=point_id;
        table_out->push_back(point);
      }   
      *total_knns+=it.count();
      *total_nodes+=1;
    } 
  } else {
    SampleRecursion(table_in.get_node_left_child(node), 
        diameter, table_in, table_out, total_knns, total_nodes);
    SampleRecursion(table_in.get_node_right_child(node), 
        diameter, table_in, table_out, total_knns, total_nodes);
  }
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::ComputeTrainingError(
        Table_t &references,
        Table_t &queries, 
        std::map<index_t, double> &nnz_support_vectors,
        double *error) {
  index_t class_one_correct=0;
  index_t total_class_one=0;
  index_t class_mone_correct=0;
  index_t total_class_mone=0;
  Point_t point;
  for(index_t i=0; i<queries.n_entries(); ++i) {
    queries.get(i, &point);
    int label=int(point.meta_data().template get<0>());
    if (label==1) {
      total_class_one+=1;
      double eval=EvaluateSvm(point, references, nnz_support_vectors);
      if (eval>0) {
        class_one_correct+=1;
      }
    } else {
      total_class_mone+=1;
      double eval=EvaluateSvm(point, references, nnz_support_vectors);
      if (eval<0) {
        class_mone_correct+=1;
      }
    } 
  }
  fl::logger->Message()<<"For class 1: total="<<total_class_one<<", success="
    <<double(class_one_correct)*100.0/total_class_one<<"%"<<std::endl;
  fl::logger->Message()<<"For class -1: total="<<total_class_mone<<", success="
    <<double(class_mone_correct)*100.0/total_class_mone<<"%"<<std::endl;

  *error=100.0*(1.0-double(class_one_correct+class_mone_correct)/
      double(total_class_one+total_class_mone));
}

template<typename TableType>
template<typename GeometryType>
index_t Svm<TableType>::Trainer<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::get_cache_id(index_t i,
        index_t j, index_t n) {
  if (i<=j) {
    return i*n+j;
  } else  {
    return j*n+i;
  }
}


template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Predictor<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::set_query_table(Table_t *query_table) {
  query_table_=query_table;
}


template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Predictor<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::set_kernel(Kernel_t &kernel) {
  kernel_=kernel;
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Predictor<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::set_support_vectors(Table_t *support_vectors) {
  support_vectors_=support_vectors;    
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Predictor<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::set_alphas(std::vector<double> *alphas) {
  alphas_=alphas;    
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Predictor<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::Predict(std::vector<double> *margins, double *prediction_accuracy) {
  Point_t point;
  for(index_t i=0; i<query_table_->n_entries(); ++i) {
    query_table_->get(i, &point);
    Point_t s_point;
    margins->operator[](i)=0;
    for(index_t j=0; j<support_vectors_->n_entries(); ++j) {
      support_vectors_->get(j, &s_point);
      double alpha=alphas_->operator[](j);
      margins->operator[](i)+=kernel_.Dot(point, s_point)*alpha 
        * s_point.meta_data().template get<0>();
    }
    if (prediction_accuracy!=NULL) {
      *prediction_accuracy+=margins->operator[](i) 
        * point.meta_data().template get<0>()>0; 
    }
  }
  *prediction_accuracy/=query_table_->n_entries();  
}

template<typename TableType>
template<typename GeometryType>
void Svm<TableType>::Predictor<fl::math::GaussianDotProduct<
    typename TableType::CalcPrecision_t, GeometryType> >::Predict(const Point_t &point, 
        double *margin) {
  Point_t s_point;
  *margin=0;
  for(index_t j=0; j<support_vectors_.n_entries(); ++j) {
    support_vectors_->get(j, &s_point);
    double alpha=alphas_->operator[](j);
    *margin+=kernel_.Dot(point, s_point)*alpha 
      * s_point.meta_data().template get<0>();
  }
}

}}

#endif
