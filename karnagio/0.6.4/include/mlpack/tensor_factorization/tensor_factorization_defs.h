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

#ifndef PAPERBOAT_INCLUDE_MLPACK_TENSOR_FACTORIZATION_DEFS_H_
#define PAPERBOAT_INCLUDE_MLPACK_TENSOR_FACTORIZATION_DEFS_H_
#include "tensor_factorization.h"
#include "fastlib/optimization/lbfgs/lbfgs_dev.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/util/string_utils.h"
#include "fastlib/workspace/based_on_table_run.h"

namespace fl { namespace ml {
  
  template<typename WorkSpaceType>
  template<typename TableType,
               bool A_REGULARIZATION,
               bool B_REGULARIZATION,
               bool C_REGULARIZATION,
               bool L2_REGULARIZATION
               > 
  void TensorFactorization<WorkSpaceType>::LbfgsFun<
      TableType,
      A_REGULARIZATION,
      B_REGULARIZATION,
      C_REGULARIZATION,
      L2_REGULARIZATION
  >::AssignMats(
      fl::data::MonolithicPoint<double> &variable,
      fl::dense::Matrix<double> *a_mat,
      fl::dense::Matrix<double> *b_mat,
      fl::dense::Matrix<double> *c_mat) {
    a_mat->Alias(variable.ptr(),
        rank_,
        (*tensor_)[0]->n_entries());
    b_mat->Alias(variable.ptr()+a_mat->n_elements(),
        rank_,
        tensor_->size());
    c_mat->Alias(variable.ptr()+a_mat->n_elements()
                 +b_mat->n_elements(),
                 rank_,
                 (*tensor_)[0]->n_attributes());

  }

  template<typename WorkSpaceType>
  template<typename TableType>
  void TensorFactorization<WorkSpaceType>::AssignMats(fl::data::MonolithicPoint<double> &variable,
              std::vector<boost::shared_ptr<TableType> > &tensor,
              int32 rank,
              fl::dense::Matrix<double> *a_mat,
              fl::dense::Matrix<double> *b_mat,
              fl::dense::Matrix<double> *c_mat) {
    a_mat->Alias(variable.ptr(),
        rank,
        tensor[0]->n_entries());
    b_mat->Alias(variable.ptr()+a_mat->n_elements(),
        rank,
        tensor.size());
    c_mat->Alias(variable.ptr()+a_mat->n_elements()
                 +b_mat->n_elements(),
                 rank,
                 tensor[0]->n_attributes());
 
  }

  template<typename WorkSpaceType>
  template<typename TableType,
           bool A_REGULARIZATION,
           bool B_REGULARIZATION,
           bool C_REGULARIZATION,
           bool L2_REGULARIZATION
          >
  double TensorFactorization<WorkSpaceType>::LbfgsFun<
      TableType,
      A_REGULARIZATION,
      B_REGULARIZATION,
      C_REGULARIZATION,
      L2_REGULARIZATION
  >::tensor_sq_norm() const {
    return tensor_sq_norm_;
  }

  template<typename WorkSpaceType>
  template<typename TableType,
           bool A_REGULARIZATION,
           bool B_REGULARIZATION,
           bool C_REGULARIZATION,
           bool L2_REGULARIZATION
          >
  double TensorFactorization<WorkSpaceType>::LbfgsFun<
      TableType,
      A_REGULARIZATION,
      B_REGULARIZATION,
      C_REGULARIZATION,
      L2_REGULARIZATION
  >::eval_result() const {
    return eval_result_;
  }


  template<typename WorkSpaceType>
  template<typename TableType,
           bool A_REGULARIZATION,
           bool B_REGULARIZATION,
           bool C_REGULARIZATION,
           bool L2_REGULARIZATION
          >
  void TensorFactorization<WorkSpaceType>::LbfgsFun<
      TableType,
      A_REGULARIZATION,
      B_REGULARIZATION,
      C_REGULARIZATION,
      L2_REGULARIZATION
  >::Init(
      std::vector<boost::shared_ptr<TableType> > *tensor,
      int32 factorization_rank,
      double a_regularization,
      double b_regularization, 
      double c_regularization) {
    tensor_=tensor;
    rank_=factorization_rank;
    a_regularization_=a_regularization;
    b_regularization_=b_regularization;
    c_regularization_=c_regularization;
    num_dimensions_=(tensor_->size()
      +(*tensor_)[0]->n_entries()
      +(*tensor_)[0]->n_attributes())*rank_;
     tensor_sq_norm_=0;
    typename TableType::Point_t point;
    for(size_t i=0; i<tensor->size(); ++i) {  
      for(index_t j=0; j<(*tensor)[i]->n_entries(); ++j) {
        (*tensor)[i]->get(j, &point);
        tensor_sq_norm_+=fl::math::Pow<double, 2, 1>(
          fl::la::LengthEuclidean(point));
      }
    }
  }

  template<typename WorkSpaceType>
  template<typename TableType,
           bool A_REGULARIZATION,
           bool B_REGULARIZATION,
           bool C_REGULARIZATION,
           bool L2_REGULARIZATION
          >
  void TensorFactorization<WorkSpaceType>::LbfgsFun<
      TableType,
      A_REGULARIZATION,
      B_REGULARIZATION,
      C_REGULARIZATION,
      L2_REGULARIZATION
  >::Gradient(
      const fl::data::MonolithicPoint<double> &variable,
      fl::data::MonolithicPoint<double> *gradient) {
    fl::dense::Matrix<double> a_mat;
    fl::dense::Matrix<double> b_mat;
    fl::dense::Matrix<double> c_mat;
    fl::dense::Matrix<double> grad_a_mat;
    fl::dense::Matrix<double> grad_b_mat;
    fl::dense::Matrix<double> grad_c_mat;

    AssignMats(const_cast<fl::data::MonolithicPoint<double> &>(variable), &a_mat, 
        &b_mat, &c_mat);
    gradient->SetZero();
    AssignMats(*gradient, &grad_a_mat, &grad_b_mat, &grad_c_mat);
    typedef typename TableType::Point_t Point_t;
    Point_t point;
    for(size_t i=0; i<tensor_->size(); ++i) {
      for(index_t j=0; j<(*tensor_)[i]->n_entries(); ++j) {
        (*tensor_)[i]->get(j, &point);
        for(typename Point_t::iterator it=point.begin(); 
            it!=point.end(); ++it) {
          double predicted_value=0;        
          for(size_t k=0; k<rank_; ++k) {
            predicted_value+=a_mat.get(k, j)
              *b_mat.get(k, i)
              *c_mat.get(k, it.attribute());
          }
          double error=-2*(it.value()-predicted_value);
          for(size_t k=0; k<rank_; ++k) {
            grad_a_mat.set(k, j, 
                grad_a_mat.get(k, j)
                +error*b_mat.get(k, i)*c_mat.get(k, it.attribute()));
            
            grad_b_mat.set(k, i, 
                grad_b_mat.get(k, i)+
                error*a_mat.get(k, j)*c_mat.get(k, it.attribute()));
            grad_c_mat.set(k, it.attribute(),
                grad_c_mat.get(k, it.attribute())+
                error*a_mat.get(k, j)*b_mat.get(k, i));

          }
        }     
      }
    }
    if (A_REGULARIZATION) {
      fl::la::AddExpert(2*a_regularization_, a_mat, &grad_a_mat);
    }
    if (B_REGULARIZATION) {
      fl::la::AddExpert(2*b_regularization_, b_mat, &grad_b_mat);
    }
    if (C_REGULARIZATION) {
      fl::la::AddExpert(2*c_regularization_, c_mat, &grad_c_mat);
    }


  }

  template<typename WorkSpaceType>
  template<typename TableType,
           bool A_REGULARIZATION,
           bool B_REGULARIZATION,
           bool C_REGULARIZATION,
           bool L2_REGULARIZATION
          >
  double TensorFactorization<WorkSpaceType>::LbfgsFun<
      TableType,
      A_REGULARIZATION,
      B_REGULARIZATION,
      C_REGULARIZATION,
      L2_REGULARIZATION
  >::Evaluate(
      const fl::data::MonolithicPoint<double> &variable) {
    typedef typename TableType::Point_t TPoint_t;
    TPoint_t point;
    typedef typename WorkSpaceType::DefaultTable_t::Point_t FPoint_t;
    double error=0;
    fl::dense::Matrix<double> a_mat;
    fl::dense::Matrix<double> b_mat;
    fl::dense::Matrix<double> c_mat;
    AssignMats(const_cast<fl::data::MonolithicPoint<double>&>(variable), 
        &a_mat, &b_mat, &c_mat);
    for(size_t i=0; i<tensor_->size(); ++i) {
      for(index_t j=0; j<(*tensor_)[i]->n_entries(); ++j) {
        (*tensor_)[i]->get(j, &point);
        for(typename TPoint_t::iterator it=point.begin(); 
            it!=point.end(); ++it) {
          double predicted_value=0;        
          for(size_t k=0; k<rank_; ++k) {
            predicted_value+=a_mat.get(k, j)
              *b_mat.get(k, i)
              *c_mat.get(k, it.attribute());
          }
          error+=fl::math::Pow<double,2,1>(it.value()-predicted_value);
        }     
      }
    }

    if (A_REGULARIZATION) {
      error+=a_regularization_*fl::math::Pow<double, 2, 1>(fl::la::LengthEuclidean(a_mat));          
    }
    if (B_REGULARIZATION) {
      error+=b_regularization_*fl::math::Pow<double, 2, 1>(fl::la::LengthEuclidean(b_mat));          
    }
    if (C_REGULARIZATION) {
      error+=c_regularization_*fl::math::Pow<double, 2, 1>(fl::la::LengthEuclidean(c_mat));            
    }
    eval_result_=error;  
    return error; 
  }

  template<typename WorkSpaceType>
  template<typename TableType,
           bool A_REGULARIZATION,
           bool B_REGULARIZATION,
           bool C_REGULARIZATION,
           bool L2_REGULARIZATION
          >
  const index_t TensorFactorization<WorkSpaceType>::LbfgsFun<
      TableType,
      A_REGULARIZATION,
      B_REGULARIZATION,
      C_REGULARIZATION,
      L2_REGULARIZATION
  >::num_dimensions() const {
    return num_dimensions_;
  }

  template<typename WorkSpaceType>
  template<typename TableType, typename FunType, typename EngineType>
  bool TensorFactorization<WorkSpaceType>::Optimizer(
      std::vector<boost::shared_ptr<TableType> > &tensor, 
      FunType &fun,
      EngineType &engine,
      int32 parafac_rank,
      double a_regularization,
      double b_regularization,
      double c_regularization,
      int32 num_iterations,
      int32 max_num_line_searches_in,
      int32 num_basis,
      fl::data::MonolithicPoint<double> *iterate) {
    fun.Init(&tensor,
              parafac_rank,
              a_regularization,
              b_regularization,
              c_regularization);
    bool optimized;
    
    engine.Init(fun, num_basis);
    engine.set_max_num_line_searches(max_num_line_searches_in);
    fun.Evaluate(*iterate);
    double error=
        double(int32(fl::math::Pow<double, 1, 2>(fun.eval_result()/fun.tensor_sq_norm())
                *10000))/100;
    fl::logger->Message()<<"iteration="<<0
        <<", error="<<error<<"%"<<std::endl;
    if (num_iterations>0) {
      for(int32 i=0; i<num_iterations/5; ++i) {
        optimized=engine.Optimize(5,
                                  iterate);
        double error=
            double(int32(fl::math::Pow<double, 1, 2>(fun.eval_result()/fun.tensor_sq_norm())
                *10000))/100;
        fl::logger->Message()<<"iteration="<<(i+1)*5
                <<", error="<<error<<"%"<<std::endl;
        if (optimized==false) {
          return optimized;
        }
      }
      if (num_iterations % 5>0) {      
        optimized=engine.Optimize(num_iterations % 5, iterate);
        double error=
            double(int32(fl::math::Pow<double, 1, 2>(fun.eval_result()/fun.tensor_sq_norm())
                *10000))/100;
        fl::logger->Message()<<"iteration="<<num_iterations
                <<", error="<<error<<"%"<<std::endl;
      }
    } else {
      optimized=engine.Optimize(num_iterations % 5, iterate);
      double error=
            double(int32(fl::math::Pow<double, 1, 2>(fun.eval_result()/fun.tensor_sq_norm())
                *10000))/100;
      fl::logger->Message()<<"iteration="<<num_iterations
                <<", error="<<error<<"%"<<std::endl;

    }
    return optimized;
  }

  template<typename WorkSpaceType>
  template<typename TableType>
  void TensorFactorization<WorkSpaceType>::ComputeSGD(
      std::vector<boost::shared_ptr<TableType> > &tensor, 
      int32 rank,
      const double a_regularization,
      const double b_regularization,
      const double c_regularization,
      double step0,
      int32 epochs,
      int32 num_iterations,
      boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *a_table,
      boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *b_table,
      boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *c_table) {
 
    FL_SCOPED_LOG(Sdg);
    typedef typename TableType::Point_t TPoint_t;
    typedef typename WorkSpaceType::DefaultTable_t::Point_t DPoint_t;
    TPoint_t tpoint;
    DPoint_t apoint;
    DPoint_t bpoint;
    DPoint_t cpoint;
    fl::dense::Matrix<double> a_mat;
    a_mat.Init((*a_table)->n_attributes(), (*a_table)->n_entries());
    fl::dense::Matrix<double> b_mat;
    b_mat.Init((*b_table)->n_attributes(), (*b_table)->n_entries());
    fl::dense::Matrix<double> c_mat;
    c_mat.Init((*c_table)->n_attributes(), (*c_table)->n_entries());
    for(index_t i=0; i<(*a_table)->n_entries(); ++i) {
      (*a_table)->get(i, &apoint);
      memcpy(a_mat.ptr()+i*rank, apoint.template dense_point<double>().ptr(), rank*sizeof(double));
    }
    for(index_t i=0; i<(*b_table)->n_entries(); ++i) {
      (*b_table)->get(i, &bpoint);
      memcpy(b_mat.ptr()+i*rank, bpoint.template dense_point<double>().ptr(), rank*sizeof(double));
    }
    for(index_t i=0; i<(*c_table)->n_entries(); ++i) {
      (*c_table)->get(i, &cpoint);
      memcpy(c_mat.ptr()+i*rank, cpoint.template dense_point<double>().ptr(), rank*sizeof(double));
    }

    double norm=0;
    index_t n_elements=0;
    for(size_t i=0; i<tensor.size(); ++i) {
      for(index_t j=0; j<tensor[i]->n_entries(); ++j) {
        tensor[i]->get(j, &tpoint);
        for(typename TPoint_t::iterator it=tpoint.begin(); 
            it!=tpoint.end(); ++it) {
          n_elements++;
          norm+=fl::math::Pow<double, 2, 1>(it.value());
        }  
      }
    }
    index_t skipped_updates;
    for(int32 epoch=0; epoch<epochs; ++epoch) {
      double eta=step0/(epoch+1.0);
      for(int32 it=0; it<num_iterations; ++it) {
        skipped_updates=0;
        double total_error=0;
        for(size_t i=0; i<tensor.size(); ++i) {
          for(index_t j=0; j<tensor[i]->n_entries(); ++j) {
            tensor[i]->get(j, &tpoint);
            for(typename TPoint_t::iterator it=tpoint.begin(); 
                it!=tpoint.end(); ++it) {
              double predicted_value=0;        
              for(size_t k=0; k<rank; ++k) {
                predicted_value+=a_mat.get(k, j)
                  *b_mat.get(k, i)
                  *c_mat.get(k, it.attribute());
              }
              double error=-(it.value()-predicted_value);
              fl::dense::Matrix<double> a_vec(rank, 1);
                fl::dense::Matrix<double> b_vec(rank, 1);
                fl::dense::Matrix<double> c_vec(rank, 1);
              for(size_t k=0; k<rank; ++k) {
                double grad_a=
                    2*error*b_mat.get(k, i)*c_mat.get(k, it.attribute());
                if (a_regularization>0) {
                  grad_a+=2*a_regularization*a_mat.get(k, j);
                }
                double grad_b=
                    2*error*a_mat.get(k, j)*c_mat.get(k, it.attribute());
                if (b_regularization>0) {
                  grad_b+=b_mat.get(k, i);
                }
                double grad_c=
                    2*error*a_mat.get(k, j)*b_mat.get(k, i);
                if (c_regularization>0) {
                  grad_c+=c_mat.get(k, it.attribute()); 
                }
                a_vec.set(k, 0, a_mat.get(k, j) - eta * grad_a);
                b_vec.set(k, 0, b_mat.get(k, i) - eta * grad_b);
                c_vec.set(k, 0, c_mat.get(k, it.attribute())-grad_c);
              }
              predicted_value=0;
              for(size_t k=0; k<rank; ++k) {
                predicted_value+=a_vec.get(k, 0)
                  *b_vec.get(k, 0)
                  *c_mat.get(k, 0);
              }
              total_error+=fl::math::Pow<double,2,1>(error);
              if (fabs(error)<eta*fabs(it.value()-predicted_value)) {
                skipped_updates++; 
              } else {
                for(size_t k=0; k<rank; ++k) {
                  a_mat.set(k, j, a_vec.get(k, 0));
                  b_mat.set(k, i, b_vec.get(k, 0));
                  c_mat.set(k, it.attribute(), c_vec.get(k, 0));
                }
              }
            }     
          }
        }
        fl::logger->Message()<<"epoch="<<epoch
        <<", iteration="<<it
        <<", noncontributing update="<<int(10000.0*skipped_updates/n_elements)/100.0<<"%"
        <<", error estimate="<< int(10000.0*sqrt(total_error/norm))/100.0<<"%"<<std::endl;

      }
    }
    for(index_t i=0; i<(*a_table)->n_entries(); ++i) {
      (*a_table)->get(i, &apoint);
      memcpy(apoint.template dense_point<double>().ptr(), a_mat.ptr()+i*rank, rank*sizeof(double));
    }
    for(index_t i=0; i<(*b_table)->n_entries(); ++i) {
      (*b_table)->get(i, &bpoint);
      memcpy(bpoint.template dense_point<double>().ptr(), b_mat.ptr()+i*rank, rank*sizeof(double));
    }
    for(index_t i=0; i<(*c_table)->n_entries(); ++i) {
      (*c_table)->get(i, &cpoint);
      memcpy(cpoint.template dense_point<double>().ptr(), c_mat.ptr()+i*rank, rank*sizeof(double));
    }
  }

  template<typename WorkSpaceType>
  template<typename TableType>
  void TensorFactorization<WorkSpaceType>::ComputeCwopt(
        std::vector<boost::shared_ptr<TableType> > &tensor,
        int32 parafac_rank,
        double a_regularization,
        double b_regularization,
        double c_regularization,
        int32 num_basis,
        int32 max_num_line_searches_in,
        int32 num_iterations,
        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *a_table,
        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *b_table,
        boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> *c_table) {
   
    FL_SCOPED_LOG(Lbfgs);
    bool optimized=false;
    fl::data::MonolithicPoint<double> iterate;
    iterate.Init(
        (tensor.size()
         +tensor[0]->n_entries()
         +tensor[1]->n_attributes())*parafac_rank);
    {
      fl::dense::Matrix<double> a_mat;
      fl::dense::Matrix<double> b_mat;
      fl::dense::Matrix<double> c_mat;
      AssignMats(
        iterate,
        tensor,
        parafac_rank,
        &a_mat,
        &b_mat,
        &c_mat);
      
      typename WorkSpaceType::DefaultTable_t::Point_t point;
      for(index_t i=0; i<(*a_table)->n_entries(); ++i) {
        (*a_table)->get(i, &point);
        memcpy(a_mat.ptr()+i*parafac_rank, point.template dense_point<double>().ptr(), parafac_rank*sizeof(double));      
      }
      for(index_t i=0; i<(*b_table)->n_entries(); ++i) {
        (*b_table)->get(i, &point);
        memcpy(b_mat.ptr()+i*parafac_rank, point.template dense_point<double>().ptr(), parafac_rank*sizeof(double));      
      }
      for(index_t i=0; i<(*c_table)->n_entries(); ++i) {
        (*c_table)->get(i, &point);
        memcpy(c_mat.ptr()+i*parafac_rank, point.template dense_point<double>().ptr(), parafac_rank*sizeof(double));      
      }
    }

    int choice=(a_regularization>0)*4
      +(b_regularization>0)*2
      +(c_regularization>0);
    switch (choice) {
      case 0: 
        {
          Lbfgs<LbfgsFun<TableType, false, false, false, true> > engine;
          LbfgsFun<TableType, false, false, false, true> fun;
          optimized=Optimizer(
            tensor, 
            fun,
            engine,
            parafac_rank,
            a_regularization,
            b_regularization,
            c_regularization,
            num_iterations,
            max_num_line_searches_in,
            num_basis,
            &iterate);

        }
        break;
      case 1:
        {
          Lbfgs<LbfgsFun<TableType, false, false, true, true> > engine;
          LbfgsFun<TableType, false, false, true, true> fun;
          optimized=Optimizer(
            tensor, 
            fun,
            engine,
            parafac_rank,
            a_regularization,
            b_regularization,
            c_regularization,
            num_iterations,
            max_num_line_searches_in,
            num_basis,
            &iterate);
        }
        break;
      case 2:
        {
          Lbfgs<LbfgsFun<TableType, false, true, false, true> > engine;
          LbfgsFun<TableType, false, true, false, true> fun;
          optimized=Optimizer(
            tensor, 
            fun,
            engine,
            parafac_rank,
            a_regularization,
            b_regularization,
            c_regularization,
            num_iterations,
            max_num_line_searches_in,
            num_basis,
            &iterate);
        }
        break;
      case 3:
        {
          Lbfgs<LbfgsFun<TableType, false, true, true, true> > engine;
          LbfgsFun<TableType, false, true, true, true> fun;
          optimized=Optimizer(
            tensor, 
            fun,
            engine,
            parafac_rank,
            a_regularization,
            b_regularization,
            c_regularization,
            num_iterations,
            max_num_line_searches_in,
            num_basis,
            &iterate);
        }
        break;
      case 4:
        {
          Lbfgs<LbfgsFun<TableType, true, false, false, true> > engine;
          LbfgsFun<TableType, true, false, false, true> fun;
          optimized=Optimizer(
            tensor, 
            fun,
            engine,
            parafac_rank,
            a_regularization,
            b_regularization,
            c_regularization,
            num_iterations,
            max_num_line_searches_in,
            num_basis,
            &iterate);
        }
        break;
      case 5:
        {
          Lbfgs<LbfgsFun<TableType, true, false, true, true> > engine;
          LbfgsFun<TableType, true, false, true, true> fun;
          optimized=Optimizer(
            tensor, 
            fun,
            engine,
            parafac_rank,
            a_regularization,
            b_regularization,
            c_regularization,
            num_iterations,
            max_num_line_searches_in,
            num_basis,
            &iterate);
        }
        break;
      case 6:
        {
          Lbfgs<LbfgsFun<TableType, true, true, false, true> > engine;
          LbfgsFun<TableType, true, true, false, true> fun;
          optimized=Optimizer(
            tensor, 
            fun,
            engine,
            parafac_rank,
            a_regularization,
            b_regularization,
            c_regularization,
            num_iterations,
            max_num_line_searches_in,
            num_basis,
            &iterate);
        }
        break;
      case 7:
        {
          Lbfgs<LbfgsFun<TableType, true, true, true, true> > engine;
          LbfgsFun<TableType, true, true, true, true> fun;
          optimized=Optimizer(
            tensor, 
            fun,
            engine,
            parafac_rank,
            a_regularization,
            b_regularization,
            c_regularization,
            num_iterations,
            max_num_line_searches_in,
            num_basis,
            &iterate);
        }
        break;
    }
    if (optimized==false) {
      fl::logger->Warning()<<"Optimization with LBFGS failed"<<std::endl;
    }
    {
      fl::dense::Matrix<double> a_mat;
      fl::dense::Matrix<double> b_mat;
      fl::dense::Matrix<double> c_mat;
      AssignMats(
        iterate,
        tensor,
        parafac_rank,
        &a_mat,
        &b_mat,
        &c_mat);
      
      typename WorkSpaceType::DefaultTable_t::Point_t point;
      for(index_t i=0; i<(*a_table)->n_entries(); ++i) {
        (*a_table)->get(i, &point);
        memcpy(point.template dense_point<double>().ptr(), a_mat.ptr()+i*parafac_rank, parafac_rank*sizeof(double));      
      }
      for(index_t i=0; i<(*b_table)->n_entries(); ++i) {
        (*b_table)->get(i, &point);
        memcpy(point.template dense_point<double>().ptr(), b_mat.ptr()+i*parafac_rank, parafac_rank*sizeof(double));      
      }
      for(index_t i=0; i<(*c_table)->n_entries(); ++i) {
        (*c_table)->get(i, &point);
        memcpy(point.template dense_point<double>().ptr(), c_mat.ptr()+i*parafac_rank, parafac_rank*sizeof(double));      
      }
    }
  }

  template<typename WorkSpaceType>
  template<typename TableType>
  void TensorFactorization<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
    FL_SCOPED_LOG(TensorFactorization);
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "references_prefix_in",
      boost::program_options::value<std::string>(),
      "the reference data "
    )(
      "references_num_in",
      boost::program_options::value<int32>(),
      "number of references file with the prefix defined above"
    )(
      "method",
      boost::program_options::value<std::string>()->default_value("parafac"),
      "The method for factoring a 3 dimensional tensor, it can be \n"
      "  parafac\n"
      "  tucker\n"
    )(
      "algorithm",
      boost::program_options::value<std::string>(),
      "The algorithm for computing the factorizatio.\n"
      "  cpwopt_lbfgs      : for parafac, with lbfgs\n"
      "  cpwopt_sgd       : for paracac, with sgd\n"
      "  cpwopt_sgd_lbfgs : for parafac, with sgd_lbfgs\n"
      "                    : for tucker \n" 
    )(
      "a_factor_out",
      boost::program_options::value<std::string>(),
      "The a factor of the PARAFAC factorization X=ABC" 
    )(
      "b_factor_out",
      boost::program_options::value<std::string>(),
      "The b factor of the PARAFAC factorization X=ABC"
    )(
      "c_factor_out",
      boost::program_options::value<std::string>(),
      "The c factor of the PARAFAC factorization X=ABC"  
    )(
      "a_regularization",
      boost::program_options::value<double>()->default_value(0.0),
      "The regularization for the a matrix"  
    )(
      "b_regularization",
      boost::program_options::value<double>()->default_value(0.0),
      "The regularization for the b matrix"
    )(
      "c_regularization",
      boost::program_options::value<double>()->default_value(0.0),
      "The regularization for the c matrix"
    )(
      "rank",
      boost::program_options::value<int32>()->default_value(5),
      "The factorization rank"
    )(
      "sgd_step0",
      boost::program_options::value<double>()->default_value(1.0),
      "step0 for stochastic gradient descent"
    )(
      "sgd_epochs", 
      boost::program_options::value<int32>()->default_value(100),
      "number of epochs to run in stochastic gradient descent"
    )(
      "sgd_iterations",
      boost::program_options::value<int32>()->default_value(1),
      "number of iterations to run per epoch. In every iteration "
      "sgd sweeps the whole tensor"
    )(
      "lbfgs_rank",
      boost::program_options::value<int32>()->default_value(3),
      "number of basis to use for lbfgs method"  
    )(
      "lbfgs_max_line_searches",
      boost::program_options::value<int32>()->default_value(5),
      "number of line searches to attempt before failing for Lbfgs"  
    )(
      "lbfgs_iterations",
      boost::program_options::value<int32>()->default_value(10),
      "number of iterations to run the Lbfgs optimization"
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

    fl::ws::RequiredArgs(vm, "references_prefix_in");
    fl::ws::RequiredArgs(vm, "references_num_in");

    fl::ws::RequiredArgValues(vm, "method:parafac", 
        "algorithm:cwopt_lbfgs,algorithm:cwopt_sgd,algorithm:cwopt_sgd_lbfgs");
    std::vector<boost::shared_ptr<TableType> > tensor;
    std::string reference_prefix=vm["references_prefix_in"].as<std::string>();
    int32 reference_num=vm["references_num_in"].as<int32>();
    tensor.resize(reference_num);
    fl::logger->Message()<<"Loaded tensor"<<std::endl;
    for(size_t i=0; i<tensor.size(); ++i) {
      ws_->Attach(reference_prefix+boost::lexical_cast<std::string>(i),
          &tensor[i]);
    }
    fl::ws::RequiredArgs(vm, "rank");
    int32 parafac_rank=vm["rank"].as<int32>();
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> a_table;
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> b_table;
    boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> c_table;
    fl::logger->Message()<<"Initializing factors"<<std::endl;
    ws_->Attach(vm.count("a_factor_out")?vm["a_factor_out"].as<std::string>():ws_->GiveTempVarName(),
        std::vector<index_t>(1, parafac_rank),
        std::vector<index_t>(),
        tensor[0]->n_entries(),
        &a_table);
    ws_->Attach(vm.count("b_factor_out")?vm["b_factor_out"].as<std::string>():ws_->GiveTempVarName(),
        std::vector<index_t>(1, parafac_rank),
        std::vector<index_t>(),
        tensor.size(),
        &b_table);
    ws_->Attach(vm.count("c_factor_out")?vm["c_factor_out"].as<std::string>():ws_->GiveTempVarName(),
        std::vector<index_t>(1, parafac_rank),
        std::vector<index_t>(),
        tensor[0]->n_attributes(),
        &c_table);
    {
      typename WorkSpaceType::DefaultTable_t::Point_t point;
      for(index_t i=0; i<a_table->n_entries(); ++i) {
        a_table->get(i, &point);
        point.SetRandom(0.0, 1.0);
      }
      for(index_t i=0; i<b_table->n_entries(); ++i) {
        b_table->get(i, &point);
        point.SetRandom(0.0, 1.0);
      }
      for(index_t i=0; i<c_table->n_entries(); ++i) {
        c_table->get(i, &point);
        point.SetRandom(0.0, 1.0);
      }
    }    

    double a_regularization=0;
    double b_regularization=0;
    double c_regularization=0;
    if (vm.count("a_regularization")>0) {
      a_regularization=vm["a_regularization"].as<double>();
      if (a_regularization<0) {
        fl::logger->Die()<<"--a_regularization must be positive";
      }
    }
    if (vm.count("b_regularization")>0) {
      b_regularization=vm["b_regularization"].as<double>();
      if (b_regularization<0) {
        if (b_regularization<0) {
          fl::logger->Die()<<"--b_regularization must be positive";
        }
      }
    }
    if (vm.count("c_regularization")>0) {
      c_regularization=vm["c_regularization"].as<double>();
      if (c_regularization<0) {
        fl::logger->Die()<<"--c_regularization must be positive";
      }
    }
    const std::string method=vm["method"].as<std::string>();
    const std::string algorithm=vm["algorithm"].as<std::string>();
    int32 rank=vm["rank"].as<int32>();
    fl::logger->Message()<<"Running PARAFAC with stochastic gradient descent"<<std::endl;
    if (algorithm=="cwopt_sgd" || algorithm=="cwopt_sgd_lbfgs") {
      fl::ws::RequiredArgs(vm, "sgd_step0");
      fl::ws::RequiredArgs(vm, "sgd_epochs");
      fl::ws::RequiredArgs(vm, "sgd_iterations");
      double step0=vm["sgd_step0"].as<double>();
      int32 epochs=vm["sgd_epochs"].as<int32>();
      int32 num_iterations=vm["sgd_iterations"].as<int32>();
      TensorFactorization<WorkSpaceType>::ComputeSGD(
        tensor, 
        rank,
        a_regularization,
        b_regularization,
        c_regularization,
        step0,
        epochs,
        num_iterations,
        &a_table,
        &b_table,
        &c_table);
    }
    if (algorithm=="cwopt_lbfgs" || algorithm=="cwopt_sgd_lbfgs") {
      int32 num_basis=vm["lbfgs_rank"].as<int32>();
      int32 max_num_line_searches_in=vm["lbfgs_max_line_searches"].as<int32>();
      int32 num_iterations=vm["lbfgs_iterations"].as<int32>();
      fl::logger->Message()<<"Running PARAFAC with CWOPT algorithm" <<std::endl;
      TensorFactorization<WorkSpaceType>::ComputeCwopt(
          tensor,
          parafac_rank,
          a_regularization,
          b_regularization,
          c_regularization,
          num_basis,
          max_num_line_searches_in,
          num_iterations,
          &a_table,
          &b_table,
          &c_table);
      fl::logger->Message()<<"Finished PARAFAC optimization with CWOPT"<<std::endl;
    }
    ws_->Purge(a_table->filename());
    ws_->Detach(a_table->filename());
    ws_->Purge(b_table->filename());
    ws_->Detach(b_table->filename());
    ws_->Purge(c_table->filename());
    ws_->Detach(c_table->filename());
       
  }

  template<typename WorkSpaceType>
  int TensorFactorization<boost::mpl::void_>::Run(
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
        references_in=tokens[1]+"0";
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
  TensorFactorization<boost::mpl::void_>::Core<WorkSpaceType>::Core(
     WorkSpaceType *ws, const std::vector<std::string> &args) :
   ws_(ws), args_(args)  {}


}}

#endif
