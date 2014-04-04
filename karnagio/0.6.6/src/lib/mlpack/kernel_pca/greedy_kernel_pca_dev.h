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
#ifndef FL_LITE_MLPACK_KERNEL_PCA_GREEDY_KERNEL_PCA_DEV_H
#define FL_LITE_MLPACK_KERNEL_PCA_GREEDY_KERNEL_PCA_DEV_H

#include "mlpack/kernel_pca/greedy_kernel_pca.h"
#include "mlpack/kernel_pca/greedy_kernel_pca_defs.h"
#include "mlpack/kernel_pca/dense_kernel_matrix_inverse.h"
#include "fastlib/util/timer.h"

namespace fl {
namespace ml {

template<typename TableType, bool do_centering>
template<typename KernelType>
void GreedyKernelPca<TableType, do_centering>::
Dictionary<KernelType>::
ComputeProjections(
  const KernelType &kernel,
  fl::dense::Matrix<double, false> *projections) {

  // Temporary vector used for computing the kernel values.
  fl::dense::Matrix<double, true> temp_kernel_vector;
  temp_kernel_vector.Init(this->size());

  // Initialize the projection matrix.
  projections->Init(this->size(), table_->n_entries());

  // Loop and compute the projection of each point.
  for (int i = 0; i < table_->n_entries(); i++) {

    int training_point_index = random_permutation_[i];
    fl::dense::Matrix<double, true> projections_column;
    projections->MakeColumnVector(i, &projections_column);

    if (in_dictionary_[ training_point_index ] == false) {

      // Get the current point to project.
      typename TableType::Dataset_t::Point_t current_point_to_project;
      table_->get(training_point_index , &current_point_to_project);

      // Compute the kernel value of the current point against the
      // points in the dictionary.
      for (int j = 0; j < this->size(); j++) {
        typename TableType::Dataset_t::Point_t current_dictionary_point;
        table_->get(point_indices_in_dictionary_[j],
                    &current_dictionary_point);
        temp_kernel_vector[j] = kernel.Dot(current_dictionary_point,
                                           current_point_to_project);

      }

      fl::dense::ops::Mul<fl::la::Overwrite>(
        *current_kernel_matrix_inverse_, temp_kernel_vector,
        &projections_column);
    }
    else {
      int position_in_dictionary =
        training_index_to_dictionary_position_[ training_point_index ];
      projections_column.SetZero();
      projections_column[ position_in_dictionary ] = 1.0;
    }

  } // end of looping through points to be projected.

  // If the centering is required, then we need to subtract the
  // average column vector from each projection.
  if (do_centering) {

    fl::dense::Matrix<double, true> average;
    average.Init(projections->n_rows());
    average.SetZero();
    for (int i = 0; i < table_->n_entries(); i++) {
      fl::dense::Matrix<double, true> projections_column;
      projections->MakeColumnVector(i, &projections_column);
      fl::dense::ops::AddTo(projections_column, &average);
    }
    fl::dense::ops::SelfScale(1.0 / ((double) table_->n_entries()),
                              &average);
    for (int i = 0; i < table_->n_entries(); i++) {
      fl::dense::Matrix<double, true> projections_column;
      projections->MakeColumnVector(i, &projections_column);
      fl::dense::ops::SubFrom(average, &projections_column);
    }
  }
}

template<typename TableType, bool do_centering>
template<typename KernelType>
void GreedyKernelPca<TableType, do_centering>::
Dictionary<KernelType>::UpdateDictionary_(
  const int &new_point_index,
  const fl::dense::Matrix<double, true> &temp_kernel_vector,
  const double &projection_error,
  const fl::dense::Matrix<double, true>
  &inverse_times_kernel_vector) {

  // Add the point to the dictionary.
  point_indices_in_dictionary_.push_back(new_point_index);
  in_dictionary_[ new_point_index ] = true;
  training_index_to_dictionary_position_[ new_point_index ] =
    point_indices_in_dictionary_.size() - 1;

  // Update the kernel matrix.
  fl::dense::Matrix<double, false> *new_kernel_matrix =
    new fl::dense::Matrix<double, false>();
  new_kernel_matrix->Init(current_kernel_matrix_->n_rows() + 1,
                          current_kernel_matrix_->n_cols() + 1);

  for (int j = 0; j < current_kernel_matrix_->n_cols(); j++) {
    for (int i = 0; i < current_kernel_matrix_->n_rows(); i++) {
      new_kernel_matrix->set(i, j, current_kernel_matrix_->get(i, j));
    }
  }
  for (int j = 0; j < current_kernel_matrix_->n_cols(); j++) {
    new_kernel_matrix->set(j, current_kernel_matrix_->n_cols(),
                           temp_kernel_vector[j]);
    new_kernel_matrix->set(current_kernel_matrix_->n_rows(), j,
                           temp_kernel_vector[j]);
  }
  new_kernel_matrix->set(current_kernel_matrix_->n_rows(),
                         current_kernel_matrix_->n_cols(), 1.0);

  delete current_kernel_matrix_;
  current_kernel_matrix_ = new_kernel_matrix;

  // Update the kernel matrix inverse.
  fl::dense::Matrix<double, false> *new_kernel_matrix_inverse =
    fl::ml::DenseKernelMatrixInverse::Update(
      *current_kernel_matrix_inverse_,
      inverse_times_kernel_vector, projection_error);
  delete current_kernel_matrix_inverse_;
  current_kernel_matrix_inverse_ = new_kernel_matrix_inverse;

  // Update the kernel matrix inverse sum and the row sum.
  current_kernel_matrix_inverse_sum_ = 0;
  fl::dense::Matrix<double, true> *new_kernel_matrix_inverse_row_sum =
    new fl::dense::Matrix<double, true>();
  new_kernel_matrix_inverse_row_sum->Init(
    current_kernel_matrix_inverse_row_sum_->length() + 1);
  new_kernel_matrix_inverse_row_sum->SetZero();
  for (int j = 0; j < current_kernel_matrix_inverse_->n_cols(); j++) {
    for (int i = 0; i < current_kernel_matrix_inverse_->n_rows(); i++) {
      (*new_kernel_matrix_inverse_row_sum)[i] +=
        current_kernel_matrix_inverse_->get(i, j);
      current_kernel_matrix_inverse_sum_ +=
        current_kernel_matrix_inverse_->get(i, j);
    }
  }
  delete current_kernel_matrix_inverse_row_sum_;
  current_kernel_matrix_inverse_row_sum_ =
    new_kernel_matrix_inverse_row_sum;
}

template<typename TableType, bool do_centering>
template<typename KernelType>
fl::dense::Matrix<double, false>
*GreedyKernelPca<TableType, do_centering>::
Dictionary<KernelType>::current_kernel_matrix() {

  return current_kernel_matrix_;
}

template<typename TableType, bool do_centering>
template<typename KernelType>
fl::dense::Matrix<double, false>
*GreedyKernelPca<TableType, do_centering>::
Dictionary<KernelType>::current_kernel_matrix_inverse() {

  return current_kernel_matrix_inverse_;
}

template<typename TableType, bool do_centering>
template<typename KernelType>
void GreedyKernelPca<TableType, do_centering>::Dictionary<KernelType>::set_dictionary_limit(
    index_t dictionary_limit) {
  dictionary_limit_=dictionary_limit;
}

template<typename TableType, bool do_centering>
template<typename KernelType>
void GreedyKernelPca<TableType, do_centering>::Dictionary<KernelType>::set_greedy_kernel_pca_threshold(double greedy_kernel_pca_threshold) {
  greedy_kernel_pca_threshold_=greedy_kernel_pca_threshold;
}

template<typename TableType, bool do_centering>
template<typename KernelType>
int GreedyKernelPca<TableType, do_centering>::
Dictionary<KernelType>::size() const {
  return point_indices_in_dictionary_.size();
}

template<typename TableType, bool do_centering>
template<typename KernelType>
void GreedyKernelPca<TableType, do_centering>::
Dictionary<KernelType>::AddBasis(
  const int &iteration_number,
  const KernelType &kernel) {
  FL_SCOPED_LOG(AddBasis);
  if (point_indices_in_dictionary_.size()>=dictionary_limit_) {
    fl::logger->Warning()<<"Dictionary is full, cannot add more vectors";
    return;
  }
  // The new point to consider for adding.
  int new_point_index = random_permutation_[iteration_number];

  // The vector for storing kernel values.
  fl::dense::Matrix<double, true> temp_kernel_vector;
  temp_kernel_vector.Init(point_indices_in_dictionary_.size());

  // Get the new candidate point.
  typename TableType::Dataset_t::Point_t new_point;
  table_->get(new_point_index, &new_point);

  // Compute the kernel value between the new candidate point and
  // the previously existing basis points.
  for (int i = 0; i < (int) point_indices_in_dictionary_.size(); i++) {
    typename TableType::Dataset_t::Point_t basis_point;
    table_->get(point_indices_in_dictionary_[i], &basis_point);
    temp_kernel_vector[i] = kernel.Dot(new_point,
                                       basis_point);
  }
  // Compute the matrix-vector product.
  fl::dense::Matrix< double, true > inverse_times_kernel_vector;
  fl::dense::ops::Mul<fl::la::Init>(*current_kernel_matrix_inverse_,
                                    temp_kernel_vector,
                                    &inverse_times_kernel_vector);

  // Compute the projection error.
  double projection_error =
    kernel.NormSq(new_point) -
    fl::la::Dot(temp_kernel_vector, inverse_times_kernel_vector);
  // If the projection error is above the threshold, add it to the
  // dictionary.
  if (projection_error > greedy_kernel_pca_threshold_) {

    UpdateDictionary_(new_point_index, temp_kernel_vector, projection_error,
                      inverse_times_kernel_vector);
  }
}

template<typename TableType, bool do_centering>
template<typename KernelType>
void GreedyKernelPca<TableType, do_centering>::
Dictionary<KernelType>::Init(
  TableType &table_in,
  const KernelType &kernel_in) {

  table_ = &table_in;
  kernel_ = &kernel_in;

  // Allocate the boolean flag for the presence of each training
  // point in the dictionary.
  in_dictionary_.resize(table_in.n_entries());
  training_index_to_dictionary_position_.resize(table_in.n_entries());

  // Generate a random permutation and initialize the inital
  // dictionary which consists of the first random point.
  random_permutation_.resize(table_in.n_entries());
  for (int i = 0; i < table_in.n_entries(); i++) {
    random_permutation_[i] = i;
    in_dictionary_[i] = false;
    training_index_to_dictionary_position_[i] = -1;
  }
  RandomPermutation_(random_permutation_);

  // The first random point goes into the initial dictionary.
  typename TableType::Dataset_t::Point_t first_point;
  table_->get(random_permutation_[0], &first_point);
  point_indices_in_dictionary_.push_back(random_permutation_[0]);
  in_dictionary_[ random_permutation_[0] ] = true;
  training_index_to_dictionary_position_[ random_permutation_[0] ] = 0;

  // Dynamic allocations of matrices.
  current_kernel_matrix_ = NULL;
  current_kernel_matrix_inverse_ = NULL;
  current_kernel_matrix_inverse_row_sum_ = NULL;

  current_kernel_matrix_ = new fl::dense::Matrix<double, false>();
  current_kernel_matrix_inverse_ = new fl::dense::Matrix<double, false>();
  current_kernel_matrix_inverse_row_sum_ =
    new fl::dense::Matrix<double, true>();

  current_kernel_matrix_->Init(1, 1);
  current_kernel_matrix_inverse_->Init(1, 1);

  // Set to the default values for the dictionary.
  current_kernel_matrix_->set(0, 0,
                              kernel_in.NormSq(first_point));
  current_kernel_matrix_inverse_->set(
    0, 0, 1.0 / current_kernel_matrix_->get(0, 0));
  current_kernel_matrix_inverse_sum_ =
    current_kernel_matrix_inverse_->get(0, 0);
  current_kernel_matrix_inverse_row_sum_->Init(
    current_kernel_matrix_inverse_->n_rows());
  current_kernel_matrix_inverse_row_sum_->set(
    0, 0, current_kernel_matrix_inverse_sum_);
}

template<typename TableType, bool do_centering>
template<typename KernelType>
void GreedyKernelPca<TableType, do_centering>::
Dictionary<KernelType>::RandomPermutation_(std::vector<int> &permutation) {

  for (int i = 0; i < (int) permutation.size(); i++) {
    int random_index = fl::math::Random(i, int(permutation.size()));
    std::swap(permutation[i], permutation[random_index]);
  }
}

template<typename TableType, bool do_centering>
template<typename KernelType, typename ResultType>
void GreedyKernelPca<TableType, do_centering>::Train(
  TableType &training_set,
  const KernelType &kernel,
  const int &num_components,
  double greedy_kernel_pca_threshold, 
  index_t dictionary_limit,
  ResultType *result) {
  FL_SCOPED_LOG(Train);
  // The intial dictionary consists of the first point in the
  // randomly shuffled list.
  Dictionary<KernelType> current_dictionary;
  current_dictionary.set_dictionary_limit(dictionary_limit);
  current_dictionary.set_greedy_kernel_pca_threshold(greedy_kernel_pca_threshold);
  current_dictionary.Init(training_set, kernel);

  // The main loop.
  for (int t = 1; t < training_set.n_entries(); t++) {
    current_dictionary.AddBasis(t, kernel);
    if (current_dictionary.size()>=dictionary_limit) {
      fl::logger->Message()<<"Dictionary limit ("<<dictionary_limit<<") was reached, "
        "while ("<<t<<") points were used. The rest ("<<training_set.n_entries()-t
        <<") will not be used"<<std::endl;
      break;
    }
  }
  fl::logger->Message() << "Compressed " << training_set.n_entries()
  << " training points to " << current_dictionary.size()
  << " dictionary points."<<std::endl;

  // Allocate the result.
  if (result->IsInitialized() == false) {
    result->Init(training_set, num_components);
  }

  // Compute the projection of the training set on the final
  // dictionary.
  fl::dense::Matrix<double, false> training_set_projections;
  fl::logger->Message()<<"Computing projections on the dictionary"<<std::endl;
  current_dictionary.ComputeProjections(kernel, &training_set_projections);

  // Solve the generalized eigenvalue problem and extract the
  // eigenvectors.
  fl::logger->Message()<<"Computing the outer product"<<std::endl;
  success_t success_flag;
  fl::dense::Matrix<double, false> projection_matrix_outerproduct;
  fl::dense::Matrix<double, false> kernel_submatrix_copy;
  kernel_submatrix_copy.Copy(*(current_dictionary.current_kernel_matrix()));
  projection_matrix_outerproduct.Init(current_dictionary.size(),
                                      current_dictionary.size());
  projection_matrix_outerproduct.SetZero();
  std::cout<<training_set_projections.n_rows()<<" x "<<training_set_projections.n_cols()<<std::endl;
  index_t n_rows=training_set_projections.n_rows();
  index_t n_cols=training_set_projections.n_cols();
  fl::util::Timer timer;
  timer.Start();
  for (int i = 0; i < n_cols; i++) {
    index_t offset=i*n_rows;
    for (int k = 0; k < n_rows; k++) {
      for (int j = k; j < n_rows; j++) {
        *(projection_matrix_outerproduct.ptr()+j+k*n_rows)+=
          *(training_set_projections.ptr()+j+offset) 
          * *(training_set_projections.ptr()+k+offset);
      }
    }
  }
  for(int i=0; i<n_rows; ++i) {
    for(int j=i; i<n_rows; ++i) {
      training_set_projections.set(j, 
          i,
          training_set_projections.get(i, j));
    }
  }
  timer.End();
  fl::logger->Message()<<"Total time for outer product: "<<timer.GetTotalElapsedTime();
  fl::dense::Matrix<double, true> eigenvalues;

  fl::logger->Message()<<"Computing EigenValue decomposition"<<std::endl;
  // Solve the generalized eigenvalue problem.
  fl::dense::ops::GenEigenSymmetric(3, &projection_matrix_outerproduct,
                                    &kernel_submatrix_copy,
                                    &eigenvalues,
                                    &success_flag);

  // Normalize the eigenvalues by the number of training points.
  for (int i = 0; i < num_components; i++) {
    typename ResultType::ResultTableType::Point_t point1;
    result->principal_eigenvalues()->get(i, &point1);
    point1.set(0, eigenvalues[eigenvalues.length() - i - 1] /
               ((double) training_set.n_entries()));
  }
  fl::logger->Message()<<"Final projection"<<std::endl;
  for (int i = 0; i < num_components; i++) {
    double length = 0;
    for (int j = 0; j < training_set.n_entries(); j++) {

      // Compute the dot-product between the j-th projection in
      // the list and the i-th eigenvector.
      double dot_product = 0;
      for (int k = 0; k < training_set_projections.n_rows(); k++) {
        dot_product += training_set_projections.get(k, j) *
                       projection_matrix_outerproduct.get(
                         k, eigenvalues.length() - i - 1);
      }

      int position_to_training_point_index =
        current_dictionary.position_to_training_index_map(j);
      typename ResultType::ResultTableType::Point_t point;
      result->principal_components()->get(
        position_to_training_point_index, &point);
      point.set(i, dot_product);
      length += fl::math::Sqr(dot_product);
    }
    DEBUG_ASSERT(length != 0);

    // Take the square root so that you get the Euclidean length.
    length = sqrt(length);
    // Normalize the eigenvector.
    for (int j = 0; j < training_set.n_entries(); j++) {
      typename ResultType::ResultTableType::Point_t point;
      result->principal_components()->get(j, &point);
      point.template 
        dense_point<typename ResultType::ResultTableType::Point_t::CalcPrecision_t>().
        set(i, 
          point.template dense_point<typename 
            ResultType::ResultTableType::Point_t::CalcPrecision_t>().get(i) / length);
    }
  }

  fl::logger->Message()<<"Done"<<std::endl;
}
};
};

#endif
