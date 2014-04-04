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
#ifndef FL_LITE_FASTLIB_DATA_MULTIDATASET_DEFS_H_
#define FL_LITE_FASTLIB_DATA_MULTIDATASET_DEFS_H_



namespace fl {
namespace data {
#include "multi_dataset.h"

  template<typename ParameterList>
  template<typename T>
  void MultiDataset<ParameterList>::Init(
    const typename DenseStorageSelection<MonolithicPoint<T> >::type::type &cont) {
    dense_.template get<T>().Alias(cont);
    boost::mpl::if_ <
    boost::is_same<Storage_t, typename DatasetArgs::Compact>,
    InitFromMatrix, InitFromSTL
    >::type::Init(cont , &num_of_points_, &n_attributes_);
    dense_sizes_.resize(1);
    dense_sizes_[0] = n_attributes_;
    num_of_metadata_ = MetaDataType_t::size;
  
  }


  template<typename ParameterList>
  template<typename ContainerType>
  void MultiDataset<ParameterList>::Init(ContainerType dense_dimensions,
                                         ContainerType sparse_dimensions,
                                         index_t num_of_points) {
    num_of_metadata_ = MetaDataType_t::size;
    static const index_t num_of_dense_types =
      boost::mpl::size<DenseTypeList_t>::value;
    static const index_t num_of_sparse_types =
      boost::mpl::size<SparseTypeList_t>::value;
  
    if (num_of_dense_types != static_cast<index_t>(dense_dimensions.size())) {
      fl::logger->Die() << "Number of dense types declared in the file is "
      << dense_dimensions.size()
      << " while this class is designed for "
      << num_of_dense_types
      << " dense types";
  
    }
    if (num_of_sparse_types != static_cast<index_t>(sparse_dimensions.size())) {
      fl::logger->Die() << "Number of sparse types declared in the file is "
      << sparse_dimensions.size()
      << " while this class is designed for "
      << num_of_sparse_types << " sparse types";
    }
  
    num_of_points_ = num_of_points;
    typename ContainerType::iterator it;
    for (it = dense_dimensions.begin(); it != dense_dimensions.end(); ++it) {
      dense_sizes_.push_back(*it);
    }
  
    for (it = sparse_dimensions.begin(); it != sparse_dimensions.end(); ++it) {
      sparse_sizes_.push_back(*it);
    }
  
    index_t count = 0;
    boost::mpl::for_each<DenseTypeList_t>(Initializer<DenseBox, DenseIterators>(&dense_,
                                          &dense_its_, &dense_sizes_, &count, num_of_points_));
    count = 0;
    boost::mpl::for_each<SparseTypeList_t>(Initializer<SparseBox, SparseIterators>(&sparse_,
                                           &sparse_its_, &sparse_sizes_, &count, num_of_points_));
    // Allocate containers for meta_data
    if (HasMetaData_t::value == true) {
      meta_.resize(num_of_points_);
      meta_it_ = meta_.begin();
    }
    n_attributes_ = 0;
    for (index_t i = 0; i < static_cast<index_t>(dense_sizes_.size()); i++) {
      n_attributes_ += dense_sizes_[i];
    }
    for (index_t i = 0; i < static_cast<index_t>(sparse_sizes_.size()); i++) {
      n_attributes_ += sparse_sizes_[i];
    }
  }

  template<typename ParameterList>
  template<typename ContainerType>
  void MultiDataset<ParameterList>::InitRandom(ContainerType dense_dimensions,
       ContainerType sparse_dimensions,
       index_t num_of_points,
       const typename MultiDataset<ParameterList>::CalcPrecision_t low, 
       const typename MultiDataset<ParameterList>::CalcPrecision_t hi,
       const typename MultiDataset<ParameterList>::CalcPrecision_t sparsity) {
  
    Init(dense_dimensions, sparse_dimensions, num_of_points);
    Point_t point;
    for(index_t i=0; i<this->n_points(); ++i) {
      get(i, &point);
      point.SetRandom(low, hi, sparsity);
    }
  }

  template<typename ParameterList>
  template<typename Archive>
  void MultiDataset<ParameterList>::save(Archive &ar,
                                         const unsigned int version) const {
    try {
      ar << boost::serialization::make_nvp("num_of_points", num_of_points_);
      ar << boost::serialization::make_nvp("meta", meta_);
      ar << boost::serialization::make_nvp("labels", labels_);
      // do not uncomment this, it is toxic
 //     ar << boost::serialization::make_nvp("load_meta", load_meta_);
 //     instead do the following
      int32 dummy=load_meta_;
      ar << boost::serialization::make_nvp("load_meta", dummy);
      ar << boost::serialization::make_nvp("dense_sizes", dense_sizes_);
      ar << boost::serialization::make_nvp("sparse_sizes",sparse_sizes_);
      ar << boost::serialization::make_nvp("ignored_dense_colunmns", ignored_dense_columns_);
      ar << boost::serialization::make_nvp("num_of_metadata", num_of_metadata_);
      ar << boost::serialization::make_nvp("n_attributes", n_attributes_);
     // ar << boost::serialization::make_nvp("is_2_tokens", is_2_tokens_);
     // ar << boost::serialization::make_nvp("comma_flag", comma_flag_);
      ar << boost::serialization::make_nvp("delimeter", delimiter_);
      ar << boost::serialization::make_nvp("header", header_);
      ar << boost::serialization::make_nvp("dense", dense_);
      ar << boost::serialization::make_nvp("sparse", sparse_);
    }
    catch(const boost::archive::archive_exception &e) {
      fl::logger->Die()<< "Multidataset archiving (save): "<< e.what();
    }
  }
  
  template<typename ParameterList>
  template<typename Archive>
  void MultiDataset<ParameterList>::load(Archive &ar,
                                         const unsigned int version) {
    try {
      ar >> boost::serialization::make_nvp("num_of_points", num_of_points_);
      ar >> boost::serialization::make_nvp("meta", meta_);
      ar >> boost::serialization::make_nvp("labels", labels_);
     // do not uncomment this, it is toxic
     // ar >> boost::serialization::make_nvp("load_meta", load_meta_);
     // instead do the following
      int32 dummy;
      ar >> boost::serialization::make_nvp("load_meta", dummy);
      load_meta_=static_cast<bool>(dummy);
      ar >> boost::serialization::make_nvp("dense_sizes", dense_sizes_);
      ar >> boost::serialization::make_nvp("sparse_sizes", sparse_sizes_);
      ar >> boost::serialization::make_nvp("ignored_dense_colunmns", ignored_dense_columns_);
      ar >> boost::serialization::make_nvp("num_of_metadata", num_of_metadata_);
      ar >> boost::serialization::make_nvp("n_attributes", n_attributes_);
      //ar >> boost::serialization::make_nvp("is_2_tokens", is_2_tokens_);
      //ar >> boost::serialization::make_nvp("comma_flag", comma_flag_);
      ar >> boost::serialization::make_nvp("delimeter", delimiter_);
      ar >> boost::serialization::make_nvp("header", header_);
      ar >> boost::serialization::make_nvp("dense" ,dense_);
      ar >> boost::serialization::make_nvp("sparse", sparse_);

      Reset();
    }
    catch(const boost::archive::archive_exception &e) {
      fl::logger->Die()<< "Multidataset archiving (load): "<< e.what();
    }
  }
  
 

}
} // namespaces
#endif

