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

#ifndef FL_LITE_FASTLIB_DATA_MULTI_DATASET_DEV_H_
#define FL_LITE_FASTLIB_DATA_MULTI_DATASET_DEV_H_
#include "fastlib/data/multi_dataset.h"
#include <sstream>

namespace fl { namespace data {

  template<typename ParameterList>
  template<typename PointType, typename MetaDataType>
  void  MultiDataset<ParameterList>::SetMetaDataTrait1::type::set_meta_data(
      PointType &p, MetaDataType m) {
  }

  template<typename ParameterList>
  template<typename PointType, typename MetaDataType>
  void  MultiDataset<ParameterList>::SetMetaDataTrait2::type::set_meta_data(
      PointType &p, MetaDataType m) {
    p->set_meta_data(m);
  }

  template<typename ParameterList>
  MultiDataset<ParameterList>::ExportedPointCollection::ExportedPointCollection() {
  }

  template<typename ParameterList>
  MultiDataset<ParameterList>::ExportedPointCollection::ExportedPointCollection(
      const DenseBox *dense1,
      const SparseBox *sparse1,
      const MetaDataBox *meta1) {
    dense = const_cast<DenseBox *>(dense1);
    sparse = const_cast<SparseBox *>(sparse1);
    meta_data = const_cast<MetaDataBox *>(meta1);
  }

  template<typename ParameterList>
  template<typename MetaDataType>
  void MultiDataset<ParameterList>::MetaDataAlias1::type::Alias(
      MetaDataType *value, Point_t *entry) {
    entry->meta_data_ptr() = value;
  }

  template<typename ParameterList>
  template<typename MetaDataType>
  void MultiDataset<ParameterList>::MetaDataAlias2::type::Alias(
      MetaDataType *value, Point_t *entry) {
  }

  template<typename ParameterList>
  MultiDataset<ParameterList>::DenseAlias::DenseAlias(
      DenseBox *box, index_t ind, Point_t *entry) {
    box_   = box;
    ind_   = ind;
    entry_ = entry;

  }

  template<typename ParameterList>
  template<typename T>
  MonolithicPoint<T> &MultiDataset<ParameterList>::DenseAlias::MonolithicPointOperator::Do(
      MonolithicPoint<T> *point) {
    return *point;
  }

  template<typename ParameterList>
  template<typename T>
  MonolithicPoint<T> &MultiDataset<ParameterList>::DenseAlias::MixedPointOperator::Do(
      Point_t *point) {
    return point->template dense_point<T>();
  }

  template<typename ParameterList>
  template<typename T>
  void MultiDataset<ParameterList>::DenseAlias::CompactOperator::Do(
      DenseBox *box, index_t i, MonolithicPoint<T> &point) {
    DenseBox::template get<T>(*box).MakeColumnVector(i, &point);
  }

  template<typename ParameterList>
  template<typename T>
  void MultiDataset<ParameterList>::DenseAlias::ExtendableOperator::Do(
      DenseBox *box, index_t i, MonolithicPoint<T> &point) {
    point.Alias(DenseBox::template get<T>(*box)[i]);
  }

  template<typename ParameterList>
  template<typename T>
  void MultiDataset<ParameterList>::DenseAlias::operator()(T) {
    typedef typename boost::mpl::if_c < IsMonolithic<Point_t>::value,
    MonolithicPointOperator, MixedPointOperator >::type Operator1;

    typedef typename boost::mpl::if_ < boost::is_same<Storage_t, DatasetArgs::Compact>,
    CompactOperator, ExtendableOperator >::type Operator2;
    Operator2::template Do<T>(box_, ind_, Operator1::template Do<T>(entry_));
  }

  template<typename ParameterList>
  template<typename PointType>
  void MultiDataset<ParameterList>::NullaryMetaFunctionSetSize1::type::set_size(
      PointType *p, index_t size) {
  
  }

  template<typename ParameterList>
  template<typename PointType>
  void MultiDataset<ParameterList>::NullaryMetaFunctionSetSize2::type::set_size(
      PointType *p, index_t size) {
    p->size_ = size;
  }

  template<typename ParameterList>
  MultiDataset<ParameterList>::SparseAlias::SparseAlias(
      SparseBox *box, index_t ind, Point_t *entry) {
    box_   = box;
    ind_   = ind;
    entry_ = entry;
  }

  template<typename ParameterList>
  template<typename T>
  typename MultiDataset<ParameterList>::Point_t &MultiDataset<ParameterList>::SparseAlias::
      SparsePointOperator::Do(Point_t *point) {
    return *point;
  }

  template<typename ParameterList>
  template<typename T>
  SparsePoint<T> &MultiDataset<ParameterList>::SparseAlias::
      MixedPointOperator::Do(Point_t *point) {
    return point->template sparse_point<T>();
  }

  template<typename ParameterList>
  template<typename T>
  void MultiDataset<ParameterList>::SparseAlias::operator()(T) {
    typedef typename boost::mpl::if_ < boost::is_same<Point_t, SparsePoint<T> >,
    SparsePointOperator, MixedPointOperator >::type Operator1;
    Operator1::template Do<T>(entry_).Alias(SparseBox::template get<T>(*box_)[ind_]);
  }

  template<typename ParameterList>
  template<typename BoxType, typename IteratorsType>
  MultiDataset<ParameterList>::
      ResetIterators<BoxType,IteratorsType>::ResetIterators(BoxType *box,
                 IteratorsType *its) {
    box_ = box;
    its_ = its;
  }

  template<typename ParameterList>
  template<typename BoxType, typename IteratorsType>
  template<typename T>
  void MultiDataset<ParameterList>::
      ResetIterators<BoxType,IteratorsType>::operator()(T) {
    IteratorsType::template get<T>(*its_) = BoxType::template get<T>(*box_).begin();
    
  }

  template<typename ParameterList>
  template<typename PrecisionType>
  bool MultiDataset<ParameterList>::ChooseDenseIterator<PrecisionType>::type::HasNext(
      DenseIterators &dense_its,
      DenseBox &dense_box,
      SparseIterators &sparse_its,
      SparseBox &sparse_box) {
    return DenseIterators::template get<PrecisionType>(dense_its) != DenseBox::template
    get<PrecisionType>(dense_box).end();
  }

  template<typename ParameterList>
  template<typename PrecisionType>
  bool MultiDataset<ParameterList>::ChooseSparseIterator<PrecisionType>::type::HasNext(
      DenseIterators &dense_its,
      DenseBox &dense_box,
      SparseIterators &sparse_its,
      SparseBox &sparse_box) {
    return SparseIterators::template get<PrecisionType>(sparse_its) != SparseBox::template
    get<PrecisionType>(sparse_box).end();
  }

  template<typename ParameterList>
  template<typename PointType>
  MultiDataset<ParameterList>::MakePointFromDenseIterators<PointType>::
      MakePointFromDenseIterators(DenseIterators *its, Point_t *point) {
    its_ = its;
    point_ = point;
  }

  template<typename ParameterList>
  template<typename PointType>
  template<typename T>
  void MultiDataset<ParameterList>::MakePointFromDenseIterators<PointType>::operator()(T) {
    point_->template dense_point<T>().Alias(*(its_->template get<T>()));
  }

  template<typename ParameterList>
  template<typename T>
  MultiDataset<ParameterList>::MakePointFromDenseIterators<MonolithicPoint<T> >::
      MakePointFromDenseIterators(DenseIterators *its, Point_t *point) {
    its_ = its;
    point_ = point;
  }

  template<typename ParameterList>
  template<typename T1>
  template<typename T>
  void MultiDataset<ParameterList>::MakePointFromDenseIterators<MonolithicPoint<T1> >::operator()(T) {
    point_->Alias(*(its_->template get<T>()));
  }

  template<typename ParameterList>
  template<typename PointType>
  MultiDataset<ParameterList>::MakePointFromSparseIterators<PointType>::
      MakePointFromSparseIterators(SparseIterators *its, Point_t *point) {
    its_ = its;
    point_ = point;
  }

  template<typename ParameterList>
  template<typename PointType>
  template<typename T>
  void MultiDataset<ParameterList>::MakePointFromSparseIterators<PointType>::operator()(T) {
    point_->template sparse_point<T>().Alias(*(its_->template get<T>()));
  }
 
  template<typename ParameterList>
  template<typename T1>
  MultiDataset<ParameterList>::MakePointFromSparseIterators<SparsePoint<T1> >::
      MakePointFromSparseIterators(SparseIterators *its, Point_t *point) {
    its_ = its;
    point_ = point;
  }

  template<typename ParameterList>
  template<typename T1>
  template<typename T>
  void MultiDataset<ParameterList>::MakePointFromSparseIterators<SparsePoint<T1> >::operator()(T) {
    point_->Alias(*(its_->template get<T>()));
  }
   
  template<typename ParameterList>
  template<typename T>
  void MultiDataset<ParameterList>::InitFromMatrix::Init(
      T &mat, index_t *num_of_points, index_t *n_attributes) {
    *num_of_points  = mat.n_cols();
    *n_attributes = mat.n_rows();
  }

  template<typename ParameterList>
  template<typename T>
  void MultiDataset<ParameterList>::InitFromSTL::Init(
      T &v, index_t *num_of_points, index_t *n_attributes) {
    *num_of_points  = static_cast<index_t>(v.size());
    *n_attributes = static_cast<index_t>(v[0].size());
  }

  template<typename ParameterList>
  template<typename BoxType, typename Iterators>
  template<typename T>
  MonolithicPoint<T> &MultiDataset<ParameterList>::PushBackOperator<BoxType, Iterators>::
      DenseCase::get(Point_t *point) {
    return boost::mpl::if_ <
        boost::is_same<Point_t, MonolithicPoint<T> >,
        Get2,
        Get1
      >::type:: template get<T>(point);
  }

  template<typename ParameterList>
  template<typename BoxType, typename Iterators>
  template<typename T>
  MonolithicPoint<T> &MultiDataset<ParameterList>::PushBackOperator<BoxType, Iterators>::
      DenseCase::Get1::get(Point_t *point) {
    return point->template dense_point<T>();
  }

  template<typename ParameterList>
  template<typename BoxType, typename Iterators>
  template<typename T>
  MonolithicPoint<T> &MultiDataset<ParameterList>::PushBackOperator<BoxType, Iterators>::
      DenseCase::Get2::get(MonolithicPoint<T> *point) {
    return *point;
  }

  template<typename ParameterList>
  template<typename BoxType, typename Iterators>
  template<typename T>
  SparsePoint<T> &MultiDataset<ParameterList>::PushBackOperator<BoxType, Iterators>::
      SparseCase::get(Point_t *point) {
    return boost::mpl::if_ <
        boost::is_same<Point_t, SparsePoint<T> >,
        Get2,
        Get1
      >::type:: template get<T>(point);
  }

  template<typename ParameterList>
  template<typename BoxType, typename Iterators>
  template<typename T>
  SparsePoint<T> &MultiDataset<ParameterList>::PushBackOperator<BoxType, Iterators>::
      SparseCase::Get1::get(Point_t *point) {
    return point->template sparse_point<T>();
  }

  template<typename ParameterList>
  template<typename BoxType, typename Iterators>
  template<typename T>
  SparsePoint<T> &MultiDataset<ParameterList>::PushBackOperator<BoxType, Iterators>::
      SparseCase::Get2::get(SparsePoint<T> *point) {
    return *point;
  }

  template<typename ParameterList>
  template<typename BoxType, typename IteratorsType>
  MultiDataset<ParameterList>::PushBackOperator<BoxType, IteratorsType>::
      PushBackOperator(BoxType *box, IteratorsType *its, Point_t *point) {
    box_ = box;
    its_ = its;
    point_ = point;
  }

  template<typename ParameterList>
  template<typename BoxType, typename Iterators>
  template<typename T>
  void MultiDataset<ParameterList>::PushBackOperator<BoxType, Iterators>::operator()(T) {
    typedef typename boost::mpl::if_ < boost::is_same<BoxType, DenseBox>,
    DenseCase, SparseCase >::type  Accessor;
    box_->template get<T>().push_back(Accessor::template get<T>(point_));
    its_->template get<T>() = box_->template get<T>().end() - 1;
  }


  template<typename ParameterList>
  template<typename MetaDataBoxType, typename PointType>
  void MultiDataset<ParameterList>::SetMetaDataOperator1::type::
      Set(PointType &point, MetaDataBoxType *box) {
    if (point.meta_data_ptr() != NULL) {
      box->push_back(point.meta_data());
    } else {
      try {
        box->resize(box->size() + 1);
      }
      catch(const std::bad_alloc &e) {
        fl::logger->Die() << "Problems in allocating, the dataset is "
          << "probably too big";
      }
    }
  }

  template<typename ParameterList>
  template<typename MetaDataBoxType, typename PointType>
  void MultiDataset<ParameterList>::SetMetaDataOperator2::type::
      Set(PointType &point, MetaDataBoxType *box) {
  }

  template<typename ParameterList>
  typename MultiDataset<ParameterList>::CalcPrecision_t MultiDataset<ParameterList>::get(
      index_t i, index_t j) const {
     Point_t entry;
     const_cast<MultiDataset<ParameterList> *>(this)->get(i, &entry);
     return entry[j];
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::get(index_t point_id,
                                        typename MultiDataset<ParameterList>::Point_t *entry)  {
    BOOST_MPL_ASSERT((boost::mpl::not_ < boost::is_same < Storage_t,
                      DatasetArgs::Deletable > >));
    DEBUG_ASSERT(point_id < num_of_points_);
    DEBUG_ASSERT(point_id >= 0);
    boost::mpl::for_each<DenseTypeList_t>(DenseAlias(&dense_, point_id, entry));
    boost::mpl::for_each<SparseTypeList_t>(SparseAlias(&sparse_, point_id, entry));

    boost::mpl::eval_if <
    IsMixed_t,
    NullaryMetaFunctionSetSize2,
    NullaryMetaFunctionSetSize1
    >::type::set_size(entry, n_attributes_);
    if (HasMetaData_t::value==true) {
      boost::mpl::eval_if <
        HasMetaData_t,
        MetaDataAlias1,
        MetaDataAlias2
      >::type::Alias(&meta_[point_id], entry);
    }
  }
 
  template<typename ParameterList>
  void MultiDataset<ParameterList>::push_back(
    typename MultiDataset<ParameterList>::Point_t &point) {
    boost::mpl::for_each<DenseTypeList_t>(
      PushBackOperator<DenseBox, DenseIterators>(&dense_,
          &dense_its_, &point));
    boost::mpl::for_each<SparseTypeList_t>(
      PushBackOperator<SparseBox, SparseIterators>(&sparse_,
          &sparse_its_, &point));
    boost::mpl::eval_if <
    IsMixed_t,
    SetMetaDataOperator1,
    SetMetaDataOperator2
    >::type::Set(point, &meta_);
    if (boost::is_same < Storage_t,
        typename DatasetArgs::Deletable >::value) {
      ++meta_it_;
    }
    else  {
      meta_it_ = meta_.end() - 1;
    }
    num_of_points_++;
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::push_back(
    std::string &point) {
    Point_t  dummy_point;
    push_back(dummy_point);
    index_t count = 0;
    boost::mpl::for_each<DenseTypeList_t>(InitBoxPoint<DenseIterators>(&dense_its_, &dense_sizes_, &count, num_of_points_));
    count = 0;
    boost::mpl::for_each<SparseTypeList_t>(InitBoxPoint<SparseIterators>(&sparse_its_, &sparse_sizes_, &count, num_of_points_));
    if (ignored_dense_columns_.size()!=0) {
      AddPoint(point, ignored_dense_columns_);
    } else {
      AddPointFast(point);
    }
  }
 
  template<typename ParameterList>
  void MultiDataset<ParameterList>::Reset() {
    boost::mpl::for_each<DenseTypeList_t>(ResetIterators<DenseBox, DenseIterators>(
                                            &dense_, &dense_its_));
    boost::mpl::for_each<SparseTypeList_t>(ResetIterators<SparseBox, SparseIterators>(
                                             &sparse_, &sparse_its_));
    if (HasMetaData_t::value == true) {
      meta_it_ = meta_.begin();
    }
  }
  
  template<typename ParameterList>
  bool MultiDataset<ParameterList>::HasNext() {
    typedef typename boost::mpl::eval_if <
    boost::mpl::empty<DenseTypeList_t>,
    boost::mpl::front<SparseTypeList_t>,
    boost::mpl::front<DenseTypeList_t>
    >::type PrecisionType;
  
    typedef typename boost::mpl::eval_if <
    boost::mpl::empty<DenseTypeList_t>,
    ChooseSparseIterator<PrecisionType>,
    ChooseDenseIterator<PrecisionType>
    >::type Choice;
    return Choice::HasNext(dense_its_, dense_, sparse_its_, sparse_);
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::Next(
    typename MultiDataset<ParameterList>::Point_t *point) {
    boost::mpl::for_each<DenseTypeList_t>(MakePointFromDenseIterators<Point_t>(
                                            &dense_its_, point));
    boost::mpl::for_each<SparseTypeList_t>(MakePointFromSparseIterators<Point_t>(
                                             &sparse_its_, point));
    boost::mpl::eval_if <
    IsMixed_t,
    NullaryMetaFunctionSetSize2,
    NullaryMetaFunctionSetSize1
    >::type::set_size(point, n_attributes_);
  
  
    if (HasMetaData_t::value == true) {
      boost::mpl::eval_if <
      HasMetaData_t,
      SetMetaDataTrait2,
      SetMetaDataTrait1
      >::type::set_meta_data(point, *meta_it_);
      ++meta_it_;
    }
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::Init(std::string filename,
                                         std::string mode) {
    Init(filename, ignored_dense_columns_, mode);
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::Init(std::string header) {
    ParseHeader(header);
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::Init(std::string filename,
                                         std::vector<index_t> &ignored_dense_columns,
                                         std::string mode) {
    FL_SCOPED_LOG(MultiDataset); 
    num_of_metadata_ = MetaDataType_t::size;
    ignored_dense_columns_ = ignored_dense_columns;
    load_meta_ = false;
    if (mode == "r") {
      num_of_points_ = 0;
      // we need this variable in case the file is just a dense csv
      index_t single_dense_csv_dimension = -1;
      num_of_points_ = 0;
      index_t lines_to_skip = 0;
      std::ifstream fin(filename.c_str(), std::ios_base::in);
      if (fin.fail()) {
        fl::logger->Die() << "Could not open file " << filename.c_str()
        << "   error: " << strerror(errno);
    
      }
      std::string line;
      std::getline(fin, line);
      if (fin.fail()) {
        fl::logger->Die() << "Something went wrong while reading";
      }
      if (!fin.eof()) {
        // find out if it has header information
        boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
        if (line.find("header") != std::string::npos) {
          lines_to_skip++;
          try {
            ParseHeader(line);
          }
          catch(const fl::TypeException &e) {
            fl::logger->Die() << e.what();
          }
          if (std::getline(fin, line).good()) {
            boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
            // find out if it has labels
            if (line.find("attribute_names") != std::string::npos
                || line.find("labels") != std::string::npos) {
              lines_to_skip++;
              ParseLabels(line);
              if (std::getline(fin, line).good()) {
                boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
                FindDelimeter(line);
                num_of_points_++;
              }
              else {
                fl::logger->Die() << "Something went wrong while reading";
              }
            }
            else {
              num_of_points_ = 1;
              FindDelimeter(line);
            }
          }
          else {
            fl::logger->Die() << "Something went wrong while reading";
          }
        }
        else {
          if (line.find("attribute_names") != std::string::npos
              || line.find("labels") != std::string::npos) {
            lines_to_skip++;
            ParseLabels(line);
            if (std::getline(fin, line).good()) {
              boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
              num_of_points_ = 1;
              FindDelimeter(line);
            }
            else {
              fl::logger->Die() << "Something went wrong while reading";
            }
          }
          else {
            num_of_points_ = 1;
            FindDelimeter(line);
          }
          // if no header has been provided then the file can only be
          // dense csv file, so we need to determine the dimension of the points
          std::vector<std::string> temporary_tokens;
          boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
          boost::algorithm::split(temporary_tokens, line, boost::algorithm::is_any_of(" ,\t"));
          single_dense_csv_dimension = static_cast<index_t>(temporary_tokens.size());
          if (HasMetaData_t::value == true && load_meta_ == true) {
            single_dense_csv_dimension -= MetaDataType_t::size;
          }
  
        }
      } else {
        num_of_points_=1;
        FindDelimeter(line);
        std::vector<std::string> temporary_tokens;
        boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
        boost::algorithm::split(temporary_tokens, line, boost::algorithm::is_any_of(" ,\t"));
        single_dense_csv_dimension = static_cast<index_t>(temporary_tokens.size());
        if (HasMetaData_t::value == true && load_meta_ == true) {
          single_dense_csv_dimension -= MetaDataType_t::size;
        }
      }
        
      // find out the number of points;
      while (fin.good()) {
        std::getline(fin, line);
        // if the file contains only sparse and not metadata then a blank line
        // means all zero point
        if ((line.size() == 0) && (HasDense_t::value == true  || load_meta_ == true)) {
          break;
        }
        num_of_points_++;
      }
      if (!(HasDense_t::value == true  || load_meta_ == true)) {
        --num_of_points_;
      }
  
      // now go back in the begining of the file
      fin.close();
      fin.open(filename.c_str(), std::ios_base::in);
      fin.clear();
      for (index_t i = 0; i < lines_to_skip; i++) {
        std::getline(fin, line);
      }
  
      // Allocate the space for the containers
      // and start reading
      index_t count = 0;
      // this is an indication that the file is dense csv, it didn't have a header
      // maybe it had labels. In any event we have determined the dimension of the points
      // and we have to pass it in the vector with the dimensions
      if (single_dense_csv_dimension > 0) {
        dense_sizes_.push_back(single_dense_csv_dimension);
      }
      index_t cumsum = 0;
      for (size_t i = 0; i < dense_sizes_.size(); i++) {
        cumsum += dense_sizes_[i];
        size_t j = 0;
        while (j < ignored_dense_columns_.size() &&  ignored_dense_columns_[j] < cumsum) {
          dense_sizes_[i] -= 1;
          j++;
        }
      }
      // Allocate containers for meta_data
      if (HasMetaData_t::value == true) {
        try {
          meta_.resize(num_of_points_);
        } 
        catch(const std::bad_alloc &) {
          fl::logger->Die() << "Problems in allocating memory, the "
            << "dataset is probably too big too fit in RAM."
            << "It might be that your dataset is too big to fit in RAM or "
            << "you are using a 32bit platform which limits the process address space "
            << "to 4GB";

        }
        meta_it_ = meta_.begin();
      }
      boost::mpl::for_each<DenseTypeList_t>(Initializer<DenseBox, DenseIterators>(&dense_,
                                            &dense_its_, &dense_sizes_, &count, num_of_points_));
      count = 0;
      boost::mpl::for_each<SparseTypeList_t>(Initializer<SparseBox, SparseIterators>(&sparse_,
                                             &sparse_its_, &sparse_sizes_, &count, num_of_points_));
      n_attributes_ = 0;
      for (index_t i = 0; i < static_cast<index_t>(dense_sizes_.size()); i++) {
        n_attributes_ += dense_sizes_[i];
      }
      for (index_t i = 0; i < static_cast<index_t>(sparse_sizes_.size()); i++) {
        n_attributes_ += sparse_sizes_[i];
      }
      if (ignored_dense_columns_.size()!=0) {
        for (index_t i = 0; i < num_of_points_; ++i) {
          std::getline(fin, line);
          AddPoint(line, ignored_dense_columns_);
        }
      } else {
         for (index_t i = 0; i < num_of_points_; ++i) {
          std::getline(fin, line);
          AddPointFast(line);
        }
     
      }
      return;
    }
    fl::logger->Die() << "This option " << mode.c_str() << "is not supported";
  }  
  
  
  template<typename ParameterList>
  index_t MultiDataset<ParameterList>::n_points() const {
    return num_of_points_;
  }
  
  template<typename ParameterList>
  index_t MultiDataset<ParameterList>::n_attributes() const {
    return n_attributes_;
  }
  
  template<typename ParameterList>
  typename MultiDataset<ParameterList>::ExportedPointCollection_t
  MultiDataset<ParameterList>::point_collection() const {
    return ExportedPointCollection_t(&dense_, &sparse_, &meta_);
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::Save(const std::string file,
                                         const bool header,
                                         const std::vector<std::string> labels,
                                         const std::string delimiter)  {
  
    std::ofstream fout(file.c_str(), std::ios_base::out);
    if (fout.fail()) {
      fl::logger->Die() << "Could not open file "
      << std::string(file.c_str())
      << " error "
      << strerror(errno);
    }
    Save(fout, header, labels, delimiter);
  
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::Save(std::ostream &out,
                                         const bool header,
                                         const std::vector<std::string> labels,
                                         const std::string delimiter)  {
    delimiter_=delimiter;
    if (delimiter == ",") {
      comma_flag_ = true;
    }
    else {
      if (delimiter == " " || delimiter == "\t") {
        comma_flag_ = false;
      }
      else {
        fl::logger->Die() << "This delimiter " << delimiter.c_str()
        << " is not supported";
      }
    }
    if (header == true) {
      WriteHeader(out);
    }
    if (!labels.empty()) {
      WriteLabels(out, labels);
    }
  
    Reset();
    index_t dense_offset=0;
    for(index_t i=0; i<dense_sizes_.size(); ++i) {
      dense_offset+=dense_sizes_[i];
    }
    while (HasNext()) {
      boost::mpl::for_each<boost::mpl::range_c<int, 0, MetaDataType_t::size> >(
        MetaSaver(meta_it_,
                  num_of_metadata_,
                  delimiter,
                  &out));
      if(HasMetaData_t::value) {
        ++meta_it_;
      }  
      boost::mpl::for_each<DenseTypeList_t>(DenseSavePoint(&dense_its_,
                                            &out,
                                            &delimiter));
      index_t offset=dense_offset;
      index_t cur_size=0;
      boost::mpl::for_each<SparseTypeList_t>(SparseSavePoint(&sparse_its_,
                                             &out,
                                             &offset,
                                             &cur_size, 
                                             sparse_sizes_,  
                                             &delimiter));
      out << "\n";
    }
  
  }
  
  template<typename ParameterList>
  std::vector<std::string>  &MultiDataset<ParameterList>::labels() {
    return labels_;
  }
  
  template<typename ParameterList>
  const std::vector<std::string>  &MultiDataset<ParameterList>::labels() const {
    return labels_;
  }
  
  template<typename ParameterList>
  const std::vector<index_t> &MultiDataset<ParameterList>::dense_sizes() const {
    return dense_sizes_;
  }
  
  template<typename ParameterList>
  const std::vector<index_t> &MultiDataset<ParameterList>::sparse_sizes() const {
    return sparse_sizes_;
  }
  
  template<typename ParameterList>
  const std::string MultiDataset<ParameterList>::GetHeader() const {
    return header_;
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::SetHeader(std::string header_in) {
    header_ = header_in;
  }
  
  template<typename ParameterList>
  bool MultiDataset<ParameterList>::TryToInit(const std::string &filename) {
        std::ifstream fin(filename.c_str(), std::ios_base::in);
    if (fin.fail()) {
      fl::logger->Die() << "Could not open file " << filename.c_str()
          << "   error: " << strerror(errno);
    
    }
    std::string line;
    if (std::getline(fin, line).good()) {
      // find out if it has header information
      boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
      if (line.find("header") != std::string::npos) {
        ParseHeader(line);
        dense_sizes_.clear();
        sparse_sizes_.clear();
      } 
    }
    return true;  
  }
 
 
  template<typename ParameterList>
  bool MultiDataset<ParameterList>::FindDelimeter(std::string &line) {
    std::vector<std::string> tokens;
    boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
    if (line.find(",") != std::string::npos) {
      delimiter_=",";
      comma_flag_ = true;
      boost::algorithm::split(tokens, line, boost::algorithm::is_any_of(","));
    }
    else {
  // we have to remove this because it will not work with points of dimension one    
  //    if (line.find(" ") == std::string::npos &&
  //        line.find("\t") == std::string::npos) {
  //      fl::logger->Die() << "This file is neither comma"
  //      << " nor space/tab separated";
  //    }
  //    else {    
        if (line.find(" ") != std::string::npos) {
          delimiter_=" ";
        } else {
          delimiter_="\t";
        }
        boost::algorithm::split(tokens, line, boost::algorithm::is_any_of(" \t"));
  //    }
    }
    is_2_tokens_ = true;
    bool found_at_least_one_token=false;
    for (index_t i = 0; i < static_cast<index_t>(tokens.size()); i++) {
      if (tokens[i].find(":") != std::string::npos) {
        found_at_least_one_token=true;
        std::vector<std::string> toks;
        boost::algorithm::split(toks, tokens[i], boost::algorithm::is_any_of(":"));
        if (toks.size() == 2) {
          is_2_tokens_ = true;
        }
        break;
      }
    }
    if (found_at_least_one_token==false && 
        boost::mpl::empty<SparseTypeList_t>::value==false) {
      fl::logger->Warning() << "The sparse format couldn't be determined"
        << ", it is assumed you are using index:value format without "
        << "specifying the precision" << std::endl;
    }
    return found_at_least_one_token;
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::WriteHeader(std::ostream &out) {
    std::string delimiter;
    if (comma_flag_ == true) {
      delimiter = ",";
    }
    else {
      delimiter = " ";
    }
    out << "header" << delimiter;
    index_t ind = 0;
    if (HasMetaData_t::value == true) {
      out << "meta:" << num_of_metadata_ << delimiter;
    }
    boost::mpl::for_each<DenseTypeList_t>(WriteHeaderOperator(&out, &dense_sizes_,
                                          &ind, "", delimiter));
    ind=0;
    boost::mpl::for_each<SparseTypeList_t>(WriteHeaderOperator(&out, &sparse_sizes_,
                                           &ind, "sparse:", delimiter));
    out << "\n";
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::WriteLabels(std::ostream &out,
      const std::vector<std::string> &labels) {
    if (static_cast<index_t>(labels.size()) != n_attributes_) {
      fl::logger->Die() << "You provided "
      << labels.size()
      << "labels but the dataset has "
      << n_attributes_
      << " attributes.";
    }
    std::string delimiter;
    if (comma_flag_ == true) {
      delimiter = ",";
    }
    else {
      delimiter = " ";
    }
    out << "attribute_names" << delimiter;
    for (index_t i = 0; i < static_cast<index_t>(labels.size()); i++) {
      out << labels[i] << delimiter;
    }
    out << "\n";
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::ParseHeader(std::string &line) {
    header_ = line;
    static const index_t num_of_dense_types =
      boost::mpl::size<DenseTypeList_t>::value;
    static const index_t num_of_sparse_types =
      boost::mpl::size<SparseTypeList_t>::value;
    // Analyze the dense types
    std::vector<std::string> tokens;
    std::vector<std::string> dense_tok_vec;
    std::vector<std::string> sparse_tok_vec;
    boost::algorithm::split(tokens, line, boost::algorithm::is_any_of(" ,\t"));
    // if the user has commas and space we will get some empty strings
    tokens.erase(std::remove(tokens.begin(), tokens.end(), ""), tokens.end());
    //check if it has meta data in the header
    load_meta_ = false;
    bool found_meta_tag = false;
    if (tokens[1].find("meta", 0) != std::string::npos) {
      found_meta_tag = true;
      std::vector<std::string> two_tokens;
      boost::algorithm::split(two_tokens, tokens[1],
                              boost::algorithm::is_any_of(":"));
     
      if (two_tokens.size() != 2) {
        fl::logger->Die() << "The meta header is wrong "
        << "\"" << tokens[1] << "\""
        << ". It should be in this form meta:number";
      }
      try {  
        num_of_metadata_ = boost::lexical_cast<index_t>(two_tokens[1]);
      }
      catch(const boost::bad_lexical_cast &e) {
        fl::logger->Die() << "There is something wrong in the header of your file:\n"
          << line <<"\n"
          << "It might be the precision declaration, the acceptable types are:\n"
          << "double, float, char, int, long, long_long, unsinged_char,\n"
          << "unsigned_int, unsigned_long, unsigned_long_long";
      }

      if (num_of_metadata_ != 0) {
        load_meta_ = true;
      }
      int temp_size = MetaDataType_t::size;
      if (num_of_metadata_ > temp_size)  {
        fl::logger->Die() << "The number of metadata declared in the header "
        << num_of_metadata_
        << " is greater from what the program supports "
        << temp_size;
      }
    } else {
      fl::logger->Warning()<<"We didn't see a meta:3 keyword in the header of your file. "
        <<"If you have the 3 first columns of your file filled with metadata and you forgot "
        <<"the meta:3 keyword, bad things will happen!!!!";
    }
    // we skip the first token which is the tag "header"
    index_t counter;
    for (counter = found_meta_tag ? 2 : 1; counter < static_cast<index_t>(tokens.size()); counter++) {
      if (tokens[counter].size() == 0) {
        continue;
      }
      std::vector<std::string> two_or_three_tokens;
      boost::algorithm::split(two_or_three_tokens,
                              tokens[counter], boost::algorithm::is_any_of(":"));
      if (two_or_three_tokens.size() == 2) {
        dense_tok_vec.push_back(two_or_three_tokens[0]);
        try {
          dense_sizes_.push_back(boost::lexical_cast<index_t>(two_or_three_tokens[1]));
        }
        catch(const boost::bad_lexical_cast &e) {
          fl::logger->Die() << "There is something wrong in the header.\n"
            << line << "\n"
            << "The definition of the dense or sparse dimensions seems to be wrong\n"
            << "It might be the precision declaration, the acceptable types are:\n"
            << "double, float, char, int, long, long_long, unsinged_char,\n"
            << "unsigned_int, unsigned_long, unsigned_long_long";        

        }
      }
      else {
        // If the tokens are three then we jump
        // in the sparse cases
        if (two_or_three_tokens.size() == 3) {
          break;
        }
        else {
          fl::logger->Die() << "The header can have 2 or three tokens, "
          "for dense points (i.e type:size, ie double:10, "
          "or sparse:float:300) "
          "this token "
          << tokens[counter].c_str() << "in the header was wrong";
        }
      }
      two_or_three_tokens.clear();
    }
  
    for (; counter < static_cast<index_t>(tokens.size()); counter++) {
      std::vector<std::string> two_or_three_tokens;
      boost::algorithm::split(two_or_three_tokens,
                              tokens[counter], boost::algorithm::is_any_of(":"));
      if (two_or_three_tokens.size() == 3) {
        sparse_tok_vec.push_back(two_or_three_tokens[1]);
        try {
          sparse_sizes_.push_back(boost::lexical_cast<index_t>(two_or_three_tokens[2]));
        }
        catch(const boost::bad_lexical_cast &e) {
          fl::logger->Die() << "There is something wrong in the header.\n"
            << line << "\n"
            << "The definition of the dense or sparse dimensions seems to be wrong\n"
            << "It might be the precision declaration, the acceptable types are:\n"
            << "double, float, char, int, long, long_long, unsinged_char,\n"
            << "unsigned_int, unsigned_long, unsigned_long_long";        
        }

      }
      else {
        fl::logger->Die() << "The header for the sparse point "
        "can have 3 tokens (sparse:type:size, "
        "ie sparse:float:1120), "
        "this token "
        << tokens[counter].size() << " in the header was wrong";
      }
  
      two_or_three_tokens.clear();
    }
  
    if (num_of_dense_types != static_cast<index_t>(dense_tok_vec.size())) {
      std::ostringstream s1;
      s1 << "[DATA TYPE ERROR] Number of dense types declared in the file is "
      << dense_tok_vec.size()
      << " while this class is designed for "
      << num_of_dense_types
      << " dense types\n"
      << "This error was probably generated because you have set the --point flag "
      << "incorrectly, it appears that your file is sparse and you are treating it as "
      << "dense, please rerun your program with the --help flag to see more  options "
      << "for the --point option";
      throw fl::TypeException(s1.str());
  
    }
    if (num_of_sparse_types != static_cast<index_t>(sparse_tok_vec.size())) {
      std::ostringstream s1;
      s1 << "[DATA TYPE ERROR] Number of sparse types declared in the file is "
      << sparse_tok_vec.size()
      << " while this class is designed for "
      << num_of_sparse_types << " sparse types\n"
      << "This error was probably generated because you have set the --point flag "
      << "incorrectly, it appears that your file is dense and you are treating it as "
      << "sparse, please rerun your program with the --help flag to see more  options "
      << "for the --point option";
      throw fl::TypeException(s1.str());
    }
    counter = 0;
    boost::mpl::for_each<DenseTypeList_t>(
      CheckTypes(&dense_tok_vec, &counter));
    counter = 0;
    boost::mpl::for_each<SparseTypeList_t>(
      CheckTypes(&sparse_tok_vec, &counter));
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::ParseLabels(std::string &line) {
    boost::algorithm::split(labels_, line, boost::algorithm::is_any_of(" ,\t"));
    // remove the first token which will be "attribute_names" or "labels"
    labels_.erase(labels_.begin());
  }
  
  template<typename ParameterList>
  void MultiDataset<ParameterList>::AddPoint(std::string &line,
      std::vector<index_t> &ignored_dense_columns) {
    std::deque<std::string> tokens;
    boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
    boost::algorithm::split(tokens, line, boost::algorithm::is_any_of(" ,\t"));
    // loading meta data
    if (load_meta_ == true) {
      boost::mpl::for_each<boost::mpl::range_c<int, 0, MetaDataType_t::size> >(
        MetaLoader(meta_it_, num_of_metadata_, &tokens));
    }
    ++meta_it_;
    for (size_t i = 0; i < ignored_dense_columns.size(); i++) {
      tokens.erase(tokens.begin() + ignored_dense_columns[i]);
    }
    // the elements are in the file in an increasing order,
    // we need to offset them so that all containers are 0 based
    index_t offset = 0;
    boost::mpl::for_each<DenseTypeList_t>(
      DensePointLoader(&dense_its_, &tokens, &offset));
    boost::mpl::for_each<SparseTypeList_t>(
      SparsePointLoader(is_2_tokens_, &sparse_its_, &tokens, &offset));
  
  }

  template<typename ParameterList>
  void MultiDataset<ParameterList>::AddPointFast(std::string &line) {
    char *tok=strtok(const_cast<char*>(line.c_str()), delimiter_.c_str()); 
    // loading meta data
    if (load_meta_ == true) {
      boost::mpl::for_each<boost::mpl::range_c<int, 0, MetaDataType_t::size> >(
        MetaLoaderFast(meta_it_, delimiter_, num_of_metadata_, &tok));
	  ++meta_it_;
    }
    
    // the elements are in the file in an increasing order,
    // we need to offset them so that all containers are 0 based
    index_t offset = 0;
    boost::mpl::for_each<DenseTypeList_t>(
      DensePointLoaderFast(&dense_its_, delimiter_, &tok, &offset));
    boost::mpl::for_each<SparseTypeList_t>(
      SparsePointLoaderFast(&sparse_its_, delimiter_, &tok, &offset));
  
  }

}}
#endif
