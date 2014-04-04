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
#ifndef FL_LITE_FASTLIB_DATA_MULTIDATASET_INITIALIZER_H_
#define FL_LITE_FASTLIB_DATA_MULTIDATASET_INITIALIZER_H_
template<typename BoxType, typename IteratorsType>
struct Initializer {
  struct STLInit {
    template<typename Container>
    STLInit(Container *cont, index_t size, index_t dimension) {
      try {
        cont->resize(size);
      }
      catch(const std::bad_alloc &e) {
        fl::logger->Die()<< "I cannot allocate "
          << size << "bytes for storing the dataset in RAM"
          << "It might be that your dataset is too big to fit in RAM or "
          << "you are using a 32bit platform which limits the process address space "
          << "to 4GB";
      }
      typename Container::iterator it;
      for (it = cont->begin(); it != cont->end(); ++it) {
        try {
          it->Init(dimension);
        }
        catch(const std::bad_alloc &e) {
          fl::logger->Die()<<"The dataset is probably too big "
            <<"I cannot allocate memory "
            << "It might be that your dataset is too big to fit in RAM or "
            << "you are using a 32bit platform which limits the process address space "
            << "to 4GB";
        }
      }
    }
  };

  struct FastlibInit {
    template<typename Container>
    FastlibInit(Container *cont, index_t size, index_t dimension) {
      cont->Init(dimension, size);
    }
  };

  Initializer(BoxType *box,
              IteratorsType *its,
              std::vector<index_t> *dimensions,
              index_t *count,
              index_t num_of_points) {
    box_ = box;
    its_ = its;
    dimensions_ = dimensions;
    count_ = count;
    num_of_points_ = num_of_points;
  }

  template<typename T>
  void operator()(T) {
    typedef typename boost::mpl::if_ < boost::mpl::and_ < boost::is_same<BoxType, DenseBox>,
    boost::is_same<Storage_t, typename DatasetArgs::Compact> > ,
    FastlibInit, STLInit >::type Init;
    typedef typename
    boost::mpl::if_ < boost::is_same<BoxType, DenseBox>,
    wrap<MonolithicPoint<T>, DenseStorageSelection>,
    wrap<SparsePoint<T>, SparseStorageSelection> >::type    this_wrap;

    Init(&BoxType::template get<T>(*box_), num_of_points_, (*dimensions_)[*count_]);
    (*count_)++;
    IteratorsType::template get<T>(*its_) = BoxType::template get<T>(*box_).begin();

  }
private:
  BoxType *box_;
  IteratorsType *its_;
  std::vector<index_t> *dimensions_;
  index_t *count_;
  index_t num_of_points_;
};

template<typename IteratorsType>
struct InitBoxPoint {
  InitBoxPoint(IteratorsType *its,
               std::vector<index_t> *dimensions,
               index_t *count,
               index_t num_of_points) {

    its_ = its;
    dimensions_ = dimensions;
    count_ = count;
    num_of_points_ = num_of_points;
  }

  template<typename T>
  void operator()(T) {
    its_->template get<T>()->Init((*dimensions_)[*count_]);
    (*count_)++;

  }
private:
  IteratorsType *its_;
  std::vector<index_t> *dimensions_;
  index_t *count_;
  index_t num_of_points_;
};


#endif
