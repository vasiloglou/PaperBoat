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
#ifndef FL_LITE_FASTLIB_DATA_MULTIDATASET_STORAGE_SELECTION_H_
#define FL_LITE_FASTLIB_DATA_MULTIDATASET_STORAGE_SELECTION_H_
/**
 * @brief This class chooses the right container for storing points
 *        if we require compact and static storage we use
 *        Matrices,  if we want to append points we use a vector
 *        that contains points. If we want  to be able to add and delete
 *        we use a list of points
 */
template<typename T>
class DenseStorageSelection : public
      boost::mpl::if_ < boost::is_same < Storage_t,
      typename DatasetArgs::Compact > , CompactContainer<T>,
      typename boost::mpl::if_ < boost::is_same < Storage_t,
      typename DatasetArgs::Extendable > , ExtendableContainer<T>,
      typename boost::mpl::if_ < boost::is_same<Storage_t, DatasetArgs::Deletable>,
      DeletableContainer<T>, boost::mpl::void_ >::type >::type >::type {
  public:
    typedef typename boost::mpl::if_ < boost::is_same < Storage_t,
    typename DatasetArgs::Compact > , CompactContainer<T>,
    typename boost::mpl::if_ < boost::is_same < Storage_t,
    typename DatasetArgs::Extendable > , ExtendableContainer<T>,
    typename boost::mpl::if_ < boost::is_same<Storage_t, DatasetArgs::Deletable>,
    DeletableContainer<T>, boost::mpl::void_ >::type >::type >::type type;
};

template<typename T>
class SparseStorageSelection : public
      boost::mpl::if_ < boost::is_same < Storage_t,
      typename DatasetArgs::Compact > , ExtendableContainer<T>,
      typename boost::mpl::if_ < boost::is_same < Storage_t,
      typename DatasetArgs::Extendable > , ExtendableContainer<T>,
      typename boost::mpl::if_ < boost::is_same<Storage_t, DatasetArgs::Deletable>,
      DeletableContainer<T>, boost::mpl::void_ >::type >::type >::type::type {
  public:
    typedef typename boost::mpl::if_ < boost::is_same < Storage_t,
    typename DatasetArgs::Compact > , ExtendableContainer<T>,
    typename boost::mpl::if_ < boost::is_same < Storage_t,
    typename DatasetArgs::Extendable > , ExtendableContainer<T>,
    typename boost::mpl::if_ < boost::is_same<Storage_t, DatasetArgs::Deletable>,
    DeletableContainer<T>, boost::mpl::void_ >::type >::type >::type::type type;
};

template<typename T>
class MetaDataStorageSelection : public
      boost::mpl::if_ <
      boost::is_same <
      Storage_t,
      typename DatasetArgs::Compact
      > ,
      ExtendableContainer<T>,
      typename boost::mpl::if_ <
      boost::is_same <
      Storage_t,
      typename DatasetArgs::Extendable
      > ,
      ExtendableContainer<T>,
      typename boost::mpl::if_ <
      boost::is_same <
      Storage_t,
      DatasetArgs::Deletable
      > ,
      DeletableContainer<T>, boost::mpl::void_ >::type >::type >::type::type {
  public:
    typedef typename boost::mpl::if_ < boost::is_same < Storage_t,
    typename DatasetArgs::Compact > , ExtendableContainer<T>,
    typename boost::mpl::if_ < boost::is_same < Storage_t,
    typename DatasetArgs::Extendable > , ExtendableContainer<T>,
    typename boost::mpl::if_ < boost::is_same<Storage_t, DatasetArgs::Deletable>,
    DeletableContainer<T>, boost::mpl::void_ >::type >::type >::type::type type;
    friend class boost::serialization::access;
    template<typename Archive>
    void save(Archive &ar,
                const unsigned int file_version) const {
      ar << boost::serialization::make_nvp("vector", 
          boost::serialization::base_object<type>(*this));
    }

    template<typename Archive>
    void load(Archive &ar,
              const unsigned int file_version) {
      ar >> boost::serialization::make_nvp("vector", 
          boost::serialization::base_object<type>(*this));
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

};



#endif
