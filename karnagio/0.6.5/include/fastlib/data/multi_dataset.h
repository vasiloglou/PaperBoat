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
#ifndef FL_LITE_FASTLIB_DATA_DATASET_H_
#define FL_LITE_FASTLIB_DATA_DATASET_H_

#include <string>
#include <vector>
#include <deque>
#include <list>
#include <iostream>
#include <fstream>
#include <errno.h>
#include <typeinfo>
#include "fastlib/base/base.h"
#include "fastlib/base/mpl.h"
#include "point.h"
#include "typename.h"
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/trim.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/static_assert.hpp"
#include "boost/mpl/assert.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/front.hpp"
#include "boost/mpl/insert.hpp"
#include "boost/mpl/push_back.hpp"
#include "boost/mpl/and.hpp"
#include "boost/mpl/or.hpp"
#include "boost/mpl/if.hpp"
#include "boost/mpl/is_sequence.hpp"
#include "boost/mpl/not.hpp"
#include "boost/mpl/equal_to.hpp"
#include "boost/mpl/size.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/mpl/range_c.hpp"
#include "boost/mpl/empty.hpp"
#include "boost/mpl/reverse_fold.hpp"
#include "boost/mpl/bool.hpp"
#include "boost/serialization/string.hpp"
#include "boost/serialization/vector.hpp"
#include "boost/serialization/split_member.hpp"
#include "boost/serialization/base_object.hpp"
#include "boost/archive/archive_exception.hpp"

namespace fl {
namespace data {
/**
 * @brief Keys for defining attributes for the multidataset class
 * @code
 *  struct DatasetArgs {
 *   ....
 *  };
 * @endcode
 * @param DenseTypes An mpl::set with the dense types of the multidataset
 * @param SparseType An mpl::set with the sparse types of the multidataset
 * @param CalcPrecision The precision that will be used for calculations
 * @param MetaDataType If the point is using an index attribute, then you should define its
 *        type here. If not just don't define it
 * @param StorageType the type of storage you should use for storing points
 *       @code
 *        class StorageType {
 *          class Compact {};
 *          class Extendable {};
 *          class Deletable {};
 *        };
 *       @endcode
 *       @param Compact if your dataset will use a fixed number of points use this one
 *       @param Extendable if you are planning to dynamically add points use this
 *       @param Deletable if you are planning to add/delet points use this
 *
 */
struct DatasetArgs {
  typedef boost::mpl::vector0<> DenseTypes;
  typedef boost::mpl::vector0<> SparseTypes;
  typedef double CalcPrecisionType;
  typedef fl::MakeIntIndexedStruct<boost::mpl::vector0<> >::Generated MetaDataType;
  class Compact {};
  class Extendable {};
  class Deletable {};
  typedef Extendable StorageType;
};

class empty {};

/**
 * @brief class Multidataset. It can read a great variety of csv or ssv files, for different
 *        precisions. examples are
 *        a)Classical csv files that represet dense matrices
 *        b)Sparse data in the format index:value, ie 191:33.44
 *        c)Dense Dataset  that has more than one precisions, for example int and doubles
 *        d)Sparse Dataset with more  than one precisions. for example
 *         float:1002:3.14, long_dense:344:2.97 for the mapping of types to their names
 *         refer to typename.h, as a rule if the typename is more than one word replace space with
 *         underscore
 *
 *  @brief If you just using one dense type a csv file is enough, if you want to use more than
 *         one dense types or sparse you need a header defined as following:
 *         header,dense:type:size,...,sparse:type:size
 *         for example: header,dense:int:3,dense:float:45,sparse:bool:1000,sparse:double:24440
 *
 *  @brief if you want to use labels just declare a line after header (you can omit header if you want)
 *         starting with the keyword labels, for example:
 *         labels,x1,x2,x3
 *
 *  @param ParameterList is a boost::mpl::map that passes all the necessary parameters keyd by
 *        the types defined in the DatasetArgs. Some of them have a default type, for examples
 *        look at the test files.
 */
template<typename ParameterList>
class MultiDataset {
  public:
    friend class boost::serialization::access;

MultiDataset() {
  num_of_points_=0;
}

#include "multi_dataset_mpl_defs.h"
#include "multi_dataset_iterators.h"
#include "multi_dataset_storage_selection.h"
    // These are the boxes that are going to keep all the containers of points
    // for every type. Along with declare their Iterators too
    typedef typename
    PointCollection <
    DenseTypeList_t,
    MonolithicPoint,
    DenseStorageSelection
    >::Generated DenseBox;

    typedef typename
    Iterators <
    DenseTypeList_t,
    MonolithicPoint,
    DenseStorageSelection
    >::Generated DenseIterators;

    typedef typename
    PointCollection <
    SparseTypeList_t,
    SparsePoint,
    SparseStorageSelection
    >::Generated SparseBox;

    typedef typename
    Iterators <
    SparseTypeList_t,
    SparsePoint,
    SparseStorageSelection
    >::Generated SparseIterators;

    typedef MetaDataStorageSelection<MetaDataType_t> MetaDataBox;
    typedef typename MetaDataBox::iterator MetaDataIterator;
    /**
     * @brief this is a way of exporting all the collections of points, mainly the two boxes
     * sparse and dense
     *
     */
    struct ExportedPointCollection {
      ExportedPointCollection() ;
      ExportedPointCollection(const DenseBox *dense1,
                              const SparseBox *sparse1,
                              const MetaDataBox *meta1); 
      DenseBox *dense;
      SparseBox * sparse;
      MetaDataBox *meta_data;
    };

    typedef ExportedPointCollection ExportedPointCollection_t;

     CalcPrecision_t get(index_t i, index_t j) const ;
#include "multi_dataset_alias_operators.h"
     void get(index_t point_id, Point_t *entry);
#include "multi_dataset_reset_iterators.h"
    /**
     * @bried this one resets the iterators to the begining of the dataset
     */
    void Reset();
#include "multi_dataset_choose_iterator.h"
    /**
     * @brief checks if the iterator has reached the end
     *
     */
     bool HasNext();
#include "multi_dataset_make_point_from_iterators.h"
    /**
     * @brief Gives the next point by following and advancing the iterators
     */
     void Next(Point_t *point);
    /**
     * @brief Initializes a table from a file, currently it supports "r" mode.
     *        It reads the data. Depending on the configuration you can still add
     *        points
     */
    void Init(std::string filename, std::string mode);
    /**
     * @brief Initializes the data structures in the table from the header provided.
     *        No points are added. Expected to be used in conjunction with push_back()
     *        function for extendable tables.
     */
    void Init(std::string header);
    /**
     * @brief Initializes a table from a file, currently it supports "r" mode.
     *        It reads the data. Depending on the configuration you can still add
     *        points. In this version you can choose to ignore columns
     */
    void Init(std::string filename,
              std::vector<index_t> &ignored_dense_columns,
              std::string mode);

    void Init(std::string filename,
              std::vector<index_t> &ignored_meta_columns,
              std::vector<index_t> &ignored_dense_columns,
              std::string mode);

#include "multi_dataset_init_from.h"
    /**
      * @brief Initialize the MultiDataset from a dense Matrix or a container of MonolithicPoitns
      *        We utilize this functionality in tests where we need to generate random matrices
      *        It is also good for the old style people that generate their data they don't find them
      *        in files. We should do the same for sparse data too
      */
    template<typename T>
    void Init(
      const typename DenseStorageSelection<MonolithicPoint<T> >::type::type
      &cont);
    /**
     * @brief this functionality is for writing data to file, basicaly it initializes the dimensions
     *             of every type dense/sparse and sets the number of points. If you know how many
     *             points you will store you can set it, otherwise if you don't know how many you are
     *             going to use, keep it zero. After calling that function you can start adding points
     *
     */
    template<typename ContainerType>
    void Init(ContainerType dense_dimensions,
              ContainerType sparse_dimensions,
              index_t num_of_points);
    /**
     *  @brief This function will Init a table with unifolm random data
     *         it will generate random values between low and hi.
     *         Also if the dataset is sparse, it will fill the sparse
     *         components  with the defined sparsity. sparsity can be from 
     *         0 to 1. 
     *         0 corresponds to full dense
     *         1 corresponds to all zeros
     */
    template<typename ContainerType>
    void InitRandom(ContainerType dense_dimensions,
              ContainerType sparse_dimensions,
              index_t num_of_points,
              const CalcPrecision_t low,
              const CalcPrecision_t hi,
              const CalcPrecision_t sparsity);
#include "multi_dataset_push_back_operator.h"
    /**
     * @brief This function does exactly what STL containers provide. It adds a point in the
     *        dataset. Keep in mind that if you have chosen a compact configuration, you will get
     *        a compilation error, at some point we should add an mpl assertion
     *
     */
     void push_back(Point_t &point);
    /**
     * @brief This function does exactly what STL containers provide.
     *        It adds a point in string format in the
     *        dataset. Keep in mind that if you have chosen a compact configuration, you will get
     *        a compilation error, at some point we should add an mpl assertion
     *
     */
     void push_back(std::string &point);


    /**
     * @brief returns the number of points in the dataset
     */

     index_t n_points() const;
   /**
     * @brief returns the total number of attributed, also known as features
     *        or just the sum of all the dimension for every type
     *
     */
     index_t n_attributes() const;

     ExportedPointCollection_t point_collection() const ;
#include "multi_dataset_save_operators.h"
    /**
      * @brief if you want to save multidataset that has been manipulated use this
      *        functionality
      */
    void Save(const std::string file,
              const bool header,
              const std::vector<std::string> labels,
              const std::string delimiter);
    /**
      * @brief if you want to save multidataset that has been manipulated use this
      *        functionality, this one saves to an ostream
      */
    void Save(std::ostream &out,
              const bool header,
              const std::vector<std::string> labels,
              const std::string delimiter);

     std::vector<std::string> &labels();

     const std::vector<std::string>  &labels() const ;

     const std::vector<index_t> &dense_sizes() const ;

     const std::vector<index_t> &sparse_sizes() const ;

    const std::string GetHeader() const ;

    void SetHeader(std::string header_in) ;

    bool TryToInit(const std::string &file);

    template<typename Archive>
     void save(Archive &ar, const unsigned int version) const ;

    template<typename Archive>
     void load(Archive &ar, const unsigned int version);

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** Private Members  **/
  private:
    DenseBox dense_;
    DenseIterators dense_its_;
    SparseBox sparse_;
    SparseIterators sparse_its_;
    MetaDataBox meta_;
    MetaDataIterator meta_it_;
    bool load_meta_;
    index_t num_of_points_;
    std::vector<std::string> labels_;
    std::vector<index_t> dense_sizes_;
    std::vector<index_t> sparse_sizes_;
    std::vector<index_t> ignored_dense_columns_;
    index_t num_of_metadata_;
    index_t n_attributes_;
    // this flag is used when we load sparse data
    // sparse data can be in this format type:index:value
    // but in the case where we use only one type
    // we can just assume index:value
    bool is_2_tokens_;
    bool comma_flag_;
    std::string delimiter_;
    std::string header_;


#include "multi_dataset_initializer.h"
#include "multi_dataset_loader.h"
    bool FindDelimeter(std::string &line);
#include "multi_dataset_write_header.h"
    void WriteHeader(std::ostream &out);
    void WriteLabels(std::ostream &out,
                     const std::vector<std::string> &labels);
#include "multi_dataset_check_types.h"
    void ParseHeader(std::string& line);
    void ParseLabels(std::string &line);
     void AddPoint(std::string &line,
                  std::vector<index_t> &ignored_dense_columns);
     void AddPointFast(std::string &line);

};

}
} // namespaces
#include "multi_dataset_defs.h"
#endif

