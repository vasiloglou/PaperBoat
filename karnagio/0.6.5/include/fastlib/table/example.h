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

/**
 * @file example.h
 * @brief This file gives examples of how to load a table
 *        and use it.
 *
 * Table is a highly templatized structure that can handle different
 * precisions and modes (sparse or dense). We have instantiated a couple
 * of combinations for ease of use.
 *
 * @code
 *  #include "fastlib/table/default/categorical/labeled/balltree/table.h"
 *  #include "fastlib/table/default/dense/labeled/balltree/table.h"
 *  #include "fastlib/table/default/dense/labeled/kdtree/table.h"
 *  #include "fastlib/table/default/dense_categorical/labeled/balltree/table.h"
 *  #include "fastlib/table/default/dense_sparse/labeled/balltree/table.h"
 *  #include "fastlib/table/default/sparse/labeled/balltree/table.h"
 *  // for dense data with double precision and a kdtree as an index use
 *  fl::table::dense::labeled::kdtree::Table table;
 *  // for sparse data with double precision and a balltree as an index use
 *  fl::table::sparse::labeled::balltree::Table table;
 *  // for categorical data with bool precision and a balltree as an index use
 *  fl::table::categorical::labeled::balltree::Table table;
 *  // for mixed data with dense double precision and categorical and a balltree as an index use
 *  fl::table::dense_categorical::labeled::balltree::Table table;
 * @endcode
 *
 * As a note the categorical data are represented as sparse booleans.
 * 
 * Now we will show an example of how to load the table from a file. For the file format
 * @sa http://www.ismion.com/documentation/file_formats.html
 *
 * @code
 *  fl::table::dense::labeled::kdtree::Table table;
 *  table.Init("myfile.txt", "r");
 *  // now read data from the table:
 *  fl::table::dense::labeled::kdtree::Table::Point_t point;
 *  for(index_t i=0; i<table.n_entries(); ++i) {
 *    // get an alias on the point of the table
 *    table.get(i, &point);
 *    // Any change we do on the point is reflected back on the table
 *    // point is not a local copy
 *    for(int j=0; j<point.size(); ++j) {
 *      double a=point.get(j);
 *      point.set(j, a+1);
 *    }
 *  }
 * @endcode
 *
 * If you want to load the table with data you can use the following code
 *
 * @code
 *  fl::table::dense::labeled::kdtree::Table table;
 *  index_t dimension=10;
 *  index_t number_of_points=1000;
 *  table.Init("myfile", // this is a filename, if you want to save the table later
 *             std::vector<index_t>(1, dimension), // this vector is a list of
 *                                                 // the dense dimensions.
 *                                                 // The table can have several precisions
 *                                                 // In this particular case this table
 *                                                 // has been templatized to have only
 *                                                 // one precision (double), so the vector
 *                                                 // can have only one elements
 *             std::vector<index_t>(), // this vector has the sparse dimensions,
 *                                     // in general the table can several precisions
 *                                     // in this particular case the table is designed
 *                                     // to have only one dense precision and not sparse
 *                                     // ones, that is why the vector is empty
 *             number_of_points);
 *           
 *  // Now we can load the table
 *  fl::table::dense::labeled::kdtree::Table::Point_t point;
 *  for(index_t i=0; i<table.n_entries(); ++i) {
 *    table.get(i, &point);
 *    // set every element of the point to something like i+j  
 *    for(index_t j=0; j<point.size(); ++j) {
 *      point.set(j, i+j);
 *    }
 *  }
 *  // At this point the table is loaded  
 * @endcode
 *
 * As a note if we had sparse data this is how we would initialize the table
 * @code
 *  fl::table::sparse::labeled::balltree::Table table;
 *  index_t dimension=10000;
 *  index_t number_of_points=10000;
 *  table.Init("myfile", // this is a filename, if you want to save the table later
 *             std::vector<index_t>(), 
 *             std::vector<index_t>(1, dimension), 
 *             number_of_points);
 *
 * @endcode
 *
 * The table is labeled which means that except for the data in every dimension it also
 * carries metadata for every point. Metadata does not participate in linear algebra
 * operations. Every point carries three numbers (a signed char, a double and an int)
 * that hold extra information. For example the signed char can be used to store
 * class information, double for an associated value and int as an identifier. To access
 * them use the following syntax:
 *
 * @code
 *  fl::table::dense::labeled::kdtree::Table table;
 *  fl::table::dense::labeled::kdtree::Table::Point_t point;
 *  table.get(0, &point);
 *  signed char a=1;
 *  point.meta_data().template get<0>()=a;
 *  a=point.meta_data().template get<0>();
 *  double b=0.34;
 *  point.meta_data().template get<1>()=b;
 *  b=point.meta_data().template get<1>();
 *  int c=1314;
 *  point.meta_data().template get<2>()=c;
 *  c=point.meta_data().template get<2>();
 * @endcode
 */ 
