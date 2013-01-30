/*
Copyright © 2010, Ismion Inc.
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
#include "fastlib/table/table_defs.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/table/default/categorical/labeled/balltree/table.h"
#include "fastlib/table/default/dense/labeled/kdtree/table.h"
#include "fastlib/table/default/dense/labeled/balltree/table.h"
#include "fastlib/table/default/dense_categorical/labeled/balltree/table.h"
#include "fastlib/table/default/dense_sparse/labeled/balltree/table.h"
#include "fastlib/table/default/sparse/labeled/balltree/table.h"
#include "fastlib/table/default/sparse/labeled/balltree/uint8/table.h"
#include "fastlib/table/default/sparse/labeled/balltree/uint16/table.h"

template class ::fl::table::Table<fl::table::dense::labeled::kdtree::TableMap> ;
template class ::fl::table::Table<fl::table::dense::labeled::balltree::TableMap> ;
template class ::fl::table::Table<fl::table::sparse::labeled::balltree::TableMap> ;
template class ::fl::table::Table<fl::table::dense_sparse::labeled::balltree::TableMap> ;
template class fl::table::Table<fl::table::categorical::labeled::balltree::TableMap> ;
template class fl::table::Table<fl::table::dense_categorical::labeled::balltree::TableMap> ;
template class fl::table::Table<fl::table::sparse::labeled::balltree::uint8::TableMap> ;
template class fl::table::Table<fl::table::sparse::labeled::balltree::uint16::TableMap> ;

