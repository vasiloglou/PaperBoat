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

#include "fastlib/table/table_defs.h"
#include "fastlib/table/default_table.h"
#include "boost/archive/text_iarchive.hpp"
#include "boost/archive/text_oarchive.hpp"
#include "fastlib/table/table_serialization.h"
#include "fastlib/data/multi_dataset_defs.h"
#include "fastlib/data/multi_dataset_dev.h"



template void fl::table::Table<fl::table::DefaultTableMap>::
    save(boost::archive::text_oarchive &ar, const unsigned int version) const;
template void fl::table::Table<fl::table::DefaultTableMap>::
    load(boost::archive::text_iarchive &ar, const unsigned int version);

template void fl::table::Table<fl::table::DefaultLabeledTableMap>::
    save(boost::archive::text_oarchive &ar, const unsigned int version) const ;
template void fl::table::Table<fl::table::DefaultLabeledTableMap>::
    load(boost::archive::text_iarchive &ar, const unsigned int version);

