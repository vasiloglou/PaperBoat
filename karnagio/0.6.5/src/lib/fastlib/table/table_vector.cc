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
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/table/table_vector.h"

template class fl::table::TableVector<unsigned char>;
template class fl::table::Table<fl::table::TableVectorMap<unsigned char> >;
template class fl::table::TableVector<signed char>;
template class fl::table::Table<fl::table::TableVectorMap<signed char> >;
template class fl::table::TableVector<unsigned short int>;
template class fl::table::Table<fl::table::TableVectorMap<unsigned short int> >;
template class fl::table::TableVector<signed short int>;
template class fl::table::Table<fl::table::TableVectorMap<signed short int> >;
template class fl::table::TableVector<signed long int>;
template class fl::table::Table<fl::table::TableVectorMap<signed long int> >;
template class fl::table::TableVector<int>;
template class fl::table::Table<fl::table::TableVectorMap<int> >;
template class fl::table::TableVector<long long int>;
template class fl::table::Table<fl::table::TableVectorMap<long long int> >;
template class fl::table::TableVector<double>;
template class fl::table::Table<fl::table::TableVectorMap<double> >;
template class fl::table::TableVector<long double>;
template class fl::table::Table<fl::table::TableVectorMap<long double> >;
template class fl::table::TableVector<float>;
template class fl::table::Table<fl::table::TableVectorMap<float> >;





