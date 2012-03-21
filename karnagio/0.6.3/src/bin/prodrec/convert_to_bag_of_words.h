/*
Copyright © 2010, Ismion Inc
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

#ifndef PAPERBOAT_INCLUDE_FASTLIB_TABLE_CONVERT_TO_BAG_OF_WORDS_H_
#define PAPERBOAT_INCLUDE_FASTLIB_TABLE_CONVERT_TO_BAG_OF_WORDS_H_

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <boost/algorithm/string.hpp>
#include "fastlib/base/base.h"
namespace fl { namespace table {
    template<typename WorkSpaceType, typename HashTableType>
  void ConvertToBagOfWords(WorkSpaceType *ws, 
      const std::vector<HashTableType> &documents, 
      const std::vector<std::string> &labels,
      const std::vector<std::string> &categories,
      index_t minimum_words_per_description,
      const HashTableType &word2ind, 
      const HashTableType &wordfreq,
      index_t dimension,
      const std::string &reference_name,
      boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t>  *reference_table,
      std::vector<std::string> *refined_labels,
      std::vector<std::string> *refined_categories) {
  
    index_t total_documents=documents.size();
    ws->Attach(reference_name,
        std::vector<index_t>(), 
        std::vector<index_t>(1, dimension),
        0,
        reference_table);
    // compute the tdifs
    std::vector<double> tdif(dimension);
    for(typename HashTableType::const_iterator it=wordfreq.begin();
        it!=wordfreq.end(); ++it) {
      std::string word=it->first;
      index_t frequency= it->second;
      typename HashTableType::const_iterator it=word2ind.find(word);
      if (it!=word2ind.end()) {
        const index_t word_id=it->second;
        tdif[word_id]=log(double(total_documents)/frequency);
      }
    }
 
    index_t valid_document_counter=0; 
    for(unsigned int i=0; i<documents.size(); ++i) {
      std::vector<std::pair<index_t, double> > bag_of_words;
      double row_sum=0;
      for(typename HashTableType::const_iterator it=documents[i].begin(); 
          it!=documents[i].end(); ++it) {
        if (word2ind.count(it->first)) {
          const index_t word_id=word2ind.find(it->first)->second;
          double weight=tdif[word_id]*it->second;
          bag_of_words.push_back(std::make_pair(word_id, weight));
          row_sum+=weight;
        }
      }
      for(index_t j=0; j<bag_of_words.size(); ++j) {
        bag_of_words[j].second/=row_sum;
      }
      std::sort(bag_of_words.begin(), bag_of_words.end());
      if (bag_of_words.size()>=minimum_words_per_description) {
        typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t point;
        std::vector<index_t> dim(1,dimension);
        point.Init(dim);
        point.template sparse_point<double>().Load(bag_of_words.begin(), bag_of_words.end());
        (*reference_table)->push_back(point);
        valid_document_counter++;
        refined_labels->push_back(labels[i]);
        if (categories.empty()==false) {
          refined_categories->push_back(categories[i]);
        }
      }
    }
    fl::logger->Message()<<"After removing the documents with words <"<<
      minimum_words_per_description<<" only "<<refined_labels->size()<<
      " survived, out of "<<labels.size()<<std::endl;
    ws->Purge(reference_name);
    ws->Detach(reference_name);
   
  }

}} // namespaces

#endif
