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

#ifndef PAPERBOAT_INCLUDE_FASTLIB_TABLE_LOAD_DOCUMETS_H_
#define PAPERBOAT_INCLUDE_FASTLIB_TABLE_LOAD_DOCUMETS_H_

#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <boost/algorithm/string.hpp>
#include "fastlib/base/base.h"
namespace fl { namespace table {
  template<typename HashTableType>
  void LoadDocuments(const std::string &document_file, 
      const std::set<std::string> &reserved_words,
      index_t lo_freq_bound,
      index_t hi_freq_bound,
      HashTableType *word2ind, 
      HashTableType *wordfreq, 
      std::vector<HashTableType > *documents) {
    
    std::ifstream fin(document_file.c_str());
    if (fin.fail()) {
      fl::logger->Die() << "Could not open file " << document_file.c_str()
         << "   error: " << strerror(errno);
    }
    // parse the file and extract from every line the bag of words along with 
    // their frequency in that line. Every line is a text entry
    while (fin.good()) {
      std::string line;
      std::getline(fin, line);
      // split the line into tokens
      boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
      std::vector<std::string> tokens;
      boost::algorithm::split(tokens, line, boost::algorithm::is_any_of(" "));
      documents->push_back(HashTableType());
      for(std::vector<std::string>::iterator it=tokens.begin(); 
          it!=tokens.end(); ++it) {
        // convert to lowercase
        boost::algorithm::to_lower(*it);
        if (documents->back().count(*it)==0) {
          documents->back()[*it]=1;
        } else {
          documents->back()[*it]+=1;
        }
      }
    }
    // now populate the global word frequency
    for(unsigned int i=0; i<documents->size(); ++i) {
      for(typename HashTableType::const_iterator 
          it=documents->operator[](i).begin();
          it!=documents->operator[](i).end(); 
          ++it) {
        if (wordfreq->count(it->first)==0) {
          wordfreq->operator[](it->first)=1;
        } else {
          wordfreq->operator[](it->first)+=1;
        }
      }
    }
    // now we have to filter the words through the frequency bounds (reserved words are also in regardless of their frequency) and 
    // populate the map of words to index
    index_t word_universal_counter=0;
    for(typename HashTableType::const_iterator it=wordfreq->begin();
        it!=wordfreq->end(); ++it) {
      if ((it->second>lo_freq_bound && it->second<hi_freq_bound) 
          || reserved_words.count(it->first)!=0) {
        word2ind->operator[](it->first)=word_universal_counter;
        word_universal_counter++;  
      } 
    }
    // now calculate the average tdif score and assign it to reserved words
    double average_score=0;
    for(typename HashTableType::const_iterator it=word2ind->begin();
        it!=word2ind->end(); ++it) {
      if (reserved_words.count(it->first)==0) {
        average_score+=(*wordfreq)[it->first];
      }
    }
    average_score/=(word2ind->size()-reserved_words.size());
    for(std::set<std::string>::const_iterator it=reserved_words.begin();
        it!=reserved_words.end(); ++it) {
      (*wordfreq)[*it]=1;
    }

    fl::logger->Message()<<"After imposing frequency bounds [lo,hi]="
      <<"["<<lo_freq_bound<<","<<hi_freq_bound<<"], only "<<
     word_universal_counter<<" words survived out of "<<wordfreq->size() <<std::endl; 
  }

}} // namespaces
#endif
