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

#ifndef PAPERBOAT_KARNAGIO_INCLUDE_FASTLIB_TEXT_TEXT_H_
#define PAPERBOAT_KARNAGIO_INCLUDE_FASTLIB_TEXT_TEXT_H_
#include <map>
#include <vector>
#include <string>
#include "fastlib/base/base.h"

namespace fl{ namespace txt {

  class TextPreprocessor {
    public:
      template<typename WorkSpaceType>
      static int Main(WorkSpaceType *, const std::vector<std::string> &args);  
      template<typename WorkSpaceType>
      static void Run(WorkSpaceType *data,
         const std::vector<std::string> &args);

    private:
      static void LoadDocumentLabels(const std::string &file, 
        std::vector<std::string> *labels);     
      static void ExportWordFrequencies(const std::string &filename, 
          const std::map<std::string, index_t> &wordfreq);
      static void ExportWordFrequencies(const std::string &filename, 
          const std::map<std::string, index_t> &wordfreq,
          const std::map<std::string, index_t> &word2ind);
      template<typename HashTableType>
      static void LoadDocuments(const std::string &document_file, 
          const std::set<std::string> &reserved_words,
          double lo_freq_bound,
          double hi_freq_bound,
          HashTableType *word2ind, 
          HashTableType *wordfreq, 
          std::vector<HashTableType > *documents);
      template<typename WorkSpaceType, typename HashTableType>
      static void ConvertToBagOfWords(WorkSpaceType *ws, 
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
          std::vector<std::string> *refined_categories);

    template<typename WorkSpaceType, typename HashTableType>
    static void MakeWord2WordMatrix(
        WorkSpaceType *ws, 
        const std::vector<HashTableType> &documents, 
        const HashTableType &word2ind, 
        const HashTableType &wordfreq,
        const std::string &reference_name,
        boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t>  *reference_table);

    template<typename WorkSpaceType, typename HashTableType>
    static void MakeWord2DocumentMatrix(
        WorkSpaceType *ws, 
        const std::vector<HashTableType> &documents, 
        const HashTableType &word2ind, 
        const HashTableType &wordfreq,
        const std::string &reference_name,
        boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t>  *reference_table);
    
    static void LoadWordLabels(const std::string &word_labels_in, 
                               std::map<std::string, double> *word_labels);
    
    template<typename WorkSpaceType>
    static void AssignWordLabels(WorkSpaceType *ws, 
        const std::map<std::string, double> &word_labels, 
        const std::map<std::string, index_t> &word2ind, 
        const std::string &word_index_label_out);

   };

}}

#endif
