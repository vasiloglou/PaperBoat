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

#ifndef PAPERBOAT_KARNAGIO_INCLUDE_FASTLIB_TEXT_TEXT_DEFS_H_
#define PAPERBOAT_KARNAGIO_INCLUDE_FASTLIB_TEXT_TEXT_DEFS_H_
#include "boost/program_options.hpp"
#include "text.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/workspace/task.h"
#include "fastlib/util/string_utils.h"

namespace fl{ namespace txt {
   
  template<typename WorkSpaceType>
  int TextPreprocessor::Main(WorkSpaceType *ws, 
      const std::vector<std::string> &args) {
    
    FL_SCOPED_LOG(TextPreprocessor);
    std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args, "");
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
    "help", "Print this information."
    )(
     "documents_in",
     boost::program_options::value<std::string>(),
     "Every line of the file contains the document."
     " Every line must be just words separated by spaces. All others must"
     " be removed"
    )(
      "document_names_in",
      boost::program_options::value<std::string>(),
      "Every line of the file contains the document name"
    )(
      "document_categories_in", 
      boost::program_options::value<std::string>(),
      "Every line of the file contains the corresponding document category"
    )(
      "word_labels_in",
      boost::program_options::value<std::string>(),
      "Each word can be assigned a numerical label, for example in sentiment "
      "analysis each word can be assigned a polarity. "  
      "The format is <word> <numerical_score> or <prefix*> numerical_score. "
    )(
      "word_index_label_out",
      boost::program_options::value<std::string>(),
      "After applying the word_labels_in to the survived words we export a table "
      "that contains for every word index the numerical label that we know. "
      "It is exported as a sparse double table with one point. The dimensionality "
      "of the point is equal to the number of survived words and the nonzero elements "
      "have the numerical label assigned"
    )(
      "word_freq_out",
      boost::program_options::value<std::string>(),
      "use this option to output the word frequencies (before filtering through "
      "the lower and higher bounds) to a file"
    )(
      "survived_words_out",
      boost::program_options::value<std::string>(),
      "use this option to output the word index and frequency after filtering"
    )(
      "bag_of_words_out",
      boost::program_options::value<std::string>(),
      "use this option to export the sparse table that represents the bag of words"
    )(
      "bag_of_documents_out",
      boost::program_options::value<std::string>(),
      "This is the sort of the transpose of bag_of_words_out. "
      "It follows the same process with bag of words during construction "
      "with the only difference that words and documents swap places"
    )(
      "words2words_out",
      boost::program_options::value<std::string>(),
      "It short it does the bagofwords x bagofwords_transp multiplication. "
      "There is a slight difference the row normalization of bagofwords "
      "does not happen"
    )(
      "lo_frequency_bound",
      boost::program_options::value<double>()->default_value(0.0),
      "this is the minimum percentage of total documents"
      " that a word must appear, so that it participates in the bag of words"
    )(
      "hi_frequency_bound",
      boost::program_options::value<double>()->default_value(0.7),
      "this is the maximum percentage of total documents "
      " that a word must appear, so that it participates in the bag of words"
    )(
      "minimum_words_per_document",
      boost::program_options::value<index_t>()->default_value(3),
      "the bag of words conversion will reject any document that has less words "
      "than this threshold"  
    )(
      "reserved_words",
      boost::program_options::value<std::string>()->default_value(""),
      "Comma separated list of words that can not be eliminated in the tdif process. "
      "Their tdif score will be the average of all other tdifs"
    );
    boost::program_options::variables_map vm;
    boost::program_options::command_line_parser clp(args1);
    clp.style(boost::program_options::command_line_style::default_style
       ^boost::program_options::command_line_style::allow_guessing );
  
    try {
      boost::program_options::store(clp.options(desc).run(), vm);  
    }
    catch(const boost::program_options::invalid_option_value &e) {
	    fl::logger->Die() << e.what() << "in option" << e.get_option_name();
    }
    catch(const boost::program_options::invalid_command_line_syntax &e) {
	    fl::logger->Die() << e.what() ; 
    }
    catch ( const boost::program_options::multiple_occurrences &e) {
      fl::logger->Die() <<e.what() <<" from option: " << e.get_option_name();
    }
    catch (const boost::program_options::unknown_option &e) {
      fl::logger->Die() << "Unknown option: " << e.what();
    }
    catch ( const boost::program_options::error &e) {
      fl::logger->Die() << e.what();
    } 

    boost::program_options::notify(vm);
    if (vm.count("help")) {
      std::cout << fl::DISCLAIMER << "\n";
      std::cout << desc << "\n";
      return 1;
    }

    std::map<std::string, index_t> word2ind;
    std::map<std::string, index_t> wordfreq;
    std::vector<std::map<std::string, index_t> > documents;
    boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t> 
      reference_table;
    if (vm.count("documents_in")==0) {
      fl::logger->Die()<<"You must set --documents_in option"<<std::endl;
    }
    std::string document_file=vm["documents_in"].as<std::string>();
    std::vector<std::string> original_names;
    std::string document_categories_file;
    std::vector<std::string> original_categories;
    if (vm.count("document_names_in")) {
      std::string document_names_file=vm["document_names_in"].as<std::string>();
      fl::logger->Message()<<"Loading document names from "
        <<document_names_file<<std::endl;
      LoadDocumentLabels(document_names_file, &original_names);
    }
    fl::logger->Message()<<"Document name file loaded"<<std::endl;
    if (vm.count("document_categories_in")) {
      document_categories_file=vm["document_categories_in"].as<std::string>();
      fl::logger->Message()<<"Loading document categories file "
        <<document_categories_file<<std::endl;
      LoadDocumentLabels(document_categories_file, &original_categories);
      fl::logger->Message()<<"Document categories file loaded"<<std::endl;
    } 
    double lo_freq_bound=vm["lo_frequency_bound"].as<double>();
    double hi_freq_bound=vm["hi_frequency_bound"].as<double>();
    if (lo_freq_bound>=hi_freq_bound) {
      fl::logger->Die()<<"--lo_frequency_bound="<<lo_freq_bound
        <<" must be less than --hi_frequency_bound="<<hi_freq_bound;
    }
    std::set<std::string> reserved_words;
    std::string list_of_words=vm["reserved_words"].as<std::string>();
    boost::algorithm::split(reserved_words, list_of_words, 
      boost::algorithm::is_any_of(","));
    fl::logger->Message()<<"Loading document file from "
      <<document_file<<std::endl;
    LoadDocuments(document_file, 
        reserved_words, 
        lo_freq_bound,
        hi_freq_bound,
        &word2ind, &wordfreq, &documents);
    fl::logger->Message()<<"Document file loaded"<<std::endl;
    if (vm.count("word_freq_out")) {
      fl::logger->Message()<<"Exporting the word frequencies (before filtering) to "<<
          vm["word_freq_out"].as<std::string>()<<std::endl;
      ExportWordFrequencies(vm["word_freq_out"].as<std::string>(), wordfreq);
      fl::logger->Message()<<"Word frequencies exported"<<std::endl;
    }
    if (vm.count("survived_words_out")) {
      fl::logger->Message()<<"Exporting the word frequencies "
        " (after filtering) to "<<vm["survived_words_out"].as<std::string>()
        <<std::endl;
      ExportWordFrequencies(vm["survived_words_out"].as<std::string>(), 
          wordfreq, word2ind);
    }
    if (vm.count("word_labels_in")) {
      if (vm.count("word_index_label_out")==0) {
        fl::logger->Warning()<<"You have set --word_labels_in "
          "but you haven't set --word_index_label_out, no word_index_label_out "
          "table will be exported"<<std::endl;
      }
      fl::logger->Message()<<"Loading word labels"<<std::endl;
      std::map<std::string, double> word_labels;
      LoadWordLabels(vm["word_labels_in"].as<std::string>(), 
          &word_labels);
      if (vm.count("word_index_label_out")) {
        AssignWordLabels(ws, word_labels, word2ind, 
            vm["word_index_label_out"].as<std::string>());
      }
    }

    if (vm.count("bag_of_documents_out")) {
      std::string bag_of_documents_name=vm["bag_of_documents_out"].
        as<std::string>();
      boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t> bag_of_documents_table;
      MakeWord2DocumentMatrix(
        ws, 
        documents, 
        word2ind, 
        wordfreq,
        bag_of_documents_name,
        &bag_of_documents_table);
    }
    if (vm.count("words2words_out")) {
      std::string words2words_name=vm["words2words_out"].
        as<std::string>();
      boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t> bag_of_documents_table;
      MakeWord2WordMatrix(
          ws, 
          documents, 
          word2ind, 
          wordfreq,
          words2words_name,
          &bag_of_documents_table);
    }
    std::vector<std::string> refined_labels;
    std::vector<std::string> refined_categories;
    index_t minimum_words_per_description=vm["minimum_words_per_document"].as<index_t>();
    fl::logger->Message()<<"Converting documents to bag of words"
      <<std::endl;
    ConvertToBagOfWords(ws, 
        documents,
        original_names,
        original_categories,
        minimum_words_per_description,
        word2ind, 
        wordfreq, 
        index_t(word2ind.size()),
        vm["bag_of_words_out"].as<std::string>(),
        &reference_table, 
        original_names.size()==0?NULL:&refined_labels,
        original_categories.size()==0?NULL:&refined_categories);
    fl::logger->Message()<<"Bag of words conversion completed"
      <<std::endl;

    if (vm.count("bag_of_words_out")) {
      fl::logger->Message()<<"Exporting bag of words table to "
        <<vm["bag_of_words_out"].as<std::string>()<<std::endl;
      ws->ExportToFile(vm["bag_of_words_out"].as<std::string>(), 
          vm["bag_of_words_out"].as<std::string>());
      fl::logger->Message()<<"Bag of words exported"<<std::endl;
    }
    return EXIT_SUCCESS;
  }
  
  void fl::txt::TextPreprocessor::LoadDocumentLabels(const std::string &file, 
      std::vector<std::string> *labels) {
    std::ifstream fin(file.c_str());
    if (fin.fail()) {
      fl::logger->Die() << "Could not open file " << file.c_str()
           << "   error: " << strerror(errno);
    }
    while (fin.good()) {
      std::string line;
      std::getline(fin, line);
      boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
      labels->push_back(line);
    }  
  }
  
  
  template<typename WorkSpaceType>
  void fl::txt::TextPreprocessor::Run(
      WorkSpaceType *ws,
      const std::vector<std::string> &args) {
    fl::ws::Task<
      WorkSpaceType,
      &Main<WorkSpaceType> 
    > task(ws, args);
    ws->schedule(task); 
  }

  void fl::txt::TextPreprocessor::ExportWordFrequencies(const std::string &filename, 
      const std::map<std::string, index_t> &wordfreq) {
    
    std::ofstream fout(filename.c_str());
    if (fout.fail()) {
      fl::logger->Die() << "Could not open file " << filename.c_str()
           << "   error: " << strerror(errno);
    }
    for(std::map<std::string, index_t>::const_iterator it=wordfreq.begin();
        it!=wordfreq.end(); ++it) {
      fout<<it->first<<" "<<it->second<<"\n";
    }
  }

  void fl::txt::TextPreprocessor::ExportWordFrequencies(const std::string &filename, 
      const std::map<std::string, index_t> &wordfreq,
      const std::map<std::string, index_t> &word2ind) {
    
    std::ofstream fout(filename.c_str());
    if (fout.fail()) {
      fl::logger->Die() << "Could not open file " << filename.c_str()
           << "   error: " << strerror(errno);
    }
    for(std::map<std::string, index_t>::const_iterator it=word2ind.begin();
        it!=word2ind.end(); ++it) {
      fout<<it->first<<" "<<it->second<<" "
          <<wordfreq.find(it->first)->second <<"\n";
    }
  }

  template<typename HashTableType>
  void TextPreprocessor::LoadDocuments(const std::string &document_file, 
      const std::set<std::string> &reserved_words,
      double lo_freq_bound,
      double hi_freq_bound,
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
    int32 lo_freq = lo_freq_bound * documents->size();
    int32 hi_freq = hi_freq_bound * documents->size();
    fl::logger->Message()<<"Only words with frequency between "
      <<lo_freq<<" and "<< hi_freq<<" will be considered"<<std::endl; 

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
      if ((it->second>lo_freq && it->second<hi_freq) 
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

  template<typename WorkSpaceType, typename HashTableType>
  void fl::txt::TextPreprocessor::ConvertToBagOfWords(WorkSpaceType *ws, 
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
      typename HashTableType::const_iterator it1=word2ind.find(word);
      if (it1!=word2ind.end()) {
        const index_t word_id=it1->second;
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
        if (refined_labels!=NULL) {
          refined_labels->push_back(labels[i]);
        }
        if (categories.empty()==false) {
          refined_categories->push_back(categories[i]);
        }
      }
    }
    fl::logger->Message()<<"After removing the documents with words <"<<
      minimum_words_per_description<<" only "<<valid_document_counter<<
      " survived, out of "<<documents.size()<<std::endl;
    ws->Purge(reference_name);
    ws->Detach(reference_name);
   
  }

  template<typename WorkSpaceType, typename HashTableType>
  void fl::txt::TextPreprocessor::MakeWord2WordMatrix(
      WorkSpaceType *ws, 
      const std::vector<HashTableType> &documents, 
      const HashTableType &word2ind, 
      const HashTableType &wordfreq,
      const std::string &reference_name,
      boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t>  *reference_table) {
 
    index_t total_documents=documents.size();
    index_t num_words=word2ind.size();
    ws->Attach(reference_name,
        std::vector<index_t>(), 
        std::vector<index_t>(1, num_words),
        num_words,
        reference_table);
    // compute the tdifs
    std::vector<double> tdif(wordfreq.size());
    for(typename HashTableType::const_iterator it=wordfreq.begin();
        it!=wordfreq.end(); ++it) {
      std::string word=it->first;
      index_t frequency= it->second;
      typename HashTableType::const_iterator it1=word2ind.find(word);
      if (it1!=word2ind.end()) {
        const index_t word_id=it1->second;
        tdif[word_id]=log(double(total_documents)/frequency);
      }
    }
 
    std::vector<std::map<index_t, double> >w2w; 
    for(unsigned int i=0; i<documents.size(); ++i) {
      for(typename HashTableType::const_iterator it1=documents[i].begin(); 
          it1!=documents[i].end(); ++it1) {
        index_t wid1=word2ind.find(it1->first)->second;
        for(typename HashTableType::const_iterator it2=documents[i].begin(); 
            it2!=documents[i].end(); ++it2) {
          index_t wid2=word2ind.find(it2->first)->second;
          w2w[wid1][wid2]+=it1->second 
            * it2->second 
            * tdif[wid1]
            * tdif[wid2];
        } 
      } 
    }
    typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t point;
    for(unsigned int i=0; i<w2w.size(); ++i) {
      (*reference_table)->get(i, &point);  
      point.template sparse_point<double>().Load(w2w[i].begin(), 
          w2w[i].end());
    }
    ws->Purge(reference_name);
    ws->Detach(reference_name); 
  }
  
  template<typename WorkSpaceType, typename HashTableType>
  void fl::txt::TextPreprocessor::MakeWord2DocumentMatrix(
      WorkSpaceType *ws, 
      const std::vector<HashTableType> &documents, 
      const HashTableType &word2ind, 
      const HashTableType &wordfreq,
      const std::string &reference_name,
      boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t>  *reference_table) {
 
    index_t total_documents=documents.size();
    index_t num_words=word2ind.size();
    ws->Attach(reference_name,
        std::vector<index_t>(), 
        std::vector<index_t>(1, total_documents),
        num_words,
        reference_table);
    // compute the tdifs
    std::vector<double> tdif(documents.size());
    for(size_t i=0; i<documents.size(); ++i) {
      index_t frequency = documents[i].size();
      tdif[i]=log(double(num_words)/frequency);
    }
 
    std::vector<std::map<index_t, double> >w2d; 
    for(unsigned int i=0; i<documents.size(); ++i) {
      for(typename HashTableType::const_iterator it1=documents[i].begin(); 
          it1!=documents[i].end(); ++it1) {
        index_t wid1=word2ind.find(it1->first)->second;
        w2d[wid1][i]+=tdif[i];
      } 
    }
    typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t point;
    for(unsigned int i=0; i<w2d.size(); ++i) {
      double row_sum=0.0;
      for(std::map<index_t, double>::iterator it=w2d[i].begin();
          it!=w2d[i].end(); ++it) {
        row_sum+=it->second;
      }
      for(std::map<index_t, double>::iterator it=w2d[i].begin();
          it!=w2d[i].end(); ++it) {
        it->second/=row_sum;
      }
      (*reference_table)->get(i, &point);  
      point.template sparse_point<double>().Load(w2d[i].begin(),
          w2d[i].end());
    }
    ws->Purge(reference_name);
    ws->Detach(reference_name); 
  }

  void fl::txt::TextPreprocessor::LoadWordLabels(
      const std::string &word_labels_in, 
      std::map<std::string, double> *word_labels) {
    
    std::ifstream fin(word_labels_in.c_str());
    if (fin.fail()) {
      fl::logger->Die() << "Could not open file " << word_labels_in.c_str()
           << "   error: " << strerror(errno);
    }
    while (fin.good()) {
      std::string line;
      std::getline(fin, line);
      boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
      std::vector<std::string> tokens=fl::SplitString(line, " ");
      try {
        (*word_labels)[tokens[0]]=boost::lexical_cast<double>(tokens[1]);
      }
      catch(const boost::bad_lexical_cast &e) {
      
      }
    }   
  }

  template<typename WorkSpaceType>
  void fl::txt::TextPreprocessor::AssignWordLabels(WorkSpaceType *ws, 
        const std::map<std::string, double> &word_labels, 
        const std::map<std::string, index_t> &word2ind, 
        const std::string &word_index_label_out) {
    
    boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t> table;
    ws->Attach(word_index_label_out, 
        std::vector<index_t>(),
        std::vector<index_t>(1, word2ind.size()),
        1,
        &table);
    typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t point;
    table->get(0, &point);
    std::vector<std::pair<index_t, double> > loader;
    typedef typename std::map<std::string, double>::const_iterator Iterator1_t;
    typedef typename std::map<std::string, index_t>::const_iterator Iterator2_t;
    for(Iterator2_t it=word2ind.begin(); it!=word2ind.end(); ++it) {
      Iterator1_t it1=word_labels.lower_bound(it->first);
      if (it1==word_labels.end()) {
        break;
      }
      if (it1->first==it->first) {
        loader.push_back(std::make_pair(it->second, it1->second));    
      } else {
        --it1;
        if (fl::StringEndsWith(it1->first, "*")) {
          if (fl::StringStartsWith(it->first, 
                it1->first.substr(0, it->first.size()-1)) ) {
            loader.push_back(std::make_pair(it->second, it1->second));    
          }
        }
      }
    }
    point.template sparse_point<double>().Load(
        loader.begin(), loader.end());
    for(typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t::iterator 
        it=point.begin(); it!=point.end(); ++it) {
    }
    ws->Purge(word_index_label_out);
    ws->Detach(word_index_label_out);
  }

}}

#endif
