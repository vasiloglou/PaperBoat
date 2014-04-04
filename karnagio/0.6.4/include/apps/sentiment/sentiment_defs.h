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

#ifndef PAPER_BOAT_KARNAGIO_INCLUDE_APP_SENTIMENT_SENTIMENT_H_DEFS_
#define PAPER_BOAT_KARNAGIO_INCLUDE_APP_SENTIMENT_SENTIMENT_H_DEFS_
#include "boost/program_options.hpp"
#include "sentiment.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/workspace/task.h"
#include "fastlib/util/string_utils.h"
#include "fastlib/text/text.h"
#include "mlpack/graph_diffuser/graph_diffuser.h"

namespace fl{ namespace app {
  
  template<typename WorkSpaceType>
  void Sentiment<WorkSpaceType>::PrettyWordSentimentExport(
            WorkSpaceType *ws,
            const std::string &word_sentiment_table_name,
            const std::string &survived_words_filename,
            const std::string &outfilename) {
    
    boost::shared_ptr<
      typename WorkSpaceType::DefaultTable_t> table;
    ws->Attach(word_sentiment_table_name, &table);   
    typedef typename WorkSpaceType::DefaultTable_t::Point_t Point_t;
    typename WorkSpaceType::DefaultTable_t::Point_t point;
    table->get(0, &point);
    std::ifstream fin(survived_words_filename.c_str());
    if (fin.fail()) {
      fl::logger->Die() << "Could not open file " 
        << survived_words_filename.c_str()
        << "   error: " << strerror(errno);
    }
    
    std::map<index_t, std::string> word2ind;
    while (fin.good()) {
      std::string line;
      std::getline(fin, line);
      boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
      std::vector<std::string> tokens=fl::SplitString(line, " ");
      if (tokens.size()!=3) {
        continue;
      }
      word2ind[boost::lexical_cast<index_t>(tokens[1])]=tokens[0];
    } 

    std::ofstream fout(outfilename.c_str());
    if (fout.fail()) {
      fl::logger->Die() << "Could not open file " << outfilename.c_str()
         << "   error: " << strerror(errno);
    }

    for(typename Point_t::iterator it=point.begin(); 
        it!=point.end(); ++it) {
      if (it.value()==0) {
        continue;
      }
      fout<<word2ind[it.attribute()]
          <<" "
          <<it.value()
          <<std::endl;
    }
  }

  template<typename WorkSpaceType>
  int Sentiment<WorkSpaceType>::Main(WorkSpaceType *ws, 
      const std::vector<std::string> &args) {
    
    FL_SCOPED_LOG(Sentiment);
    std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args, "");
    boost::program_options::options_description desc(
        "This is a sentiment analysis tool. It uses the preptext component "
        "to build a document-term matrix and then build a graph and do "
        "graph diffusion to infer the sentiment using the graphd component. "
        "Below you see some options for running the tool. To see the options for "
        "for the preptext component use --preptext:help and for the graphd "
        "component use --graphd:help. To pass options to both of them use "
        "the --preptext: or --graphd: prefixes.\n\n"  
        "Available options");
    desc.add_options()(
    "help", "Print this information."
    )(
     "documents_in",
     boost::program_options::value<std::string>(),
     "Every line of the file contains the document."
     " Every line must be just words separated by spaces. All others must"
     " be removed"
    )(
     "word_sentiments_in",
     boost::program_options::value<std::string>(),
     "this is a file that contains the the sentiment assigned in predefined words "
     "the format is <word> <sentiment_score> or <prefix*> sentiment_score. "
    )(
      "word_sentiments_out",
      boost::program_options::value<std::string>(),
      "in this file we export the word associated with a sentiment"
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
    if (vm.count("documents_in")==0) {
      fl::logger->Die()<<"You must provide the --documents_in argument";
    }
    std::vector<std::string> graphd_args=fl::ws::MakeArgsFromPrefix(args, 
        "graphd");
    std::vector<std::string> preptext_args=fl::ws::MakeArgsFromPrefix(args, 
        "preptext");
    preptext_args.push_back("--documents_in="
        +vm["documents_in"].as<std::string>());
    preptext_args.push_back("--bag_of_words_out=bagofwords");
    if (vm.count("word_sentiments_in")==0) {
      fl::logger->Die()<<"You must set the word_sentiments_in "
        "option";
    }
    preptext_args.push_back("--word_labels_in="+
        vm["word_sentiments_in"].as<std::string>());
    const std::string survived_words_filename="survived_words";
    preptext_args.push_back("--survived_words_out="
        +survived_words_filename);
    const std::string word_sentiment=ws->GiveTempVarName();
    preptext_args.push_back("--word_index_label_out="+word_sentiment);
    fl::txt::TextPreprocessor::Run(ws, preptext_args);
    graphd_args.push_back("--references_in=bagofwords");    
    graphd_args.push_back("--right_labels_in="+word_sentiment);
    fl::ml::GraphDiffuser<boost::mpl::void_>::Run(ws, graphd_args);
    ws->ExportAllTables(graphd_args);
    if (vm.count("word_sentiments_out")) {
      std::map<std::string, std::string> graphd_argmap=
        fl::ws::GetArgumentPairs(graphd_args);
      if (graphd_argmap.count("--right_labels_out")) {
        fl::logger->Message()<<"Exporting words sentiment to "
          <<vm["word_sentiments_out"].as<std::string>()
          <<std::endl;
        PrettyWordSentimentExport(
            ws,
            graphd_argmap["--right_labels_out"],
            survived_words_filename,
            vm["word_sentiments_out"].as<std::string>());
      }
    }
    return 0;
  }

  template<typename WorkSpaceType>
  void Sentiment<WorkSpaceType>::Run(
      WorkSpaceType *ws,
      const std::vector<std::string> &args) {

    bool found=false;
    std::string documents_in;
    for(size_t i=0; i<args.size(); ++i) {
      if (fl::StringStartsWith(args[i],"--documents_in=")) {
        found=true;
        std::vector<std::string> tokens=fl::SplitString(args[i], "=");
        if (tokens.size()!=2) {
          fl::logger->Die()<<"Something is wrong with the --documents_in flag";
        }
        documents_in=tokens[1];
        break;
      }
    }
    fl::ws::Task<WorkSpaceType, 
      Sentiment<WorkSpaceType>::Main> task(ws, args);
    if (found==false) {
      task();
      return ;
    } 
    ws->schedule(task);
  }


}}
#endif


