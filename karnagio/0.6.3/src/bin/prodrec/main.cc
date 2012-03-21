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
#include <map>
#include <set>
#include <vector>
#include <cstdlib>
#include "boost/algorithm/string.hpp"
#include "boost/tuple/tuple.hpp"
#include "boost/bind.hpp"
#include "fastlib/base/base.h"
#include "mlpack/nmf/nmf.h"
#include "mlpack/allkn/allkn.h"
#include "mlpack/svd/svd.h"
#include "fastlib/workspace/workspace_defs.h"
#include "fastlib/workspace/arguments.h"
#include "load_documents.h"
#include "convert_to_bag_of_words.h"

void LoadProductLabels(const std::string &filename, 
    std::vector<std::string> *labels);

template<typename WorkSpaceType>
void ExportProductNeighbors(WorkSpaceType *workspace,
    index_t reliability_threshold,
    const std::string &filename, 
    const std::vector<std::string> &labels,
    index_t nmf_runs,
    std::vector<std::string> *refined_labels); 

template<typename TableType1, typename TableType2>
void ExportProductNeighbors(
    const std::string &filename, 
    const std::vector<std::string> &labels,
    TableType1 &indices_table,
    TableType2 &distances_table) ;

void ExportWordFrequencies(const std::string &filename, 
    const std::map<std::string, index_t> &wordfreq);

template<typename TableType>
void ExportNmfClusters(TableType &w_factor, 
    const std::vector<std::string> &labels, 
    const std::string &filename);  
 
template<typename WorkSpaceType, typename TableType1, typename TableType2>
void MakeProductGraph(WorkSpaceType *ws, const std::string &graph_name,
    TableType1 &indices_table, boost::shared_ptr<TableType2> *product_graph);


template<typename WorkSpaceType>
void ComputeNeighbors(WorkSpaceType &ws, 
                      const std::vector<std::string> &args,
                      const std::string &references_in, 
                      const std::string &indices_out, 
                      const std::string &distances_out, 
                      const std::vector<std::string> &refined_categories);

int main(int argc, char* argv[]) {
  FL_SCOPED_LOG(product_similarity);
  // Convert C input to C++; skip executable name for Boost
  std::vector<std::string> args(argv + 1, argv + argc);

  try {
    std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args, "");
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
    "help", "Print this information."
    )(
     "product_descriptions_in",
     boost::program_options::value<std::string>()->default_value("descriptions"),
     "Every line of the file contains the product description."
     " Every line must be just words separated by spaces. All others must"
     " be removed"
    )(
      "product_names_in",
      boost::program_options::value<std::string>()->default_value("products"),
      "Every line of the file contains tha product name"
    )(
      "product_categories_in", 
      boost::program_options::value<std::string>()->default_value(""),
      "Every line of the file contains the corresponding product category"
    )(
      "word_freq_out",
      boost::program_options::value<std::string>()->default_value(""),
      "use this option to output the word frequencies (before filtering through "
      "the lower and higher bounds) to a file"
    )(
      "results1_out",
      boost::program_options::value<std::string>()->default_value("results1"),
      "Nearest neighbrs result after nmf"
      "every line contains pairs of neighboring products along with the distance"
    )(
      "results2_out",
      boost::program_options::value<std::string>()->default_value("results2"),
      "Nearest neighbrs result after spectral clustering"
      "every line contains pairs of neighboring products along with the distance"
    )(
      "bag_of_words_out",
      boost::program_options::value<std::string>()->default_value(""),
      "use this option to export the sparse table that represents the bag of words"
    )(
      "nmf_clusters_out", 
      boost::program_options::value<std::string>()->default_value(""),
      "use this option to export the clusters after nmf into a file"
    )(
      "lo_frequency_bound",
      boost::program_options::value<double>()->default_value(0.0),
      "this is the minimum percentage of total documents (product_descriptions)"
      " that a word must appear, so that it participates in the bag of words"
    )(
      "hi_frequency_bound",
      boost::program_options::value<double>()->default_value(0.7),
      "this is the maximum percentage of total documents (product_descriptions)"
      " that a word must appear, so that it participates in the bag of words"
    )(
      "minimum_words_per_description",
      boost::program_options::value<index_t>()->default_value(3),
      "the bag of words conversion will reject any document that has less words "
      "than this threshold"  
    )(
      "dr_algorithm",
      boost::program_options::value<std::string>()->default_value("nmf"),
      "If you set it to nmf it will do nmf for dimensionality reduction. "
      "If you set it to svd it will do svd for dimensionality reduction"
    )(
      "nmf_runs",
      boost::program_options::value<index_t>()->default_value(5),
     " how many times to run nmf in order to stabilize the results" 
    )(
      "reliability_threshold",
      boost::program_options::value<index_t>()->default_value(5),
      "After computing nmf several times, we consider only products that have consistent "
      "nearest neighbors. The reliability threshold measures how stable neighbors are"
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
	    fl::logger->Die() << "Invalid Argument: " << e.what();
    }
    catch(const boost::program_options::invalid_command_line_syntax &e) {
	   fl::logger->Die() << "Invalid command line syntax: " << e.what(); 
    }
    catch (const boost::program_options::unknown_option &e) {
       fl::logger->Die() << "Unknown option: " << e.what();
    }

    boost::program_options::notify(vm);
    if (vm.count("help")) {
      std::cout << fl::DISCLAIMER << "\n";
      std::cout << desc << "\n";
      return 1;
    }
    fl::Logger::SetLogger("debug");
    fl::ws::WorkSpace ws;
    ws.set_schedule_mode(1);
    std::map<std::string, index_t> word2ind;
    std::map<std::string, index_t> wordfreq;
    std::vector<std::map<std::string, index_t> > documents;
    boost::shared_ptr<fl::ws::WorkSpace::DefaultSparseDoubleTable_t> reference_table;
    std::string labels_file=vm["product_names_in"].as<std::string>();
    std::vector<std::string> original_labels;
    std::string categories_file=vm["product_categories_in"].as<std::string>();
    std::vector<std::string> original_categories;
    LoadProductLabels(labels_file, &original_labels);
    if (categories_file!="") {
      LoadProductLabels(categories_file, &original_categories);
    }
    std::string document_file=vm["product_descriptions_in"].as<std::string>();
    index_t lo_freq_bound=vm["lo_frequency_bound"].as<double>()*original_labels.size();
    index_t hi_freq_bound=vm["hi_frequency_bound"].as<double>()*original_labels.size();
    std::set<std::string> reserved_words;
    std::string list_of_words=vm["reserved_words"].as<std::string>();
    boost::algorithm::split(reserved_words, list_of_words, boost::algorithm::is_any_of(","));
    fl::table::LoadDocuments(document_file, 
        reserved_words, 
        lo_freq_bound,
        hi_freq_bound,
        &word2ind, &wordfreq, &documents);
    if (vm["word_freq_out"].as<std::string>()!="") {
      fl::logger->Message()<<"Exporting the word frequencies to "<<
          vm["word_freq_out"].as<std::string>()<<std::endl;
      ExportWordFrequencies(vm["word_freq_out"].as<std::string>(), wordfreq);
    }
    std::vector<std::string> refined_labels;
    std::vector<std::string> refined_categories;
    index_t minimum_words_per_description=vm["minimum_words_per_description"].as<index_t>();
    ConvertToBagOfWords(&ws, 
        documents,
        original_labels,
        original_categories,
        minimum_words_per_description,
        word2ind, 
        wordfreq, 
        index_t(word2ind.size()),
        std::string("bag_of_words"),
        &reference_table, 
        &refined_labels,
        &refined_categories);

    std::vector<std::string> refined_labels1;
    if (vm["dr_algorithm"].as<std::string>()=="nmf") {
      index_t nmf_runs=vm["nmf_runs"].as<index_t>();
      for(int i=0; i<nmf_runs; ++i) {
        std::string nmf_run(boost::lexical_cast<std::string>(i));
        fl::ws::Arguments nmf_args;
        nmf_args.Add(fl::ws::MakeArgsFromPrefix(args, "nmf"));
        nmf_args.Add("references_in", "bag_of_words");
        nmf_args.Add("w_factor_out", std::string("w_factor")+nmf_run);
        nmf_args.Add("h_factor_out", std::string("h_factor")+nmf_run);
        nmf_args.Add("run_mode", "train");
        fl::ml::Nmf::Run(&ws, nmf_args.args());
  //    if (vm["nmf_clusters_out"].as<std::string>()!="") {
  //      boost::shared_ptr<fl::ws::WorkSpace::DefaultTable_t> w_factor_table;
  //      ws.Attach(std::string("w_factor").append(nmf_run), &w_factor_table);
  //      ExportNmfClusters(*w_factor_table, refined_labels, vm["nmf_clusters_out"].as<std::string>());  
  //    }
  //
        std::string references_in=std::string("w_factor")+nmf_run;
        std::string indices_out=std::string("indices_table")+nmf_run;
        std::string distances_out=std::string("distances_table")+nmf_run;

        if (refined_categories.empty()) {
          fl::ws::Arguments allkn1_args;
          allkn1_args.Add(fl::ws::MakeArgsFromPrefix(args, "allkn1"));
          ws.IndexTable(std::string("w_factor")+nmf_run, 
              "l2",
              "",
              10);
          allkn1_args.Add("references_in", std::string("w_factor")+nmf_run);
          allkn1_args.Add("indices_out",   std::string("indices_table")+nmf_run);
          allkn1_args.Add("distances_out", std::string("distances_table")+nmf_run);
          fl::ml::AllKN<boost::mpl::void_>::Run(&ws, allkn1_args.args());
        } else { 
          ComputeNeighbors(ws, 
                           args,
                           references_in, 
                           indices_out, 
                           distances_out, 
                           refined_categories);
        }
      }
  
      std::string result1_file=vm["results1_out"].as<std::string>();
      index_t reliability_threshold=vm["reliability_threshold"].as<index_t>();
      ws.schedule(boost::bind(ExportProductNeighbors<fl::ws::WorkSpace>, 
          &ws,
          reliability_threshold,
          result1_file, 
          refined_labels,
          nmf_runs,
          &refined_labels1));
    } else {
      if (vm["dr_algorithm"].as<std::string>()=="svd") {
        fl::ws::Arguments svd_args;
        svd_args.Add(fl::ws::MakeArgsFromPrefix(args, "svd"));
        svd_args.Add("references_in", "bag_of_words");
        svd_args.Add("lsv_out", "lsv");
        svd_args.Add("rsv_out", "rsv");
        svd_args.Add("sv_out", "sv");
        fl::ml::Svd<boost::mpl::void_>::Run(&ws, svd_args.args());
        if (refined_categories.empty()==true) {
          fl::ws::Arguments allkn1_args;
          allkn1_args.Add(fl::ws::MakeArgsFromPrefix(args, "allkn1"));
          ws.IndexTable("lsv", 
            "l2",
            "",
            10);
          ws.ExportToFile("lsv", "lsv");
          ws.ExportToFile("rsv", "rsv");
          ws.ExportToFile("sv", "sv");
          allkn1_args.Add("references_in", "lsv");
          allkn1_args.Add("indices_out", "indices_table");
          allkn1_args.Add("distances_out", "distances_table");
          fl::ml::AllKN<boost::mpl::void_>::Run(&ws, allkn1_args.args());
        } else {
          std::string references_in=std::string("lsv");
          std::string indices_out=std::string("indices_table");
          std::string distances_out=std::string("distances_table");
          ComputeNeighbors(ws, 
                           args,
                           references_in, 
                           indices_out, 
                           distances_out, 
                           refined_categories);
        }
        boost::shared_ptr<fl::ws::WorkSpace::UIntegerTable_t> indices_table;
        boost::shared_ptr<fl::ws::WorkSpace::DefaultTable_t> distances_table;
        ws.Attach("indices_table", &indices_table);
        ws.Attach("distances_table", &distances_table);
        std::string result1_file=vm["results1_out"].as<std::string>();
        if (result1_file!="") {
          ws.schedule(boost::bind(ExportProductNeighbors<fl::ws::WorkSpace::UIntegerTable_t,
            fl::ws::WorkSpace::DefaultTable_t>, 
          result1_file, 
          refined_labels, 
          boost::ref(*indices_table),
          boost::ref(*distances_table)));
        }
      } else {
        fl::logger->Die()<<"This dr_algortihm ("
         <<vm["dr_algorithm"].as<std::string>() 
         <<")  is not supported"<<std::endl;
      }
    }

    if (vm["bag_of_words_out"].as<std::string>()!="") {
      ws.ExportToFile("bag_of_words", vm["bag_of_words_out"].as<std::string>());
    }
    if (vm["dr_algorithm"].as<std::string>()=="nmf") {
  //    boost::shared_ptr<fl::ws::WorkSpace::DefaultSparseDoubleTable_t> product_graph;
  //    MakeProductGraph(&ws, "product_graph", *indices_table1, &product_graph);
      fl::ws::Arguments quicsvd_args;
      quicsvd_args.Add(fl::ws::MakeArgsFromPrefix(args, "qsvd"));
      quicsvd_args.Add("references_in", "product_graph");
      quicsvd_args.Add("lsv_out", "lsv");
      quicsvd_args.Add("rsv_out", "rsv");
      quicsvd_args.Add("sv_out", "sv");
      fl::ml::Svd<boost::mpl::void_>::Run(&ws, quicsvd_args.args());
  
      fl::ws::Arguments allkn2_args;
      allkn2_args.Add(fl::ws::MakeArgsFromPrefix(args, "allkn2"));
      ws.IndexTable("lsv", 
          "l2",
          "",
          10);
      ws.ExportToFile("lsv", "lsv");
      ws.ExportToFile("rsv", "rsv");
      ws.ExportToFile("sv", "sv");
      allkn2_args.Add("references_in", "lsv");
      allkn2_args.Add("indices_out", "indices_table_2");
      allkn2_args.Add("distances_out", "distances_table_2");
      fl::ml::AllKN<boost::mpl::void_>::Run(&ws, allkn2_args.args());
      boost::shared_ptr<fl::ws::WorkSpace::UIntegerTable_t> indices_table2;
      boost::shared_ptr<fl::ws::WorkSpace::DefaultTable_t> distances_table2;
      ws.Attach("indices_table_2", &indices_table2);
      ws.Attach("distances_table_2", &distances_table2);
      std::string result2_file=vm["results2_out"].as<std::string>();
      if (result2_file!="") {
        ws.schedule(boost::bind(ExportProductNeighbors<fl::ws::WorkSpace::UIntegerTable_t,
            fl::ws::WorkSpace::DefaultTable_t>, 
        result2_file, 
        refined_labels1, 
        boost::ref(*indices_table2),
        boost::ref(*distances_table2)));
      }
    }
  } 
  catch (const fl::Exception &exception) {
    return EXIT_FAILURE;
  }
}

void LoadProductLabels(const std::string &file, 
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
void ExportProductNeighbors(WorkSpaceType *ws,
    index_t reliability_threshold,
    const std::string &filename, 
    const std::vector<std::string> &labels,
    index_t nmf_runs,
    std::vector<std::string> *refined_labels) {
  
  std::ofstream fout;
  if (filename!="") {
    fout.open(filename.c_str());
    if (fout.fail()) {
      fl::logger->Die() << "Could not open file " << filename.c_str()
           << "   error: " << strerror(errno);
    }
  }

  std::vector<std::map<index_t, index_t> > neighbor_votes(labels.size());
  boost::shared_ptr<typename WorkSpaceType::UIntegerTable_t> indices_table;
  boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> distances_table;

  for(int r=0; r<nmf_runs; ++r) {
    std::string nmf_run(boost::lexical_cast<std::string>(r));
    ws->Attach(std::string("indices_table")+nmf_run, &indices_table);
    ws->Attach(std::string("distances_table")+nmf_run, &distances_table);
    typename WorkSpaceType::UIntegerTable_t::Point_t indices;
    typename WorkSpaceType::DefaultTable_t::Point_t distances;
    for(index_t i=0; i<indices_table->n_entries(); ++i) {
      indices_table->get(i, &indices);
      distances_table->get(i, &distances);
      for(index_t j=0; j<indices.size(); ++j) {
        if (neighbor_votes[i].count(indices[j])==0) {
          neighbor_votes[i][indices[j]]=1;
        } else {
          neighbor_votes[i][indices[j]]+=1;
        }
      }
    }
  }
  boost::shared_ptr<typename WorkSpaceType::DefaultSparseDoubleTable_t>  new_neighbors;
  index_t dimension=indices_table->n_entries();
  ws->Attach("product_graph",
      std::vector<index_t>(),
      std::vector<index_t>(1, dimension),
      0,
      &new_neighbors);
      
  // now output the neighbors with the higher votes
  for(index_t i=0; i<neighbor_votes.size(); ++i) {  
    std::vector<std::pair<index_t, index_t> > vec;
    for(std::map<index_t, index_t>::const_iterator it=neighbor_votes[i].begin();
        it!=neighbor_votes[i].end(); ++it) {
      vec.push_back(std::make_pair(it->second, it->first));
    }
    std::sort(vec.begin(), vec.end(), std::greater<std::pair<index_t, index_t> >());
    std::vector<std::pair<index_t, double> >  row;
    double row_sum=0;
    for(index_t j=0; j<vec.size(); ++j) {
      index_t votes=vec[j].first;
      if (votes<reliability_threshold) {
        continue;
      } else {
        row.push_back(std::make_pair(vec[j].second, 1));
        row_sum+=1;
      }
      std::string neighbor=labels[vec[j].second];
      std::string product=labels[i];
      if (filename!="") {
        fout << product <<" --- "<<neighbor<<" --- " << votes<<"\n";
      }
    }
    if (row.size()>0) {
      for(unsigned int k=0; k<row.size(); ++k) {
        row[k].second/=row_sum;
      }
      typename WorkSpaceType::DefaultSparseDoubleTable_t::Point_t point;
      std::vector<index_t> dim(1, dimension);
      point.Init(dim);
      point.template sparse_point<double>().Load(row.begin(), row.end());
      new_neighbors->push_back(point);
      refined_labels->push_back(labels[i]);
    }
  }
  fl::logger->Message()<<"Graph built with "<<refined_labels->size()
    <<" nodes"<<std::endl;
  ws->Purge("product_graph");
  ws->Detach("product_graph");
}

template<typename TableType1, typename TableType2>
void ExportProductNeighbors(
    const std::string &filename, 
    const std::vector<std::string> &labels,
    TableType1 &indices_table,
    TableType2 &distances_table) {

  std::ofstream fout(filename.c_str());
  if (fout.fail()) {
    fl::logger->Die() << "Could not open file " << filename.c_str()
         << "   error: " << strerror(errno);
  }

  typename TableType1::Point_t indices;
  typename TableType2::Point_t distances;
  for(index_t i=0; i<indices_table.n_entries(); ++i) {
    indices_table.get(i, &indices);
    distances_table.get(i, &distances);
    for(index_t j=0; j<indices.size(); ++j) {
      std::string neighbor=labels[indices[j]];
      std::string product=labels[i];
      fout << product <<" --- "<<neighbor<<" --- " << distances[j]<<"\n";
    }
  }
}


void ExportWordFrequencies(const std::string &filename, 
    const std::map<std::string, index_t> &wordfreq) {
  
  std::ofstream fout(filename.c_str());
  if (fout.fail()) {
    fl::logger->Die() << "Could not open file " << filename.c_str()
         << "   error: " << strerror(errno);
  }
  for(std::map<std::string, index_t>::const_iterator it=wordfreq.begin();
      it!=wordfreq.end(); ++it) {
    fout<<it->first<<" ---  "<<it->second<<"\n";
  }
}

template<typename TableType>
void ExportNmfClusters(TableType &w_factor, 
    const std::vector<std::string> &labels, 
    const std::string &filename) {

  std::ofstream fout(filename.c_str());
  if (fout.fail()) {
    fl::logger->Die() << "Could not open file " << filename.c_str()
         << "   error: " << strerror(errno);
  }
  index_t k_rank=w_factor.n_attributes();
  std::vector<std::map<std::string, double> > clusters(k_rank);
  typename TableType::Point_t point;
  for(index_t i=0; i<w_factor.n_entries(); ++i) {
    w_factor.get(i, &point);
    double max_value=-1;
    index_t arg_max=-1;
    for(index_t j=0; j<point.size(); ++j) {
      if (point[j]>max_value) {
        max_value=point[j];
        arg_max=j;
      }
    }
    std::string label=labels[i];
    clusters[arg_max][label]=max_value;
  }
  for(unsigned int i=0; i<clusters.size(); ++i) {
    fout<<"*** cluster "<<i<<" ***\n";
    for(std::map<std::string, double>::const_iterator it=clusters[i].begin();
        it!=clusters[i].end();
        ++it) {
      fout<<"\t"<<it->first<<" --- "<<it->second<<"\n";
    }
    fout<<"\n";
  }
}

template<typename WorkSpaceType, typename TableType1, typename TableType2>
void MakeProductGraph(WorkSpaceType *ws, const std::string &graph_name,
    TableType1 &indices_table, boost::shared_ptr<TableType2> *product_graph) {
  typename TableType1::Point_t point1;
  ws->Attach(graph_name, 
      std::vector<index_t>(),
      std::vector<index_t>(1, indices_table.n_entries()),
      0,
      product_graph);
  for(index_t i=0; i<indices_table.n_entries(); ++i) {
    indices_table.get(i, &point1);
    std::vector<std::pair<index_t, double> > vec;
    for(index_t j=0; j<point1.size(); ++j) {
      typename TableType1::Point_t p1;
      indices_table.get(point1[j], &p1);
      for(index_t k=0; k<p1.size(); ++k) {
        if (p1[k]==i) {
          vec.push_back(std::make_pair(point1[j],1));
        }
      }
    }
    typename TableType2::Point_t point2;
    std::vector<index_t> dim(1, (*product_graph)->n_attributes());
    point2.Init(dim);
    point2.template sparse_point<double>().Load(vec.begin(), vec.end());
    (*product_graph)->push_back(point2);
  }
  ws->Purge(graph_name);
  ws->Detach(graph_name);
}

template<typename WorkSpaceType>
void ComputeNeighbors(WorkSpaceType &ws, 
                      const std::vector<std::string> &args,
                      const std::string &references_in, 
                      const std::string &indices_out, 
                      const std::string &distances_out, 
                      const std::vector<std::string> &refined_categories) {

  boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> references;
  ws.Attach(references_in, &references);
  WorkSpaceType local_ws;
  local_ws.set_schedule_mode(1);
  std::map<
    std::string,
    boost::tuple<
      boost::shared_ptr<typename WorkSpaceType::DefaultTable_t>,
      std::vector<index_t>
    >
   > tables;
  for(index_t i=0; i<refined_categories.size(); ++i) {
    if (tables.count(refined_categories[i])==0) {
      local_ws.Attach(
          refined_categories[i],
          std::vector<index_t>(1,1),
          std::vector<index_t>(),
          0,
          &(tables[refined_categories[i]]). template get<0>());
    }
    typename WorkSpaceType::DefaultTable_t::Point_t point;
    references->get(i, &point);
    tables[refined_categories[i]].template get<0>()->push_back(point);
    tables[refined_categories[i]].template get<1>().push_back(i);
  }
  typename std::map<std::string, 
    boost::tuple<
      boost::shared_ptr<typename WorkSpaceType::DefaultTable_t>,
      std::vector<index_t>
    >    
   >::iterator it;
  for(it=tables.begin(); it!=tables.end(); ++it) {
    // first we need to detach the small tables
    local_ws.Purge(it->first);
    local_ws.Detach(it->first);
    fl::ws::Arguments allkn_args;
    allkn_args.Add(fl::ws::MakeArgsFromPrefix(args, "allkn1"));
    local_ws.IndexTable(it->first, 
          "l2",
          "",
          10);
    allkn_args.Add("references_in", it->first);
    allkn_args.Add("indices_out", std::string("indices_table:")+it->first);
    allkn_args.Add("distances_out", std::string("distances_table:")+it->first);
    fl::ml::AllKN<boost::mpl::void_>::Run(&local_ws, allkn_args.args()); 
    boost::shared_ptr<typename WorkSpaceType::UIntegerTable_t> indices1;
    //local_ws.Attach(std::string("indices_table:")+it->first, &indices1);
  }  
  
  boost::shared_ptr<typename WorkSpaceType::UIntegerTable_t> indices_table;
  boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> distances_table;
  boost::shared_ptr<typename WorkSpaceType::UIntegerTable_t> indices1;
  boost::shared_ptr<typename WorkSpaceType::DefaultTable_t> distances1;
  // we need this trick so that we get the number of neighbors
  local_ws.Attach(std::string("indices_table:")+tables.begin()->first, &indices1);

  ws.Attach("indices_table", 
      std::vector<index_t>(1, indices1->n_attributes()),
      std::vector<index_t>(),
      references->n_entries(),
      &indices_table);
  ws.Attach("distances_table", 
      std::vector<index_t>(1, indices1->n_attributes()),
      std::vector<index_t>(),
      references->n_entries(),
      &distances_table);
 
  for(it=tables.begin(); it!=tables.end(); ++it) {
    local_ws.Attach(std::string("indices_table:")+it->first, &indices1);
    local_ws.Attach(std::string("distances_table:")+it->first, &distances1);
    for(index_t i=0; i<indices1->n_entries(); ++i) {
      typename WorkSpaceType::UIntegerTable_t::Point_t ind1, ind2;
      typename WorkSpaceType::DefaultTable_t::Point_t dist1, dist2;
      indices1->get(i, &ind1);
      distances1->get(i, &dist1);
      index_t point_id=it->second. template get<1>()[i]; 
      indices_table->get(point_id, &ind2);
      distances_table->get(point_id, &dist2);
      for(index_t j=0; j<indices_table->n_attributes();  ++j) {
        ind2.set(j, it->second. template get<1>()[ind1[j]]);
        dist2.set(j, dist1[j]);
      }
    }
  }
  ws.Purge("distances_table");
  ws.Detach("distances_table");
  ws.Purge("indices_table");
  ws.Detach("indices_table");
}


