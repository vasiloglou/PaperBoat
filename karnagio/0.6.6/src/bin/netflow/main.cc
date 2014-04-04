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
#include <vector>
#include <string>
#include <fstream>
#include "mlpack/graph_diffuser/graph_diffuser.h"
#include "mlpack/clustering/kmeans.h"
#include "mlpack/svd/svd.h"
#include "fastlib/workspace/workspace_defs.h"
#include "fastlib/table/branch_on_table_dev.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/data/multi_dataset_dev.h"
#include "fastlib/workspace/arguments.h"

void LoadIps(const std::string &filename,
    std::vector<std::string> *ips) {
  std::ifstream fin(filename.c_str());
  if (fin.fail()) {
    fl::logger->Die() << "Could not open file " << filename.c_str()
      << "   error: " << strerror(errno);
  }
  while (fin.good()) {
    std::string line;
    std::getline(fin, line);
    boost::algorithm::trim_if(line,  boost::algorithm::is_any_of(" ,\t\r\n"));
    ips->push_back(line);
  }  
}

template<typename TableType>
void ExportSimmilarity(
    const std::string &filename,
    TableType &table,
    const std::vector<std::string> &ips
    ) {
  std::ofstream fout(filename.c_str());
  if (fout.fail()) {
    fl::logger->Die() << "Could not open file " << filename.c_str()
        << "   error: " << strerror(errno);
  }
  typename TableType::Point_t point;
  for(index_t i=0; i<table.n_entries(); ++i) {
    table.get(i, &point);
    fout<<ips[i]<<" => ";
    for(typename TableType::Point_t::iterator it=point.begin();
        it!=point.end(); ++it) {
      fout<<ips[it.attribute()]<<",";
    }
    fout<<std::endl;
  }
}

template<typename TableType>
void ExportClusters(
    const std::string &filename,
    const int32 k_clusters,
    TableType &table,
    const std::vector<std::string> &ips) {
  std::ofstream fouts[k_clusters];
  for(size_t i=0; i<k_clusters;++i) {
    fouts[i].open((filename+boost::lexical_cast<std::string>(i)).c_str());
    if (fouts[i].fail()) {
      fl::logger->Die() << "Could not open file " 
        << filename+boost::lexical_cast<std::string>(i)
        << "   error: " << strerror(errno);
    }
  }
  typename TableType::Point_t point;
  for(index_t i=0; i<table.n_entries(); ++i) {
    table.get(i, &point);
    fouts[int(point[0])]<<ips[i]<<std::endl;
  }
}


int main(int argc, char *argv[]) {
  fl::logger->SetLogger("debug");
  // Convert C input to C++; skip executable name for Boost
  std::vector<std::string> args(argv + 1, argv + argc);
  FL_SCOPED_LOG(NetFlowAnalyzer);
  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "flow_packets_in",
    boost::program_options::value<std::string>(),
    "a sparse table between ip ids and packet flow"
  )(
    "ip2ids_in",
    boost::program_options::value<std::string>(),
    "a file that contains the ip address for every row of the "
    "--flow_packets_in " 
  )(
    "ip_cluster_prefix_out",
    boost::program_options::value<std::string>(),
    "if you perform clustering then the results will be printed in a set "
    "of k files with this prefix"  
  )(
    "ip_simmilarity_out",
    boost::program_options::value<std::string>(),
    "if you just perform simillarity between the ip address" 
  )(
    "action",
    boost::program_options::value<std::string>()->default_value("graphd"),
    "What to do with the data, can be either graphd or kmeans" 
  );
      
  boost::program_options::variables_map vm;
  std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args, "");
  boost::program_options::command_line_parser clp(args1);
  clp.style(boost::program_options::command_line_style::default_style
     ^boost::program_options::command_line_style::allow_guessing);
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
     fl::logger->Die() << e.what()
      <<" . This option will be ignored";
  }
  catch ( const boost::program_options::error &e) {
    fl::logger->Die() << e.what();
  } 
  boost::program_options::notify(vm);
  if (vm.count("help")) {
    fl::logger->Message() << fl::DISCLAIMER << "\n";
    fl::logger->Message() << desc << "\n";
    return EXIT_SUCCESS;
  }

  try {
    // Use a generic workspace model
    fl::ws::WorkSpace ws;
    ws.set_schedule_mode(2);
    ws.set_pool(2);
    std::vector<std::string> ips;
    if (vm.count("ip2ids_in")>0) {
      LoadIps(vm["ip2ids_in"].as<std::string>(), &ips);
    }
    
    if (vm.count("flow_packets_in")==0) {
      fl::logger->Die()<<"You must provide --flow_packets_in ";
    } else {
      ws.LoadDataTableFromFile(vm["flow_packets_in"].as<std::string>(),
          vm["flow_packets_in"].as<std::string>());
    }

    if (vm["action"].as<std::string>()=="graphd") {
      std::vector<std::string> args2=fl::ws::MakeArgsFromPrefix(args, "graphd");
      ws.LoadAllTables(args2);
      args2.push_back("--graph_out=graph");
      args2.push_back("--references_in="+ 
          vm["flow_packets_in"].as<std::string>());
      fl::ml::GraphDiffuser<boost::mpl::void_>::Run(&ws, args2);
      if (vm.count("ip_simmilarity_out")) {
        boost::shared_ptr<fl::ws::WorkSpace::DefaultSparseDoubleTable_t> table;
        ws.Attach("graph", &table);
        ExportSimmilarity(
          vm["ip_simmilarity_out"].as<std::string>(),
          *table,
          ips);
        ws.ExportAllTables(args2);
      }
    } else {
      if (vm["action"].as<std::string>()=="kmeans") {
        std::vector<std::string> args3=fl::ws::MakeArgsFromPrefix(args, "svd");
        ws.LoadAllTables(args3);
        args3.push_back("--lsv_out=lsv");
        args3.push_back("--references_in="+ 
          vm["flow_packets_in"].as<std::string>());
        fl::ml::Svd<boost::mpl::void_>::Run(&ws, args3);
        ws.ExportAllTables(args3);

        std::vector<std::string> args4=fl::ws::MakeArgsFromPrefix(args, "kmeans");
        ws.LoadAllTables(args4);
        args4.push_back("--references_in=lsv");
        std::string memberships_name=ws.GiveTempVarName();
        args4.push_back("--memberships_out="+memberships_name);
        ws.IndexAllReferencesQueries(args4);
        fl::ml::KMeans<boost::mpl::void_>::Run(&ws, args4);
        if (vm.count("ip_cluster_prefix_out")>0) {
          std::map<std::string, std::string> map=fl::ws::GetArgumentPairs(args4);
          boost::shared_ptr<fl::ws::WorkSpace::UIntegerTable_t> table; 
          ws.Attach(memberships_name, &table);
          ExportClusters(
              vm["ip_cluster_prefix_out"].as<std::string>(),
              boost::lexical_cast<int32>(map["--k_clusters"]),
              *table,
              ips);
        } 
        ws.ExportAllTables(args4);
      } else {
        fl::logger->Die()<<"this option --action="
          << vm["action"].as<std::string>()
          << " is not supported";
      }
    }

  } catch (const fl::Exception &exception) {
    return EXIT_FAILURE;
  }
}
