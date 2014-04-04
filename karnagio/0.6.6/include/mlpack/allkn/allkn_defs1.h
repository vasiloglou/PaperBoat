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
#ifndef FL_LITE_MLPACK_ALLKN_ALLKN_DEFS1_H_
#define FL_LITE_MLPACK_ALLKN_ALLKN_DEFS1_H_

#include <string>
#include <iostream>
#include "allkn.h"
#include "fastlib/workspace/task.h"

namespace fl {
namespace ml {


template<typename DataAccessType, typename BranchType>
int AllKN<boost::mpl::void_>::Main(
  DataAccessType *data,
  const std::vector<std::string> &args
) {
  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::string>(),
    "REQUIRED file containing reference data"
  )(
    "queries_in",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file containing query positions.  If omitted, allkn "
    "finds leave-one-out neighbors for each reference point."
  )(
    "indices_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file to store found neighbor indices"
  )(
    "indices_in",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file containing neighbor pairs. "
    "if flag --method=classification then you should provide --indices_in"
  )(
    "distances_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file to store found neighbor distances"
  )(
    "distances_in",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file containing distances between neighboring points. "
    "if flag --method=classification then you should provide --distances_in"
  )(
    "reference_labels_in",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file containing the labels of reference points. "
    "if flag --method=classification then you should provide --reference_labels_in"
  )(
    "query_labels_in",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file containing the labels of the query points."
  )(
    "labels_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL FILE for outputting the computed nearest neighbor classification "
    "labels (for --method=classification or --method=nnclassification)."
  )(
    "method",
    boost::program_options::value<std::string>()->default_value("nearest"),
    "Which neighbors method to perform.  One of:\n"
    "  nearest: computes nearest neighbors \n"
    "  furthest: computes furthest neighbors \n"
    "  nnclassification: compute  nearest neighbors and do classification\n"
    "  classification: compute the classification scores based on precomputed neighbors" 
  )(
    "k_neighbors",
    boost::program_options::value<index_t>()->default_value(-1),
    "The number of neighbors to find for the all-k-neighbors method."
  )(
    "r_neighbors",
    boost::program_options::value<double>()->default_value(-1.0),
    "The query radius for the all-range-neighbors method.\n"
    "One of --k_neighbors or --r_neighbors must be given."
  )(
    "metric",
    boost::program_options::value<std::string>()->default_value("l2"),
    "Metric function used by allkn.  One of:\n"
    "  l2, weighted_l2, hellinger"
  )(
    "metric_weights_in",
    boost::program_options::value<std::string>()->default_value(""),
    "A file containing weights for use with --metric=weighted_l2"
  )(
    "dense_sparse_scale",
    boost::program_options::value<double>()->default_value(1.0),
    "The scaling factor for the sparse part of --point=dense_sparse or "
    "--point=dense_categorical for use with --metric=weighted_l2"
  )(
    "algorithm",
    boost::program_options::value<std::string>()->default_value("dual"),
    "Algorithm used to compute neighbors.  One of:\n"
    "  dual, single"
  )("auc", 
    boost::program_options::value<bool>()->default_value("true"),
    "If this flag is set to true then the classifier computes the "
    "Area Under Curve (AUC) score"
  )("auc_label",
    boost::program_options::value<int>()->default_value(1),
    "label of the class based on which the AUC will be computed"
  )(
    "roc_out",
    boost::program_options::value<std::string>()->default_value(""),
    "OPTIONAL file for exporting the ROC curve"
  )(
    "iterations",
    boost::program_options::value<index_t>()->default_value(-1),
    "Allkn can run in either batch or progressive mode.  If --iterations=i "
    "is omitted, allkn finds exact neighbors; otherwise, it terminates after "
    "i progressive refinements."
  )(
    "max_trace_size",
    boost::program_options::value<index_t>()->default_value(std::numeric_limits<index_t>::max()),
    "when running in iterative mode, the trace can become too big, so we need to limit it" 
  )(
    "log_tree_stats",
    boost::program_options::value<bool>()->default_value(true),
    "If this flag is set true then it outputs some statistics about the tree after it is built. "
    "We suggest you set that flag on. If the tree is not correctly built, due to wrong options"
    " or due to pathological data then there is not point in running nearest neighbors"
  );
  boost::program_options::variables_map vm;
  boost::program_options::command_line_parser clp(args);
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
  
  if(vm["method"].as<std::string>() == "nnclassification" && 
         vm["reference_labels_in"].as<std::string>() == "") {
    fl::logger->Die() << "You need to specify the file that contains the labels.\n";
  }
 
  if(vm["method"].as<std::string>() == "classification" &&
     vm["indices_in"].as<std::string>() == "") {
    fl::logger->Die() << "You need to specify the file that contains the "
      "precomputed neighbor indices.\n";
  }
 
  return BranchType::template BranchOnTable<AllKN<boost::mpl::void_>, DataAccessType>(data, vm);
}

template<typename DataAccessType>
void fl::ml::AllKN<boost::mpl::void_>::Run(
    DataAccessType *data,
    const std::vector<std::string> &args) {
  fl::ws::Task<
    DataAccessType, 
    &Main<DataAccessType,
      typename DataAccessType::Branch_t
    >
  > task(data, args);
  data->schedule(task); 
}

}} // namespaces

#endif
