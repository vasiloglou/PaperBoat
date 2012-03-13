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
#ifndef FL_LITE_MLPACK_MVU_MVU_OBJECTIVES_DEFS_H_
#define FL_LITE_MLPACK_MVU_MVU_OBJECTIVES_DEFS_H_
#include <string>
#include <vector>
#include "boost/mpl/void.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/mpl/at.hpp"
#include "boost/program_options.hpp"
#include "mlpack/allkn/allkn.h"
#include "fastlib/la/linear_algebra.h"
#include "fastlib/metric_kernel/lmetric.h"
#include "mvu_objectives.h"
#include "fastlib/optimization/augmented_lagrangian/lbfgs.h"
#include "fastlib/optimization/augmented_lagrangian/optimization_utils.h"
#include "fastlib/workspace/task.h"

namespace fl {
namespace ml {

template<typename TableType>
template<typename DataAccessType>
int DimensionalityReduction<boost::mpl::void_>::Core<TableType>::Main(DataAccessType *data,
                      boost::program_options::variables_map &vm) {

  typedef TableType Table_t;
  fl::optim::LbfgsOpts<double> lbfgs_opts;


  lbfgs_opts.sigma = vm["sigma"].as<double>();
  lbfgs_opts.objective_factor = vm["objective_factor"].as<double>();
  lbfgs_opts.eta = vm["eta"].as<double>();
  lbfgs_opts.gamma = vm["gamma"].as<double>();
  lbfgs_opts.new_dimension = vm["new_dimension"].as<index_t>();
  lbfgs_opts.feasibility_tolerance = vm["feasibility_tolerance"].as<double>();
  lbfgs_opts.desired_feasibility = vm["desired_feasibility"].as<double>();
  lbfgs_opts.wolfe_sigma1 = vm["wolfe_sigma1"].as<double>();
  lbfgs_opts.wolfe_sigma2 = vm["wolfe_sigma2"].as<double>();
  lbfgs_opts.step_size = vm["step_size"].as<double>();
  lbfgs_opts.silent = vm["silent"].as<bool>();
  lbfgs_opts.show_warnings = vm["show_warnings"].as<bool>();
  lbfgs_opts.use_default_termination = vm["use_default_termination"].as<bool>();
  lbfgs_opts.norm_grad_tolerance = vm["norm_grad_tolerance"].as<double>();
  lbfgs_opts.wolfe_beta = vm["wolfe_beta"].as<double>();
  lbfgs_opts.min_beta = vm["min_beta"].as<double>();
  lbfgs_opts.max_iterations = vm["max_iterations"].as<index_t>();
  lbfgs_opts.mem_bfgs = vm["mem_bfgs"].as<index_t>();

  index_t knns =  vm["k_neighbors"].as<index_t>();
  typename MVUObjective::MVUOpts mvu_opts;
  mvu_opts.knns = knns;
  mvu_opts.new_dimension = vm["new_dimension"].as<index_t>();
  mvu_opts.auto_tune = vm["auto_tune"].as<bool>();
  mvu_opts.desired_feasibility_error = vm["desired_feasibility"].as<double>();
  mvu_opts.infeasibility_tolerance = vm["feasibility_tolerance"].as<double>();;
  mvu_opts.grad_tolerance = vm["norm_grad_tolerance"].as<double>();

  typename MFNUObjective::MFNOpts mfn_opts;
  mfn_opts.knns = knns;
  mfn_opts.new_dimension = vm["new_dimension"].as<index_t>();
  mfn_opts.auto_tune = vm["auto_tune"].as<bool>();
  mfn_opts.desired_feasibility_error = vm["desired_feasibility"].as<double>();
  mfn_opts.infeasibility_tolerance = vm["feasibility_tolerance"].as<double>();;
  mfn_opts.grad_tolerance = vm["norm_grad_tolerance"].as<double>();

  std::string result_file = vm["result_out"].as<std::string>();

  if (vm.count("nn_file") == false) {
    std::string data_file = vm["references_in"].as<std::string>();
    boost::shared_ptr<Table_t> table;
    fl::logger->Message() << "Loading reference data from " << data_file << std::endl;
    data->Attach(data_file, &table);
    boost::shared_ptr<typename DataAccessType::DefaultTable_t> result;
    const int num_of_points=table->n_entries();
    data->Attach(result_file,
                 std::vector<index_t>(1, mfn_opts.new_dimension),
                 std::vector<index_t>(),
                 num_of_points,     
                 &result);

    fl::logger->Message() << "Completed reference data loading" <<std::endl;
    lbfgs_opts.num_of_points = table->n_entries();
    mvu_opts.num_of_points = table->n_entries();
    mfn_opts.num_of_points = table->n_entries();

    typename Table_t::template IndexArgs<fl::math::LMetric<2> > index_args;
    index_args.leaf_size = vm["leaf_size"].as<index_t>();
    fl::logger->Message() << "Building index on reference data" <<std::endl;
    table->IndexData(index_args);
    fl::logger->Message() << "Index built"<<std::endl;
    DefaultAllKNN allknn;
    allknn.Init(table.get(), NULL);
    std::vector<index_t> ind_neighbors(
      table->n_entries() * knns);
    std::vector<double> dist_neighbors(
      table->n_entries() * knns);

    mvu_opts.from_tree_neighbors = &ind_neighbors;
    mvu_opts.from_tree_distances = &dist_neighbors;
    mfn_opts.from_tree_neighbors = &ind_neighbors;
    mfn_opts.from_tree_distances = &dist_neighbors;

    fl::logger->Message() << "Computing nearest neighbors for KNN=" << knns << std::endl;
    // compute neighbors
    allknn.ComputeNeighbors(std::string("dual"),
                            index_args.metric,
                            knns,
                            &dist_neighbors,
                            &ind_neighbors);

    fl::logger->Message() << "Computation of neighbors completed"<<std::endl;

    // we need to insert the number of points
    lbfgs_opts.num_of_points = table->n_entries();
    bool done = false;
    std::string optimized_function = vm["optimized_function"].as<std::string>();
    if (optimized_function == "mvu") {
      MVUObjective opt_function;
      opt_function.Init(mvu_opts);
      fl::optim::Lbfgs <LbfgsTypeOptsMVU> engine;
      engine.Init(&opt_function, lbfgs_opts);
      engine.ComputeLocalOptimumBFGS();
      fl::logger->Message() << "Emitting results"<<std::endl;
      for(index_t i=0; i<engine.coordinates()->n_cols(); ++i) {
        typename DataAccessType::DefaultTable_t::Point_t point;
        result->get(i, &point);
        for(index_t j=0; j<point.size(); ++i) {
          point.set(j, engine.coordinates()->get(j, i));
        }
      }
      data->Purge(result_file);
      data->Detach(result_file);
      fl::logger->Message() << "Finished emitting results"<<std::endl;
      done = true;
    }
    if (optimized_function == "mfnu") {
      DefaultAllKFN allkfn;
      allkfn.Init(table.get(), NULL);
      std::vector<index_t> furthest_ind_neighbors(
        table->n_entries() *1);
      std::vector<double> furthest_dist_neighbors(
        table->n_entries() * 1);

      mfn_opts.from_tree_furhest_neighbors = &furthest_ind_neighbors;
      mfn_opts.from_tree_furthest_distances = &furthest_dist_neighbors;

      fl::logger->Message() << "Computing furthest neighbors "<<std::endl;
      // compute neighbors
      allkfn.ComputeNeighbors(std::string("dual"),
                              index_args.metric,
                              1,
                              &furthest_dist_neighbors,
                              &furthest_ind_neighbors);
      fl::logger->Message() << "Computation of neighbors completed"<<std::endl;
      MFNUObjective opt_function;
      opt_function.Init(mfn_opts);
      fl::optim::Lbfgs<LbfgsTypeOptsMFNU> engine;
      engine.Init(&opt_function, lbfgs_opts);
      engine.ComputeLocalOptimumBFGS();
      fl::logger->Message() << "Emitting results"<<std::endl;
      for(index_t i=0; i<engine.coordinates()->n_cols(); ++i) {
        typename DataAccessType::DefaultTable_t::Point_t point;
        result->get(i, &point);
        for(index_t j=0; j<point.size(); ++i) {
          point.set(j, engine.coordinates()->get(j, i));
        }
      }
      data->Purge(result_file);
      data->Detach(result_file);
      fl::logger->Message() << "Finished emitting results"<<std::endl;
      done = true;
    }
    if (done == false) {
      fl::logger->Die() << "The method you provided "<< optimized_function << "is not supported";
            
    }
  } else {
    fl::logger->Die() << "Running MVU from a nearest neighbor file " << fl::NOT_SUPPORTED_MESSAGE;

    /*    bool done=false;

        if (optimized_function == "mvu") {
          MVUObjective opt_function;
          opt_function.Init(mvu_opts);
          fl::dense::Matrix<double> init_mat;
          //we need to insert the number of points
          LBfgs<
            boost::mpl::map1<
              fl::ml::LbfgsTypeOpts::OptimizedFunctionType,
              MVUObjective
            >
          > engine;
           engine.Init(&opt_function, &lbfgs_opts);
          engine.ComputeLocalOptimumBFGS();
          MultiDatasetDenseDouble result;
          result.Init(*engine.coordinates());
          result.Save(result_file,
                      false,
                      std::vector<std::string>(),
                      "");
          done=true;
        }
        if (optimized_function == "mfnu"){
          MFNU opt_function;
          opt_function.Init(mfn_opts);
          //we need to insert the number of points
          lbfgs_opts.num_of_points=opt_function.num_of_points();
          LBfgs<
            boost::mpl::map1<
              fl::ml::LbfgsTypeOpts::OptimizedFunctionType,
              MFNUObjective
            >
          > engine;
          engine.Init(&opt_function, &lbfgs_opts);
          engine.ComputeLocalOptimumBFGS();
          MultiDatasetDenseDouble result;
          result.Init(*engine.coordinates());
          result.Save(result_file,
                      false,
                      std::vector<std::string>(),
                      "");
          done=true;
        }
        if (done==false) {
          FATAL("The method you provided %s is not supported",
              optimized_function.c_str());
        }
        */
  }
  return 0;
}

template<typename DataAccessType, typename BranchType>
int DimensionalityReduction<boost::mpl::void_>::Main(DataAccessType *data,
                                  const std::vector<std::string> &args) {

  fl::optim::LbfgsOpts<double> lbfgs_opts;
  boost::program_options::options_description desc("Available options");
  desc.add_options()
  ("help", "Maximum Variance Unfolding, Maximum Furthest Neighbor Unfolding")
  ("sigma", boost::program_options::value<double>()->default_value(lbfgs_opts.sigma),
   "  The initial penalty parameter on the augmented lagrangian.\n")
  ("objective_factor", boost::program_options::value<double>()->default_value(
     lbfgs_opts.objective_factor),
   "  obsolete.\n")
  ("eta", boost::program_options::value<double>()->default_value(lbfgs_opts.eta),
   "  wolfe parameter.\n")
  ("gamma", boost::program_options::value<double>()->default_value(lbfgs_opts.gamma),
   "  sigma increase rate, after inner loop is done sigma is multiplied by gamma.\n")
  ("desired_feasibility", boost::program_options::value<double>()->default_value(
     0.1),
   "  Since this is used with augmented lagrangian, we need to know "
   "when the  feasibility is sufficient.\n")
  ("feasibility_tolerance", boost::program_options::value<double>()->default_value(
     lbfgs_opts.feasibility_tolerance),
   "  if the feasibility is not improved by that quantity, then it stops.\n")
  ("wolfe_sigma1", boost::program_options::value<double>()->default_value(
     lbfgs_opts.wolfe_sigma1),
   "  wolfe parameter.\n")
  ("wolfe_sigma2", boost::program_options::value<double>()->default_value(
     lbfgs_opts.wolfe_sigma2),
   "  wolfe parameter.\n")
  ("min_beta", boost::program_options::value<double>()->default_value(
     lbfgs_opts.min_beta),
   "  wolfe parameter.\n")
  ("wolfe_beta", boost::program_options::value<double>()->default_value(
     lbfgs_opts.wolfe_beta),
   "  wolfe parameter.\n")
  ("step_size", boost::program_options::value<double>()->default_value(
     lbfgs_opts.step_size),
   "  Initial step size for the wolfe search.\n")
  ("silent", boost::program_options::value<bool>()->default_value(
     lbfgs_opts.silent),
   "  if true then it doesn't emmit updates.\n")
  ("show_warnings", boost::program_options::value<bool>()->default_value(
     lbfgs_opts.show_warnings),
   "  if true then it does show warnings.\n")
  ("use_default_termination", boost::program_options::value<bool>()->default_value(false),
   "  let this module decide where to terminate. If false then"
   " the objective function decides .\n")
  ("norm_grad_tolerance", boost::program_options::value<double>()->default_value(
     lbfgs_opts.norm_grad_tolerance),
   "  If the norm of the gradient doesn't change more than "
   "this quantity between two iterations and the use_default_termination "
   "is set, the algorithm terminates.\n")
  ("max_iterations", boost::program_options::value<index_t>()->default_value(
     lbfgs_opts.max_iterations),
   "  maximum number of iterations required.\n")
  ("mem_bfgs", boost::program_options::value<index_t>()->default_value(
     lbfgs_opts.mem_bfgs),
   "  the limited memory of BFGS.\n")
  ("nn_file", boost::program_options::value<std::string>(),
   "if the nearest neighbors are precomputed they can be provided as a text file")
  ("fn_file", boost::program_options::value<std::string>()->default_value(""),
   "if the furthest neighbors are precomputed they can be provided as a text file")
  ("k_neighbors", boost::program_options::value<index_t>()->default_value(5),
   "If the number of nearest neighbors is negative then it switches to autotuning")
  ("leaf_size", boost::program_options::value<index_t>()->default_value(30),
   "Size of a leaf node in the tree when knns are computed")
  ("references_in", boost::program_options::value<std::string>(),
   "File that containes raw data")
  ("result_out", boost::program_options::value<std::string>(),
   "The result of MVU, MFNU stored in a file")
  ("auto_tune", boost::program_options::value<bool>()->default_value(false),
   "if this flag is set to true then it automatically finds the right k for mvu")
  ("new_dimension", boost::program_options::value<index_t>()->default_value(2),
   "The new dimension that the data will be projected with MVU or MFNU")
  ("algorithm", boost::program_options::value<std::string>()->default_value("mfnu"),
   "Optimization method MFNU or MVU")
  ("cores",
    boost::program_options::value<int>()->default_value(1),
    "Number of cores to use for running the algorithm. If you use large number of cores "
    "increase the leaf_size" )
  ("log",
    boost::program_options::value<std::string>()->default_value(""),
    "A file to receive the log, or omit for stdout.")
  ( "loglevel",
    boost::program_options::value<std::string>()->default_value("debug"),
    "Level of log detail.  One of:\n"
    "  debug: log everything\n"
    "  verbose: log messages and warnings\n"
    "  warning: log only warnings\n"
    "  silent: no logging" );

  boost::program_options::variables_map vm;
  boost::program_options::command_line_parser clp(args);
  clp.style(boost::program_options::command_line_style::default_style
     ^boost::program_options::command_line_style::allow_guessing );
  boost::program_options::store(clp.options(desc).run(), vm);
  boost::program_options::notify(vm);
  if (vm.count("help")) {
    std::cout << fl::DISCLAIMER << "\n";
    std::cout << desc << "\n";
    return 1;
  }

  return BranchType::template BranchOnTable<DimensionalityReduction, DataAccessType>(data, vm);
}

template<typename DataAccessType>
void DimensionalityReduction<boost::mpl::void_>::Run(
      DataAccessType *data,
      const std::vector<std::string> &args) {
  fl::ws::Task<
    DataAccessType,
    &Main<
       DataAccessType, 
       typename DataAccessType::Branch_t
    > 
  > task(data, args);
  data->schedule(task); 
}

}}
#endif
