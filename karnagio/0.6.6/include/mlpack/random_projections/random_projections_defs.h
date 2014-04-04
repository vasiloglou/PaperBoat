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

#ifndef PAPERBOAT_MLPACK_RANDOM_PROJECTIONS_RANDOM_PROJECTIONS_DEFS_H_
#define PAPERBOAT_MLPACK_RANDOM_PROJECTIONS_RANDOM_PROJECTIONS_DEFS_H_

#include <string>
#include "boost/program_options.hpp"
#include "fastlib/dense/matrix.h"
#include "fastlib/workspace/task.h"
#include "fastlib/workspace/arguments.h"
#include "fastlib/table/linear_algebra.h"
#include "fastlib/workspace/based_on_table_run.h"
#include "random_projections.h"
#include "fastlib/table/linear_algebra.h"
#include "fastlib/util/string_utils.h"

namespace fl { namespace ml {
  template<typename WorkSpaceType>
  template<typename TableType>
  void RandomProjections<boost::mpl::void_>::Core<WorkSpaceType>::operator()(
      TableType&) {
    FL_SCOPED_LOG(random_projection);
    boost::program_options::options_description desc("Available options");
    desc.add_options()(
      "help", "Print this information."
    )(
      "references_prefix_in",
      boost::program_options::value<std::string>(),
      "the reference data prefix"
    )(
      "references_num_in",
      boost::program_options::value<int32>(),
      "number of references file with the prefix defined above"
    )(
      "references_in",
       boost::program_options::value<std::string>(),
      "the reference data comma separated. Each file is a time series"
      " eg --references_in=myfile,yourfile,hisfile"
    )(
      "projection_rank",
      boost::program_options::value<int32>(),
      "the dimensionality of the data after the projection"
    )(
      "projection_type",
      boost::program_options::value<std::string>()->default_value("gaussian_static"),
      "There are different types of random projection: \n"
      "gaussian_static: generates a gaussian matrix and then left multiplies the references_in\n"
      "gaussian_dynamic: generates random numbers on the fly and multiplies the references_in\n"
      "sparse: uses a sparse matrix for the projection"
    )(
      "projection_matrix_in",
      boost::program_options::value<std::string>(),
      "The projection matrix to be used. If it is not defined, then it is generated, "
      "it can also be a list of matrices"
    )(
      "projection_matrix_prefix_in",
      boost::program_options::value<std::string>(),
      "prefix of projection matrix series to be used. The partition is along "
      "the dimensionality of the reference table. If for example the references_in "
      "has N dimensions and there are 10 projection matrix. Each matrix has N div 10 dimensions "
      " and the last one if the division is not exact has N mod 10." 
    )(
      "projection_matrix_out",
      boost::program_options::value<std::string>(),
      "export the projection matrix"
    )(
      "projection_matrix_prefix_out",
      boost::program_options::value<std::string>(),
      "it is possible that the the gaussian matrix is too big to fit in memory "
      "so we might have to store it in multiple files"
    )(
      "projection_matrix_num_out",
      boost::program_options::value<std::string>(),
      "the number of the tabled for the corresponding prefix"
    )(
      "projected_out",
      boost::program_options::value<std::string>(),
      "a comma separated list of the filenames for exporting the projected tables"
    )(
      "projected_prefix_out",
      boost::program_options::value<std::string>(),
      "the prefix for storing the projected filenames. If the prefix is temp, then "
      "the projected tables will be stored in filenames temp1,temp2,...tempN-1 where "
      "N is the --projected_num_out=N"
    )(
      "projected_num_out",
      boost::program_options::value<int32>(),
      "the number of expected projected tables. It must be the same size with the number "
      "of references tables"
    )(
      "gaussian_chunk_dimensionality",
      boost::program_options::value<index_t>()->default_value(-1),
      "if the --references_in is super high dimensional then the gaussian matrix "
      "does not fit in memory so we have to chunk it,  if it is set to -1 then no chunking "
      "happens"  
    );
    boost::program_options::variables_map vm;
    std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args_, "");
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
      return ;
    }
    if (vm.count("projection_rank")==0) {
      fl::logger->Die()<<"--projection_rank must be set";
    }

    std::vector<std::string> projection_matrix;
    if (vm.count("projection_matrix_in") || vm.count("projection_matrix_prefix_in")) {
      projection_matrix=fl::ws::GetFileSequence("projection_matrix", vm);
    }

    if (projection_matrix.size()>0 && 
          vm.count("gaussian_chunk_dimensionality")>0) {
      fl::logger->Die()<<"If you have already provided projection matrices, then "
        "there is no point in setting --gaussian_chunk_dimensionality";
    }
    std::vector<std::string> references=fl::ws::GetFileSequence("references", vm);
    index_t dimensionality=0;
    ws_->GetTableInfo(references[0], NULL, &dimensionality, NULL, NULL);
    fl::logger->Message()<<"references dimensionality is ("<<dimensionality<<")";
    std::vector<std::string> projected_names;
    if (vm.count("projected_out")>0 || vm.count("projected_prefix_out")>0) {
      projected_names=fl::ws::GetFileSequence(
            "projected", vm); 
    } else {
      fl::logger->Die()<<"--projected_out or projected_prefix_out is required";
    }
    if (vm["projection_type"].as<std::string>()=="gaussian_static") {
      fl::logger->Message()<<"Using gaussian_static for projection"<<std::endl;
      std::vector<std::string> projection_matrix;
      if (vm.count("projection_matrix_in") || vm.count("projection_matrix_prefix_in")) {
        projection_matrix=fl::ws::GetFileSequence("projection_matrix", vm);
      }
      if ((projection_matrix.size()==0 && vm["gaussian_chunk_dimensionality"].as<index_t>()<0)
          || projection_matrix.size()==1) {
        // low dimensional data
        fl::logger->Message()<<"Treating data as low dimensional gaussian projection table fits in memory";
        boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> projector_table;
        if (projection_matrix.size()==0) {
          // generate the gaussian matrix
          ws_->Attach(vm.count("projection_matrix_out")>0?
              vm["projection_matrix_out"].as<std::string>():ws_->GiveTempVarName(),
              std::vector<index_t>(1, dimensionality),
              std::vector<index_t>(),
              vm["projection_rank"].as<int32>(), 
              &projector_table);
          typename WorkSpaceType::MatrixTable_t::Point_t point;
          fl::logger->Message()<<"Generating a gaussian random matrix"
              <<std::endl;
          for(index_t i=0; i<projector_table->n_entries(); ++i) {
            projector_table->get(i, &point); 
            for(index_t j=0; j<point.size(); ++j) {
              point.set(j, fl::math::RandomNormal());
            }
          }
          ws_->Purge(projector_table->filename());
          ws_->Detach(projector_table->filename());
          fl::logger->Message()<<"Finished generating gaussian projection table";
        } else {
          // load the gaussian matrix
          fl::logger->Message()<<"using gaussian table from ("<<projection_matrix[0]<<")";
          ws_->Attach(projection_matrix[0], &projector_table);
        }
        // projecting the input
        for(size_t i=0; i<references.size(); ++i) {
          boost::shared_ptr<TableType> references_table;
          fl::logger->Message()<<"Projecting ("<<references[i]<<") table";
          ws_->Attach(references[i], &references_table);
          boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> projected_table;
          ws_->Attach(projected_names[i],
              std::vector<index_t>(1, vm["projection_rank"].as<int32>()),
              std::vector<index_t>(),
              references_table->n_entries(),
              &projected_table);
          fl::table::Mul<fl::la::NoTrans, fl::la::Trans>(*references_table, 
              *projector_table, projected_table.get());
          ws_->Purge(references[i]);
          ws_->Detach(references[i]);
          ws_->Purge(projected_names[i]);
          ws_->Detach(projected_names[i]);
        }
        fl::logger->Message()<<"Finished projecting the tables";
      } else {
        fl::logger->Message()<<"Treating data as high dimensional, gaussian projection table does not fit in memory "
          "it is stored in multiple volumes";
        // high dimensional data
        if (projection_matrix.size()>1 && 
            vm["gaussian_chunk_dimensionality"].as<index_t>()<0 ) {
          // Using tables imported from disk
          fl::logger->Message()<<"Using tables imported from disk";
          std::vector<std::string> projection_names=fl::ws::GetFileSequence("projection_matrix",vm);
          fl::logger->Message()<<"Starting projection";
          fl::table::Mul<fl::la::NoTrans, fl::la::Trans>::MUL<
            WorkSpaceType,
            TableType,
            typename WorkSpaceType::MatrixTable_t, 
            typename WorkSpaceType::MatrixTable_t>(
                ws_,
                references,
                projection_names,
                &projected_names);
          fl::logger->Message()<<"Finished projection";
        } else {
          if (projection_matrix.size()==0 &&
              vm["gaussian_chunk_dimensionality"].as<index_t>()>0) {
            // generate the gaussian matrix
            fl::logger->Message()<<"Generating multiple volumes of gaussian tables";
            index_t chunk_size=vm["gaussian_chunk_dimensionality"].as<index_t>();
            index_t dimensionality=0;
            ws_->GetTableInfo(references[0], NULL, &dimensionality, NULL, NULL);
            index_t n_gaussian_matrices=(dimensionality % chunk_size==0)?
            (dimensionality / chunk_size):(dimensionality/chunk_size)+1;
            std::vector<std::string> projection_names;
            std::string prefix="";
            if (vm.count("projection_matrix_prefix")!=0) {
              prefix=vm["projection_matrix_prefix"].as<std::string>();
              if (vm.count("projection_matrix_num_out")==0) {
                fl::logger->Die()<<"since you have set --projection_matrix_prefix "
                  "you should also set projection_matrix_num_out";
              } else {
                if (vm["projection_matrix_num_out"].as<int32>()!=n_gaussian_matrices) {
                  fl::logger->Die()<<"--projection_matrix_num_out is set as ("
                    <<vm["projection_matrix_num_out"].as<int32>()
                    <<") which is different from what it is expected for the "
                    <<"--gaussian_chunk_dimensionality ("
                    <<vm["gaussian_chunk_dimensionality"].as<index_t>() 
                    <<"), it should be ("<<n_gaussian_matrices<<")";
                }
              }
            } else {
              prefix=ws_->GiveTempVarName()+"_";
            }
            for(index_t i=0; i<n_gaussian_matrices; ++i) {
              std::string name=prefix+boost::lexical_cast<std::string>(i);
              fl::logger->Message()<<"Generating table ("<<name<<")";
              projection_names.push_back(name);
              boost::shared_ptr<typename WorkSpaceType::MatrixTable_t> projector_table;
              ws_->Attach(name, 
                  std::vector<index_t>(1, chunk_size),
                  std::vector<index_t>(), 
                  vm["projection_rank"].as<int32>(),
                  &projector_table);
              typename WorkSpaceType::MatrixTable_t::Point_t point;
              for(index_t i=0; i<projector_table->n_entries(); ++i) {
                   projector_table->get(i, &point); 
                for(index_t j=0; j<point.size(); ++j) {
                  point.set(j, fl::math::RandomNormal());
                }
              }
              ws_->Purge(name);
              ws_->Detach(name);
            }
            fl::logger->Message()<<"Starting projection";
            fl::table::Mul<fl::la::NoTrans, fl::la::Trans>::MUL<
            WorkSpaceType,
            TableType,
            typename WorkSpaceType::MatrixTable_t, 
            typename WorkSpaceType::MatrixTable_t>(
                ws_,
                references,
                projection_names,
                &projected_names);         
            fl::logger->Message()<<"Finished projection";
          }
        }
      }    
    } else {
      if (vm["projection_type"].as<std::string>()=="guassian_dynamic") {
      } else {
        if (vm["projection_type"].as<std::string>()=="sparse") {
        
        }
      }
    }
  }

  template<typename WorkSpaceType>
  int RandomProjections<boost::mpl::void_>::Run(
      WorkSpaceType *ws,
      const std::vector<std::string> &args) {

    bool found=false;
    std::string references_in;
    for(size_t i=0; i<args.size(); ++i) {
      if (fl::StringStartsWith(args[i],"--references_prefix_in=")) {
        found=true;
        std::vector<std::string> tokens=fl::SplitString(args[i], "=");
        if (tokens.size()!=2) {
          fl::logger->Die()<<"Something is wrong with the --references_in flag";
        }
        references_in=ws->GiveFilenameFromSequence(tokens[1], 0);
        break;
      }
      if (fl::StringStartsWith(args[i],"--references_in=")) {
        found=true;
        std::vector<std::string> tokens=fl::SplitString(args[i], "=");
        if (tokens.size()!=2) {
          fl::logger->Die()<<"Something is wrong with the --references_in flag";
        }
        std::vector<std::string> filenames=fl::SplitString(tokens[1], ":,"); 
        references_in=filenames[0];
        break;
      }
    }

    if (found==false) {
      Core<WorkSpaceType> core(ws, args);
      typename WorkSpaceType::DefaultTable_t t;
      core(t);
      return 1;
    }

    Core<WorkSpaceType> core(ws, args);
    fl::ws::BasedOnTableRun(ws, references_in, core);
    return 0;
  }
  template<typename WorkSpaceType>
  RandomProjections<boost::mpl::void_>::Core<WorkSpaceType>::Core(
     WorkSpaceType *ws, const std::vector<std::string> &args) :
   ws_(ws), args_(args)  {}



}}
#endif

