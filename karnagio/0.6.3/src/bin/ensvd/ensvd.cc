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
#include "mlpack/ensvd/ensvd.h"
#include "fastlib/workspace/workspace_defs.h"
#include "fastlib/workspace/arguments.h"

int main(int argc, char *argv[]) {
  fl::logger->SetLogger("debug");
  // Convert C input to C++; skip executable name for Boost
  std::vector<std::string> args(argv + 1, argv + argc);
  try {
    // Use a generic workspace model
    fl::ws::WorkSpace ws;
    std::vector<std::string> args1=fl::ws::MakeArgsFromPrefix(args, "m");
    boost::program_options::options_description desc("Available options");
    desc.add_options()
    ("help", "Display help")
  ("n_threads", 
   boost::program_options::value<int32>()->default_value(2),
   "Number of cores to be used by the system"
  );


  boost::program_options::variables_map vm;
  boost::program_options::command_line_parser clp(args1);
  clp.style(boost::program_options::command_line_style::default_style
            ^ boost::program_options::command_line_style::allow_guessing);
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
  boost::program_options::notify(vm);
  if (vm.count("help")) {
    std::cout << fl::DISCLAIMER << "\n";
    std::cout << desc << "\n";
    return true;
  }
    ws.set_schedule_mode(1);
    ws.set_pool(vm["n_threads"].as<int32>());
    fl::ml::EnSvd<boost::mpl::void_>::Run(&ws, args);
  } catch (const fl::Exception &exception) {
    return EXIT_FAILURE;
  }
}


