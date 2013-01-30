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

#include <vector>
#include <string>
#include "apps/sentiment/sentiment_defs.h"
#include "mlpack/graph_diffuser/graph_diffuser.h"
#include "fastlib/text/text_defs.h"
#include "fastlib/workspace/workspace_defs.h"
#include "fastlib/table/table_dev.h"
#include "fastlib/data/multi_dataset_dev.h"

int main(int argc, char *argv[]) {
  fl::logger->SetLogger("debug");
  // Convert C input to C++; skip executable name for Boost
  std::vector<std::string> args(argv + 1, argv + argc);
  try {
    // Use a generic workspace model
    fl::ws::WorkSpace ws;
    ws.set_schedule_mode(2);
    ws.set_pool(1);
    fl::app::Sentiment<fl::ws::WorkSpace>::Run(&ws, args);
  } catch (const fl::Exception &exception) {
    return EXIT_FAILURE;
  }
}


