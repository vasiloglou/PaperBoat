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

#ifndef PAPER_BOAT_KARNAGIO_INCLUDE_APP_SENTIMENT_SENTIMENT_H_
#define PAPER_BOAT_KARNAGIO_INCLUDE_APP_SENTIMENT_SENTIMENT_H_

#include <vector>
#include <string>
#include "fastlib/text/text.h"

namespace fl { namespace app {

  template<typename WorkSpaceType>
  class Sentiment {
    public:
      static int Main(WorkSpaceType *, const std::vector<std::string> &args);  
    
      static void Run(WorkSpaceType *data,
      const std::vector<std::string> &args);

    private:
      static void PrettyWordSentimentExport(
            WorkSpaceType *ws,
            const std::string &word_sentiment_table_name,
            const std::string &survived_words_filename,
            const std::string &outfilename);

  };
}}
#endif
