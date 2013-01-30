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
#ifndef PAPERBOAT_FASTLIB_TWITTER_TWITTER_H_
#define PAPERBOAT_FASTLIB_TWITTER_TWITTER_H_

#include <vector>
#include "twitcurl/twitcurl.h"
#include "boost/program_options.hpp"
#include "fastlib/base/base.h"

namespace fl {
  class Twitter {
    public:
      class Tweet {
        public:
          const std::string &text() const ; 
          std::string &text();
          const uint64 &to_user_id() const;
          uint64 &to_user_id();
          const std::string &created_at() const ;
          std::string &created_at();
          const uint64 &from_user_id() const ;
          uint64 &from_user_id();
          const uint64 &id() const ;
          uint64 &id();

        private:
          std::string text_;
          uint64 to_user_id_;
          std::string created_at_;
          uint64 from_user_id_;
          uint64 id_;
      };
      
      class TwitterResult {
        public:
          TwitterResult(const std::string &result) ;          
          const std::string next_page() const ;
          const std::vector<Tweet> tweets() ;
      
        private:
          std::vector<Tweet> tweets_;
          std::string next_page_;
      };

      static void GetStoreSearchResults(
          twitCurl &engine,
          const std::string &query,
          const std::string &store_dir, 
          const std::string &prefix);

      static void GenerateTextFile(const std::string &store_dir, 
          const std::string &text_file); 

      static int Main(int argc, char* argv[]);
  };

}
#endif
