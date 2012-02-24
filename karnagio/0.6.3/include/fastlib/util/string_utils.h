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
#ifndef FL_LITE_FASTLIB_UTIL_STRING_UTILS_H
#define FL_LITE_FASTLIB_UTIL_STRING_UTILS_H
#include <vector>
#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/predicate.hpp"
#include "boost/algorithm/string/replace.hpp"
namespace fl{

// Splits a string on character supplied. Undefined behaviour is multiple 
// consecutive characters appear in the string.
inline void split_line_on_char(std::string& line, 
                               char split_char, 
                               std::vector<std::string>& results) {
  int start_pos = 0;
  while (1) {
    int space_pos = line.find(split_char, start_pos);
    if (space_pos == std::string::npos) {
      break;
    }
    else {
      results.push_back(line.substr(start_pos, space_pos - start_pos));
      start_pos = space_pos + 1;
    }
  }
  results.push_back(line.substr(start_pos, line.length() - start_pos)); // push last parameter
}

inline std::vector<std::string> SplitString(const std::string &input,
    const std::string &delimeter) {
  std::vector<std::string> tokens;
  boost::algorithm::split(tokens, input, boost::algorithm::is_any_of(delimeter));
  if (tokens.size()==1) {
    if (tokens[0]=="") {
      tokens.clear();
    }
  }
  return tokens;
}

inline bool StringStartsWith(const std::string &input, const std::string &test) {
  return boost::algorithm::starts_with(input, test);
}

inline void StringReplace(std::string *input, const std::string &search,
    const std::string &substitute) {
  boost::algorithm::replace_all(*input, search, substitute);
}

inline std::string StitchStrings(const std::string &a,
                                const std::string &b) {
  std::string c=a+b;
  return c;
}

template<typename T>
inline std::string StitchStrings(const std::string &a,
                                const T &b) {
  std::string c=a+boost::lexical_cast<std::string>(b);
  return c;
}
}

#endif
