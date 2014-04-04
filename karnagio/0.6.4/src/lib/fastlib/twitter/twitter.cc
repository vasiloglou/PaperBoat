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
#include "boost/filesystem.hpp"
#include <fstream>
#include "fastlib/base/logger.h"
#include "fastlib/base/base.h"
#include "fastlib/twitter/twitter.h"
#include "tinyxml.h"
#include "json_spirit/json_spirit_reader_template.h"

const std::string &fl::Twitter::Tweet::text() const {
  return text_;
}

std::string &fl::Twitter::Tweet::text() {
  return text_;
}

const uint64 &fl::Twitter::Tweet::to_user_id() const {
  return to_user_id_;
}

uint64 &fl::Twitter::Tweet::to_user_id() {
  return to_user_id_;
}

const std::string &fl::Twitter::Tweet::created_at() const {
  return created_at_;
}
 
std::string &fl::Twitter::Tweet::created_at() {
  return created_at_;
}
         
const uint64 &fl::Twitter::Tweet::from_user_id() const {
  return from_user_id_;
}

uint64 &fl::Twitter::Tweet::from_user_id() {
  return from_user_id_;
}

const uint64 &fl::Twitter::Tweet::id() const {
  return id_;
}
      
uint64 &fl::Twitter::Tweet::id() {
  return id_;
}
     
fl::Twitter::TwitterResult::TwitterResult(const std::string &result) {
  json_spirit::mValue value;
  json_spirit::read_string(result, value);
  json_spirit::mObject mobject;
  mobject=value.get_obj();
  next_page_=mobject.find("next_page")->second.get_str();
  const json_spirit::mValue::Array results=mobject["results"].get_array();
  for(json_spirit::mValue::Array::const_iterator it=results.begin();
      it!=results.end(); ++it) {
    tweets_.resize(tweets_.size()+1);
    tweets_.back().text()=it->get_obj().find("text")->second.get_str();
    tweets_.back().to_user_id()=it->get_obj().find("to_user_id")->second.get_uint64();
    tweets_.back().created_at()=it->get_obj().find("created_at")->second.get_str();
    tweets_.back().from_user_id()=it->get_obj().find("from_user_id")->second.get_uint64();
    tweets_.back().id()=it->get_obj().find("id")->second.get_uint64();
  }
}
          
const std::string fl::Twitter::TwitterResult::next_page() const {
  return next_page_;
}
          
const std::vector<fl::Twitter::Tweet> fl::Twitter::TwitterResult::tweets() {
  return tweets_; 
}
      
void fl::Twitter::GetStoreSearchResults(twitCurl &engine,
                           const std::string &query,
                           const std::string &store_dir, 
                           const std::string &prefix) {
  std::string response;
  engine.search(const_cast<std::string&>(query));
  engine.getLastWebResponse(response);
  std::string outErrResp;
  engine.getLastCurlError(outErrResp);
  if (outErrResp.size()>0) {
    fl::logger->Die()<<"Query="<<query<<" => failed."
      <<" error="<<outErrResp;
  }
  std::string filename=store_dir+"/"+prefix
    +boost::lexical_cast<std::string>(0);
  std::ofstream fout(filename.c_str());
  if (fout.fail()) {
    fl::logger->Die() << "Could not open file " << filename.c_str()
         << "   error: " << strerror(errno);
  }
  fout<<response;
  
  json_spirit::mValue value;
  json_spirit::read_string(response, value);
  json_spirit::mObject mobject;
  mobject=value.get_obj();
  std::string next_page=mobject.find("next_page")->second.get_str();
  int32 counter=1;
  while (true) {
    fl::logger->Message()<<"Retrieving page="<<next_page
      <<" for query="<<query<<std::endl;
    engine.searchNext(next_page);
    std::string response;
    engine.getLastWebResponse(response);
    engine.getLastCurlError(outErrResp);
    if (outErrResp.size()>0) {
      fl::logger->Die()<<"Query="<<query<<" => failed."
        <<" error="<<outErrResp;
    }
    std::string filename=store_dir+"/"+prefix
      +boost::lexical_cast<std::string>(counter);
    std::ofstream fout(filename.c_str());
    if (fout.fail()) {
      fl::logger->Die() << "Could not open file " << filename.c_str()
           << "   error: " << strerror(errno);
    }
    fout<<response<<std::endl;
    json_spirit::mValue value;
    json_spirit::read_string(response, value);
    json_spirit::mObject mobject;
    mobject=value.get_obj();
    if (mobject.find("next_page")==mobject.end()) {
      break; 
    }
    next_page=mobject.find("next_page")->second.get_str(); 
    counter++;
  }
}

void fl::Twitter::GenerateTextFile(const std::string &store_dir, 
          const std::string &text_file) {
  
  std::ofstream fout(text_file.c_str());
  if (fout.fail()) {
    fl::logger->Die() << "Could not open file " << text_file.c_str()
         << "   error: " << strerror(errno);
  }
  boost::filesystem::path path(store_dir);
  try {
    if (boost::filesystem::exists(path)) {
      if (boost::filesystem::is_directory(path)==false) {
        fl::logger->Die()<<store_dir<<" is not a directory";
      }
      std::vector<boost::filesystem::path> vec;
      std::copy(boost::filesystem::directory_iterator(path), 
          boost::filesystem::directory_iterator(), 
          back_inserter(vec));
      std::sort(vec.begin(), vec.end());             
      // sort, since directory iteration
      // is not ordered on some file systems
      //   
      for(std::vector<boost::filesystem::path>::const_iterator it(vec.begin()); 
          it != vec.end(); ++it) {
        std::ifstream fin(text_file.c_str());
        if (fin.fail()) {
          fl::logger->Die() << "Could not open file " << it->string()
            << "   error: " << strerror(errno);
        }
        std::string content;
        fin>>content;
        TwitterResult result(content);
        std::vector<Tweet> tweets=result.tweets();
        for(std::vector<Tweet>::iterator it=tweets.begin();
            it!=tweets.end(); ++it) {
          fout<<it->text()<<std::endl;  
        }
      }
    }
  }
  catch (const boost::filesystem::filesystem_error& ex) {
    fl::logger->Die() << ex.what();
  }
}

int fl::Twitter::Main(int argc, char* argv[]) {
  fl::logger->SetLogger("debug");
  // Convert C input to C++; skip executable name for Boost
  std::vector<std::string> args(argv + 1, argv + argc);
  FL_SCOPED_LOG(Twitter);
  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "login",
    boost::program_options::value<std::string>()->default_value("nvasil"),
    "twitter account login "
  )(
    "password",
    boost::program_options::value<std::string>(),
    "tweeter account password"
  )(
    "oauth_consumer_secret",
    boost::program_options::value<std::string>()->default_value(
      "hREqZvckQ1cJ5IdzRZ1tSi0esPbEPO3d1CQm258g"),
    "the tweeter consumer secret"  
  )(
    "oauth_consumer_key",
    boost::program_options::value<std::string>()->default_value(
      "XnAOxCq6DXWpjGIGjLnCQ"),
    "the tweeter consumer key"  
  )(
    "oauth_token_secret",
    boost::program_options::value<std::string>()->default_value(
      "vOnJBYXPNFBAfKPl2UoNifzS8CjqgeLozjJmUfEfw"),
    "the tweeter token secret"  
  )(
    "oauth_token_key",
    boost::program_options::value<std::string>()->default_value(
      "149182719-MFuNd02SmSSkjOYUkOMkUJjlSNa4Ma63rq3TRCh5"),
    "the tweeter token key"  
  )(
    "store_dir", 
    boost::program_options::value<std::string>()->default_value("./"), 
    "directory to store the results from queries"
  )(
    "file_prefix",
    boost::program_options::value<std::string>()->default_value("tweeter_"),
    "prefix for files to be stored" 
  )(
    "query",
    boost::program_options::value<std::string>()->default_value("nyquil"),
    "the twitter query"
  )(
    "task",
    boost::program_options::value<std::string>()->default_value("store"),
    "task to do:\n"
    "  store: runs a query and stores the results\n"
    "  aggregate_text: collects all tweet texts and put it in a single file\n"
  )(
    "text_out",
    boost::program_options::value<std::string>(),
    "the file that contains the tweet texts"  
  );
      
  boost::program_options::variables_map vm;
  boost::program_options::command_line_parser clp(args);
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
    return EXIT_SUCCESS ;
  }
 
  twitCurl engine;
  if (vm.count("login")>0 && vm.count("password")>0) {
    fl::logger->Message()<<"Authorizing with login/password";
    engine.setTwitterUsername(const_cast<std::string&>(vm["login"].as<std::string>())); 
    engine.setTwitterPassword(const_cast<std::string&>(vm["password"].as<std::string>()));
  } else {
    fl::logger->Message()<<"Authorizing with oauth";
    engine.getOAuth().setConsumerSecret(
        vm["oauth_consumer_secret"].as<std::string>());
    engine.getOAuth().setConsumerKey(
        vm["oauth_consumer_key"].as<std::string>()); 
    engine.getOAuth().setOAuthTokenSecret(
        vm["oauth_token_secret"].as<std::string>());
    engine.getOAuth().setOAuthTokenKey(
        vm["oauth_token_key"].as<std::string>());
 
  }

  if (vm["task"].as<std::string>()=="store") {
    std::string response;
    std::string query=vm["query"].as<std::string>();
    const std::string store_dir=vm["store_dir"].as<std::string>();
    const std::string file_prefix=vm["file_prefix"].as<std::string>();
    GetStoreSearchResults(engine,
                          query,
                          store_dir, 
                          file_prefix); 
    return EXIT_SUCCESS;
  }
  if (vm["task"].as<std::string>()=="aggregate_text") {
    std::string store_dir=vm["store_dir"].as<std::string>();
    std::string text_file=vm["text_out"].as<std::string>();
    GenerateTextFile(store_dir, text_file); 
    return EXIT_SUCCESS;
  }
  return EXIT_SUCCESS;
}

