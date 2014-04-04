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

#ifndef FL_LITE_FASTLIB_BASE_LOGGER_H_
#define FL_LITE_FASTLIB_BASE_LOGGER_H_
#include <string>
#include <map>
#include <set>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <exception>
#include "null_stream.h"
#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/thread/thread.hpp"
#include "boost/exception/all.hpp"
#include "boost/exception_ptr.hpp"
#include "boost/exception/exception.hpp"


// Macro definitions

#define FL_SCOPED_LOG(tag) fl::Logger::ScopedPrefix fl_scoped_log_var_##tag(#tag)
#define FL_LOG_MESSAGE() fl::logger->Message()
#define FL_LOG_DEBUG()   fl::logger->Debug()
#define FL_LOG_WARNING() fl::logger->Warning()

namespace fl {
  class Exception : virtual public boost::exception {
    public:
      Exception();
      virtual ~Exception() throw() ;
      Exception(const std::string &message);
      const char *what() const;
    protected:
      std::string message_;
  };

  class TypeException :  public Exception {
    public:
      TypeException();
      TypeException(const std::string &message);
      virtual ~TypeException() throw();
  };

class Logger {
  public:
    class Stream {
      public:
        Stream(std::ostream &s, boost::mutex &mut);
        virtual ~Stream();
        Stream(const Stream &other);
        template<typename T>
        Stream operator<<(const T &val) {
          *message_<<val;
          return *this;
        }
        Stream operator<<(std::ostream& ( *pf )(std::ostream&));
        void Reset();
      protected:
        boost::mutex &mutex_;
        std::ostream &s_;
        boost::shared_ptr<std::ostringstream> message_;
        int stream_call_;
    };
 
    class DummyStream : public Stream {
      public:
        DummyStream(std::ostream &s, boost::mutex &mut);
        DummyStream(const DummyStream &other);
        ~DummyStream();
        template<typename T>
        DummyStream operator<<(const T &val) {
           return *this;
        }
        DummyStream operator<<(std::ostream& ( *pf )(std::ostream&));
    };

    class Death : public Stream {
      public:
        Death(std::ostream &s, boost::mutex &mut);
        Death(const Death &other);
        ~Death() ;
    };

    class ScopedPrefix {
      public:
        ScopedPrefix(const std::string &prefix);
        ~ScopedPrefix();
    };

    Logger();
    virtual ~Logger();
    virtual Stream Debug() = 0;
    virtual Stream Message() = 0;
    virtual Stream Warning() = 0;
    Death Die();
    void Init(const std::string &file);
    void Init(std::ostream *stream);
    void AddPrefix(const boost::thread::id &id, const std::string &prefix);
    void PopPrefix(const boost::thread::id &id);
    void SuspendLogging();
    void ResumeLogging();
    static void SetLogger(const std::string &logger_type);
     

  protected:
    boost::mutex mutex_;
    std::string file_;
    std::ostream  *stream_;
    fl::onullstream  dummy_;
    std::map<boost::thread::id, std::vector<std::string> > prefixes_;  
    std::set<boost::thread::id> nolog_threads_;
    void PrintMetaInfo(Stream &stream); 
};

class DebugLogger : public Logger {
  public:
    virtual Stream Debug();
    virtual Stream Message();
    virtual Stream Warning();
    virtual ~DebugLogger();
};

class VerboseLogger : public Logger {
  public:
    virtual Stream Debug();
    virtual Stream Message();
    virtual Stream Warning();
    virtual ~VerboseLogger();
};


class WarningLogger : public Logger {
  public:
    virtual Stream Debug();
    virtual Stream Message();
    virtual Stream Warning();
    virtual ~WarningLogger();
};

class SilentLogger : public Logger {
  public:
    virtual Stream Debug();
    virtual Stream Message();
    virtual Stream Warning();
    virtual ~SilentLogger();
};


extern boost::scoped_ptr<Logger> logger;
extern boost::exception_ptr global_exception;
extern boost::shared_ptr<boost::mutex> global_exception_mutex;
}
#endif
