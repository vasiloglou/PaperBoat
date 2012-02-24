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

#include "fastlib/base/logger.h"

namespace fl {

boost::mutex logger_mutex;

Exception::Exception() : message_("something bad happened at Ismion") {

};

Exception::~Exception() throw(){

}

Exception::Exception(const std::string &message) : message_(message){
}

const char* Exception::what() const {
  return message_.c_str();
}

TypeException::TypeException() {

};

TypeException::TypeException(const std::string &message) : fl::Exception(message){
}

TypeException::~TypeException() throw (){

};

Logger::Stream::Stream(std::ostream &s, boost::mutex &m) : mutex_(m), s_(s), stream_call_(0) {
  message_.reset(new std::ostringstream());
}

Logger::Stream::~Stream() {
  mutex_.lock();
  if (stream_call_==0) {
    s_<<message_->str()<<"\n";
    s_.flush();
  }
  mutex_.unlock();
}

Logger::Stream::Stream(const Stream &other) : mutex_(other.mutex_), s_(other.s_), 
  message_(other.message_), stream_call_(other.stream_call_) {
  stream_call_++; 
}

Logger::Stream Logger::Stream::operator<<(std::ostream& ( *pf )(std::ostream&)) {
 return *this;
}

void Logger::Stream::Reset() {
  stream_call_=0;
}

Logger::DummyStream::DummyStream(std::ostream &s, boost::mutex &m) :
  Stream(s, m) {
}

Logger::DummyStream::~DummyStream() {
}

Logger::DummyStream::DummyStream(const DummyStream &other) : Stream(other)  {
}

Logger::DummyStream Logger::DummyStream::operator<<(std::ostream& ( *pf )(std::ostream&)) {
 return *this;
}


Logger::Death::Death(std::ostream &stream, boost::mutex &mut) : Stream(stream, mut) {
}

Logger::Death::Death(const Death &other) : Stream(other) {

}

Logger::Death::~Death() {
  if (stream_call_==0) {
    // we need to do this so that the ~Stream destructor
    // will not be called
    stream_call_=-1;
    mutex_.lock();
    this->s_ << message_->str()<<"\n";
    this->s_.flush();
    mutex_.unlock();
#if (defined WIN32 || defined WIN64) && defined _DEBUG
    exit(0);
#else
    throw boost::enable_current_exception(fl::Exception(message_->str()));
  }
#endif
}

Logger::ScopedPrefix::ScopedPrefix(const std::string &prefix) {
  fl::logger->AddPrefix(boost::this_thread::get_id(), prefix);
}

Logger::ScopedPrefix::~ScopedPrefix() {
  fl::logger->PopPrefix(boost::this_thread::get_id());
}

Logger::Logger() : file_(""), stream_(&std::cout) {
}

Logger::~Logger() {
  if (file_ != "") {
    delete stream_;
  }
}

Logger::Death Logger::Die() {
  Death death(*stream_, mutex_);
  death << "\n";
  PrintMetaInfo(death);
  death<<"[TERMINATING] ";
  return death;
}

void Logger::Init(const std::string &file) {
  mutex_.lock();
  if (file_ != "") {
    delete stream_;
  }
  file_ = file;
  if (file_ != "") {
    stream_ = new std::ofstream(file.c_str(), std::ios_base::out);
  }
  else {
    stream_ = &std::cout;
  }
  mutex_.unlock();
}

void Logger::Init(std::ostream *stream) {
  mutex_.lock();
  if (file_ != "") {
    delete stream_;
  }
  file_ = "";
  stream_ = stream;
  mutex_.unlock();
}

void Logger::AddPrefix(const boost::thread::id &id, 
    const std::string &prefix) {
  logger_mutex.lock();
  if (prefixes_.count(id)==0) {
    prefixes_[id]=std::vector<std::string>();
  }
  prefixes_[id].push_back(prefix);
  logger_mutex.unlock();
}

void Logger::PopPrefix(const boost::thread::id &id) {
  logger_mutex.lock();
  if (prefixes_.count(id)!=0) {
    if (prefixes_[id].size()>0) {
      prefixes_[id].pop_back();
    }
  }
  logger_mutex.unlock();
}

void Logger::SuspendLogging() {
  logger_mutex.lock();
  nolog_threads_.insert(boost::this_thread::get_id());
  logger_mutex.unlock();
}


void Logger::ResumeLogging() {
  logger_mutex.lock();
  nolog_threads_.erase(boost::this_thread::get_id());
  logger_mutex.unlock();
}

Logger::Stream DebugLogger::Debug() {
  bool nolog=false;
  logger_mutex.lock();
  nolog=(nolog_threads_.find(boost::this_thread::get_id())!=nolog_threads_.end());
  logger_mutex.unlock();
  std::ostream &local_stream=nolog==true?dummy_:*stream_;
  Stream stream(local_stream, mutex_);
  
  std::string message("[DEBUG]");
  PrintMetaInfo(stream);
  stream<<message;
  stream.Reset();
  return stream;
  
}
Logger::Stream DebugLogger::Message() {
  bool nolog=false;
  logger_mutex.lock();
  nolog=(nolog_threads_.find(boost::this_thread::get_id())!=nolog_threads_.end());
  logger_mutex.unlock();
  std::ostream &local_stream=nolog==true?dummy_:*stream_;
  Stream stream(local_stream, mutex_);
  std::string message("[MESSAGE]");
  PrintMetaInfo(stream);
  stream<<message;
  stream.Reset();
  return stream;
}
Logger::Stream DebugLogger::Warning() {
  bool nolog=false;
  logger_mutex.lock();
  nolog=(nolog_threads_.find(boost::this_thread::get_id())!=nolog_threads_.end());
  logger_mutex.unlock();
  std::ostream &local_stream=nolog==true?dummy_:*stream_;
  Stream stream(local_stream, mutex_);
  std::string message("[WARNING]");
  PrintMetaInfo(stream);
  stream<<message;
  stream.Reset();
  return stream; 
}

DebugLogger::~DebugLogger() {
  mutex_.lock();
  *stream_ << std::endl;
  if (file_ != "") {
    delete stream_;
  }
  mutex_.unlock();
}

Logger::Stream VerboseLogger::Debug() {
  return Logger::DummyStream(dummy_, mutex_);
}
Logger::Stream VerboseLogger::Message() {
  bool nolog=false;
  logger_mutex.lock();
  nolog=(nolog_threads_.find(boost::this_thread::get_id())!=nolog_threads_.end());
  logger_mutex.unlock();
  std::ostream &local_stream=nolog==true?dummy_:*stream_;
  Stream stream(local_stream, mutex_);
  std::string message("[MESSAGE]");
  PrintMetaInfo(stream);
  stream<<message;
  stream.Reset();
  return stream; 
}
Logger::Stream VerboseLogger::Warning() {
  bool nolog=false;
  logger_mutex.lock();
  nolog=(nolog_threads_.find(boost::this_thread::get_id())!=nolog_threads_.end());
  logger_mutex.unlock();
  std::ostream &local_stream=nolog==true?dummy_:*stream_;
  Stream stream(local_stream, mutex_);
  std::string message("[WARNING]");
  PrintMetaInfo(stream);
  stream<<message;
  stream.Reset();
  return stream;
}

VerboseLogger::~VerboseLogger() {
  mutex_.lock();
  *stream_ << std::endl;
  if (file_ != "") {
    delete stream_;
  }
  mutex_.unlock();
}

Logger::Stream WarningLogger::Debug() {
  return Logger::Stream(dummy_, mutex_);
}
Logger::Stream  WarningLogger::Message() {
  return Logger::Stream(dummy_, mutex_);
}
Logger::Stream WarningLogger::Warning() {
  bool nolog=false;
  logger_mutex.lock();
  nolog=(nolog_threads_.find(boost::this_thread::get_id())!=nolog_threads_.end());
  logger_mutex.unlock();
  std::ostream &local_stream=nolog==true?dummy_:*stream_;
  Stream stream(local_stream, mutex_);
  std::string message("[WARNING]");
  PrintMetaInfo(stream);
  stream<<message;
  stream.Reset();
  return stream;
}

WarningLogger::~WarningLogger() {
  mutex_.lock();
  *stream_ << std::endl;
  if (file_ != "") {
    delete stream_;
  }
  mutex_.unlock();
}

Logger::Stream SilentLogger::Debug() {
  return Logger::Stream(dummy_, mutex_);
}
Logger::Stream SilentLogger::Message() {
  return Logger::Stream(dummy_, mutex_);
}
Logger::Stream SilentLogger::Warning() {
  return Logger::Stream(dummy_, mutex_);
}

SilentLogger::~SilentLogger() {
  if (file_ != "") {
    delete stream_;
  }
}

boost::scoped_ptr<Logger> logger(new SilentLogger());
boost::exception_ptr global_exception=boost::exception_ptr();
boost::shared_ptr<boost::mutex> global_exception_mutex(new boost::mutex());

void Logger::SetLogger(const std::string &logger_type) {
  boost::mutex::scoped_lock lock(logger_mutex);
  if (logger_type == "debug") {
    logger.reset(new DebugLogger());
    return;
  }
  if (logger_type == "verbose") {
    logger.reset(new VerboseLogger());
    return;
  }
  if (logger_type == "warning") {
    logger.reset(new WarningLogger());
    return;
  }
  if (logger_type == "silent") {
    logger.reset(new SilentLogger());
    return;
  }
  logger->Die() << "you attempted to initialize logger with an unknown class "
  << logger_type << "\n";

}

void Logger::PrintMetaInfo(Stream &stream) {
  const boost::thread::id id=boost::this_thread::get_id();
  stream<<"[T"<<id<<"]";
  logger_mutex.lock();
  if (prefixes_.count(id)!=0) {
    for(std::vector<std::string>::iterator it=prefixes_[id].begin(); 
        it!=prefixes_[id].end();
        ++it) {
      stream<<"["<<*it<<"]";
    }
  }
  logger_mutex.unlock();
}
};

