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
#ifndef FL_LITE_FASTLIB_UTIL_FILTER_INPUT_STREAM_H
#define FL_LITE_FASTLIB_UTIL_FILTER_INPUT_STREAM_H

namespace fl{
namespace util{

template <class Extractor>
class FilteringInputStreambuf : public std::streambuf {
  public:

    FilteringInputStreambuf(std::streambuf* source) {
      mySource = source;
    }

    virtual ~FilteringInputStreambuf() {
      sync();
    }

    virtual int overflow(int) {
      return EOF;  // invalid call for input stream (fail)
    }

    virtual int underflow() {
      int result(EOF);
      if (gptr() < egptr()) {
        result = *gptr();
      }
      else if (mySource != NULL) {
        result = myExtractor(*mySource);
        if (result != EOF) {
          assert(result >= 0 && result <= UCHAR_MAX) ;
          myBuffer = result ;
          setg(&myBuffer , &myBuffer , &myBuffer + 1) ;
        }
      }
      return result ;
    }

    virtual int sync() {
      int result(0) ;
      if (mySource != NULL) {
        if (gptr() < egptr()) {
          result = mySource->sputbackc(*gptr()) ;
          setg(NULL , NULL , NULL) ;
        }
        if (mySource->pubsync() == EOF)
          result = EOF ;
      }
      return result ;
    }

    virtual std::streambuf*  setbuf(char* p , int len) {
      return mySource->pubsetbuf(p, len);
    }


  private:
    std::streambuf*          mySource ;
    Extractor           myExtractor ;
    char                myBuffer ;
};

}}

#endif
