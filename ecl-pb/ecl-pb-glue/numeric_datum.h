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

#ifndef PAPERBOAT_HPCC_SRC_NUMERIC_DATUM_H_
#define PAPERBOAT_HPCC_SRC_NUMERIC_DATUM_H_

namespace fl {namespace hpcc {
  template<typename PrecisionType=double>
  struct NumericDatum {
    protected:
      char *ptr_;
  
    public:
      typedef PrecisionType Value_t;
      NumericDatum(void *ptr) : ptr_((char*)ptr) {}
      void set_ptr(void *ptr) {
        ptr_=static_cast<char*>(ptr);
      }
      uint64 &id() {
        return reinterpret_cast<uint64*>(ptr_)[0];
      }  
      uint32 &number() {
        return reinterpret_cast<uint32*>(ptr_+sizeof(uint64))[0];
      }
      PrecisionType &value() {
        return reinterpret_cast<PrecisionType*>(ptr_
                                    +sizeof(uint64)
                                    +sizeof(uint32))[0];
      }
      static int32 size() {
        return (sizeof(uint64)+sizeof(uint32)+sizeof(PrecisionType));
      }
  };
}}
#endif
