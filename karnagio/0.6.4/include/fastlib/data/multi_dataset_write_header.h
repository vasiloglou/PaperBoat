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
#ifndef FL_LITE_FASTLIB_DATA_MULTIDATASET_WRITE_HEADER_H_
#define FL_LITE_FASTLIB_DATA_MULTIDATASET_WRITE_HEADER_H_
struct WriteHeaderOperator {
  WriteHeaderOperator(std::ostream *out,
                      std::vector<index_t> *sizes,
                      index_t *ind,
                      const std::string prefix,
                      std::string delim) {

    out_ = out;
    sizes_ = sizes;
    ind_ = ind;
    prefix_ = prefix;
    delim_ = delim;
  }

  template<typename T>
  void operator()(T) {
    *out_ << prefix_ << Typename<T>::Name() << ":" << sizes_->at(*ind_) << delim_;
    (*ind_)++;
  }

private:
  std::ostream *out_;
  std::vector<index_t> *sizes_;
  index_t *ind_;
  std::string prefix_;
  std::string delim_;
};


#endif
