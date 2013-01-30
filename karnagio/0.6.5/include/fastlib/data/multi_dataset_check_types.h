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
#ifndef FL_LITE_FASTLIB_DATA_MULTIDATASET_CHECK_TYPES_H_
#define FL_LITE_FASTLIB_DATA_MULTIDATASET_CHECK_TYPES_H_
struct CheckTypes {
  CheckTypes(std::vector<std::string> *type_vec,
             index_t *counter) {
    type_vec_ = type_vec;
    counter_ = counter;
  }
  template<typename T>
  void operator()(T) {
    // we check only on the first character since some compilers
    // print only the first characeter of the string returned by Typename<T>.Name()
    if (Typename<T>::Name() !=
        (*type_vec_)[*counter_]) {
      std::ostringstream s1;
      s1 << "[DATA TYPE ERROR] Type mismatch the definition in the file "
      << "is of type "
      << (*type_vec_)[*counter_].c_str()
      << " while the code is designed for "
      << "type "
      << Typename<T>::Name().c_str()
      <<". You have probably specified the --point flag incorrectly, or the precision "
        "of your data is not supported for this algorithm. Rerun your program with the "
        "--help flag and check the available options for --point";
      throw fl::TypeException(s1.str());
    }
    (*counter_)++;
  }
private:
  std::vector<std::string> *type_vec_;
  index_t *counter_;
};

#endif
