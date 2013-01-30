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

#ifndef FL_LITE_MLPACK_KDE_DUALTREE_TRACE_H
#define FL_LITE_MLPACK_KDE_DUALTREE_TRACE_H

#include <deque>

namespace fl {
namespace ml {

template<typename ArgType>
class DualtreeTrace {

  private:

    std::deque<ArgType> trace_;

  public:

    void push_front(const ArgType &arg_in) {
      trace_.push_front(arg_in);
    }

    void pop_front() {
      trace_.pop_front();
    }

    void push_back(const ArgType &arg_in) {
      trace_.push_back(arg_in);
    }

    void pop_back() {
      trace_.pop_back();
    }

    ArgType &back() {
      return trace_.back();
    }

    const ArgType &back() const {
      return trace_.back();
    }

    ArgType &front() {
      return trace_.front();
    }

    const ArgType &front() const {
      return trace_.front();
    }

    bool empty() const {
      return trace_.empty();
    }

    /** @brief Initialize the dual-tree trace.
     */
    void Init() {
      trace_.resize(0);
    }

};
};
};

#endif
