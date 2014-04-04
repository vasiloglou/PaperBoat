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
#ifndef PAPERBOAT_FASTLIB_TABLE_POINT_FILTERS_H_
#define PAPERBOAT_FASTLIB_TABLE_POINT_FILTERS_H_

namespace fl {
  namespace table {
    /** This class is used to exclude points during tree traversal. The NullFilter 
     *  does nothing, accepts all the points */
    struct NullFilter {
      template<typename PointType>
      bool FilterOut(const PointType& query, const PointType& reference) {
        return false;
      }
    };
    /** This class is used to exclude points during tree traversal. The TimeFilter 
     *  excludes all the points that have timestamp equal or greater of the query point 
     *  The point metadata<2> is treated as a timestamp 
     */
    struct TimeFilter {
      template<typename PointType>
      bool FilterOut(const PointType& query, const PointType& reference) {
        return query.meta_data().template get<2>() >= reference.meta_data(). template get<2>();
      }
    };
  }
}

#endif
