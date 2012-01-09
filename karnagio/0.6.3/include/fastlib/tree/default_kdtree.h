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
#ifndef FL_LITE_FASTLIB_TREE_DEFAULT_KDTREE_H_
#define FL_LITE_FASTLIB_TREE_DEFAULT_KDTREE_H_

#include "bounds.h"
#include "kdtree.h"
#include "spacetree.h"
#include "statistic.h"

namespace fl {
namespace tree {

struct KdTreeDefaultArgs : public TreeArgs  {
  typedef fl::tree::MidpointKdTree TreeSpecType;
  typedef fl::tree::GenHrectBound<double, double, 2> BoundType;
  typedef boost::mpl::bool_<true> SortPoints;
};
/*
  typedef  boost::mpl::map3<
    boost::mpl::pair<
      fl::tree::TreeArgs::TreeSpecType,
      fl::tree::MidpointKdTree
    >,
    boost::mpl::pair<
      fl::tree::TreeArgs::BoundType,
      fl::tree::GenHrectBound<double, double, 2>
    >,
    boost::mpl::pair<
      fl::tree::TreeArgs::SortPoints,
      boost::mpl::bool_<true>
    >
  >::type KdTreeDefaultOpts;
  */
}
}


#endif

