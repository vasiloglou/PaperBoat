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

#include "mlpack/allkn/allkn_dev.h"
#include "mlpack/allkn/allkn_computations_dev.h"
#include "fastlib/table/branch_on_table.h"
#include "allkn_test.h"

template class fl::ml::AllKN<TestAllKN<0, false, true, false, false, 1, 0>::AllKNArgs<1> >;
template void fl::ml::AllKN<TestAllKN<0, false, true, false, false, 1, 0>::AllKNArgs<1> >::
    ComputeNeighbors<fl::math::LMetric<2>, 
    int, 
    std::vector<double>, 
    std::vector<int> >(std::string const&, 
    fl::math::LMetric<2> const&, int, std::vector<double>*, 
    std::vector<int>*);
