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
#include "fastlib/base/base.h"
#include "fastlib/dense/cpplapack.h"

fl::dense::CppLapack<float>::CppLapack() {
  float fake_matrix[64];
  float fake_workspace;
  float fake_vector;
  f77_integer fake_pivots;
  f77_integer fake_info;

  /* TODO: This may want to be ilaenv */
  this->getri(1, (float *)fake_matrix, 1, &fake_pivots, &fake_workspace,
              -1, &fake_info);
  this->getri_block_size = int(fake_workspace);

  this->geqrf(1, 1, (float *)fake_matrix, 1, &fake_vector, &fake_workspace, -1,
              &fake_info);
  this->geqrf_block_size = int(fake_workspace);

  this->orgqr(1, 1, 1, (float *)fake_matrix, 1, &fake_vector, &fake_workspace, -1,
              &fake_info);
  this->orgqr_block_size = int(fake_workspace);

  this->geqrf_dorgqr_block_size =
    std::max(this->geqrf_block_size, this->orgqr_block_size);
}

int fl::dense::CppLapack<float>::getri_block_size = 0;
int fl::dense::CppLapack<float>::geqrf_block_size = 0;
int fl::dense::CppLapack<float>::orgqr_block_size = 0;
int fl::dense::CppLapack<float>::geqrf_dorgqr_block_size = 0;

fl::dense::CppLapack<double>::CppLapack() {
  double fake_matrix[64];
  double fake_workspace;
  double fake_vector;
  f77_integer fake_pivots;
  f77_integer fake_info;

  /* TODO: This may want to be ilaenv */
  this->getri(1, (double *)fake_matrix, 1, &fake_pivots, &fake_workspace,
              -1, &fake_info);
  this->getri_block_size = int(fake_workspace);

  this->geqrf(1, 1, (double *)fake_matrix, 1, &fake_vector, &fake_workspace, -1,
              &fake_info);
  this->geqrf_block_size = int(fake_workspace);

  this->orgqr(1, 1, 1, (double *)fake_matrix, 1, &fake_vector, &fake_workspace, -1,
              &fake_info);
  this->orgqr_block_size = int(fake_workspace);

  this->geqrf_dorgqr_block_size =
    std::max(this->geqrf_block_size, this->orgqr_block_size);
}

int fl::dense::CppLapack<double>::getri_block_size = 0;
int fl::dense::CppLapack<double>::geqrf_block_size = 0;
int fl::dense::CppLapack<double>::orgqr_block_size = 0;
int fl::dense::CppLapack<double>::geqrf_dorgqr_block_size = 0;

// template class fl::dense::CppLapack<float>;
// template class fl::dense::CppLapack<double>;

fl::dense::CppLapack<float> fl::dense::CppLapack<float>::singleton;
fl::dense::CppLapack<double> fl::dense::CppLapack<double>::singleton;
