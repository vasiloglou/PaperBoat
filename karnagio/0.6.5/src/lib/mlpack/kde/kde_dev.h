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
#ifndef FL_LITE_MLPACK_KDE_KDE_DEV_H
#define FL_LITE_MLPACK_KDE_KDE_DEV_H

#include "mlpack/kde/kde.h"
#include "mlpack/kde/kde_defs.h"

template<typename TemplateMap>
typename fl::ml::Kde<TemplateMap>::Table_t *
fl::ml::Kde<TemplateMap>::query_table() {
  return query_table_;
}

template<typename TemplateMap>
typename fl::ml::Kde<TemplateMap>::Table_t *
fl::ml::Kde<TemplateMap>::reference_table() {
  return reference_table_;
}

template<typename TemplateMap>
typename fl::ml::Kde<TemplateMap>::Global_t &
fl::ml::Kde<TemplateMap>::global() {
  return global_;
}

template<typename TemplateMap>
bool fl::ml::Kde<TemplateMap>::is_monochromatic() const {
  return is_monochromatic_;
}

template<typename TemplateMap>
void fl::ml::Kde<TemplateMap>::Init(
  typename fl::ml::Kde<TemplateMap>::Table_t *reference_table,
  typename fl::ml::Kde<TemplateMap>::Table_t *query_table,
  double bandwidth_in,
  double relative_error_in,
  double probability_in) {

  if (reference_table == NULL) {
    fl::logger->Die()<<"Reference table cannot be NULL";
  }
  reference_table_ = reference_table;
  if (query_table == NULL) {
    is_monochromatic_ = true;
    query_table_ = reference_table;
  }
  else {
    is_monochromatic_ = false;
    query_table_ = query_table;
  }

  // Declare the global constants.
  global_.Init(reference_table_, query_table_, NULL, bandwidth_in, is_monochromatic_,
               relative_error_in, probability_in);
}

template<typename TemplateMap>
void fl::ml::Kde<TemplateMap>::set_bandwidth(double bandwidth_in) {
  global_.set_bandwidth(bandwidth_in);
}


#endif
