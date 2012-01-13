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
#ifndef FL_LITE_MLPACK_REGRESSION_GIVENS_ROTATE_H
#define FL_LITE_MLPACK_REGRESSION_GIVENS_ROTATE_H

namespace fl {
namespace ml {

class GivensRotate {

  public:

    /** @brief Applies the Givens rotation row-wise.
     */
    template<typename MatrixType>
    static void ApplyToRow(double cosine_value, double sine_value,
                           int first_row_index, int second_row_index,
                           MatrixType &matrix);

    /** @brief Applies the Givens rotation column-wise.
     */
    template<typename MatrixType>
    static void ApplyToColumn(double cosine_value, double sine_value,
                              int first_column_index, int second_column_index,
                              MatrixType &matrix);

    /** @brief Computes the Givens rotation such that the second
     *         value becomes zero.
     */
    static void Compute(double first, double second,
                        double *magnitude, double *cosine_value,
                        double *sine_value);
};
};
};

#endif
