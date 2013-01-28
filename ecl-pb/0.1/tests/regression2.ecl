  IMPORT PB;  
              

  x:=DATASET(
     [{0,0,-0.90265,0},
      {0,1,-0.91719,0},
      {0,2,0.74029,0},
      {1,0,-0.079012,0},
      {1,1,-0.79218,0},
      {1,2,2.4023,0},
      {2,0,-0.79442,0},
      {2,1,0.40984,0},
      {2,2,-0.65182,0},
      {3,0,0.4469,0},
      {3,1,0.86663,0},
      {3,2,-0.15092,0},
      {4,0,-0.52346,0},
      {4,1,0.44427,0},
      {4,2,0.33556,0}], PB.Types.RealLDatum);

  G:=DATASET(
    [{0, 0, -2, 1},
     {0, 1, 1, 1},
     {0, 2, 0, 1}], PB.Types.RealLDatum);
  
  PB.Definitions();  // All the paperboat calls start with this initializer
  
  z:=PB.Karnagio(x+G,,).Regression( // Notice that we package all the input
                                    // data as x+G in the Karnagio
                                    // argument for real data
  
               ' --references_in=dense$double$0 '
               +' --prediction_index_prefix=0'  // this option specifies which dimension 
                                                // (number) is the dependent
                                                // variable (aka target or
                                                // predicted)
               +' --exclude_bias_term=1'        // if you want to include the
                                                // bias term, set it to 0.
                                                // y=ax+b, where b is the bias
                                                // term
               +' --ineq_in=dense$double$1'     // the inequality matrix G as
                                                // described below
               +' --ineq_rhs_column=1'          // the column of G that will be
                                                // used as h 
                                                // ONLY ONE OF THE FOLLOWING
                                                // FLAGS CAN BE ON
               +' --ineq_lsi=1'                 // Solving regression with inequality 
                                                // constraints Minimize ||Ex-f|| subject 
                                                // to  Gx>=h

               +' --ineq_nnls=0'                // Solving regression with inequality 
                                                // constraints Minimize ||Ex-f|| subject 
                                                // to x>=0
       
               +' --ineq_ldp=0'                 // Solving regression with inequality constraints
                                                // Minimize ||x|| subject to
                                                // Gx>=h

               +' --coeffs_out=dense$double$2'  // the regression coefficients.
                                                // if you exclude bias term,
                                                // then the index of the
                                                // dependent variable will be zero  
                                                // if you include the bias term,
                                                // the estimated value of the
                                                // bias term will be stored in the
                                                // index of the dependent variable
                                                // To see the difference execute
                                                // this query by toggling the
                                                // --exclude_bias_term option 
               +' --r_squared_out=dense$double$3' // the r squared statistic
               +' --sigma_out=dense$double$4'     // the sigma
               +' --t_values_out=dense$double$5'  // the t statistics
               // for more statistics run this query with the --help flag
               );
  z.call;
  OUTPUT(z.real_result);
  

  PB.UnDefinitions();

 
