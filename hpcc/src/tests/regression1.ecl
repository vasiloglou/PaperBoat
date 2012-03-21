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
  
  PB.Definitions();  // All the paperboat calls start with this initializer
  
  z:=PB.Karnagio(x,,).Regression(' --references_in=dense$double$0 '
               +' --prediction_index_prefix=0'  // this option specifies which dimension 
                                                // (number) is the dependent
                                                // variable (aka target or
                                                // predicted)
               +' --exclude_bias_term=0'        // if you want to include the
                                                // bias term, set it to 0.
                                                // y=ax+b, where b is the bias
                                                // term
               +' --stepwise=0'                 // set this to 1 if you want to
                                                // do model selection with
                                                // stepwise regression
               +' --vif_pruning=0'              // set this to 1 if you want to
                                                // do model selection with vif
                                                // pruning 
               +' --coeffs_out=dense$double$1'  // the regression coefficients.
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
               +' --r_squared_out=dense$double$2' // the r squared statistic
               +' --sigma_out=dense$double$3'     // the sigma
               +' --t_values_out=dense$double$4'  // the t statistics
               // for more statistics run this query with the --help flag
               );
  z.call;
  OUTPUT(z.real_result);
  

  PB.UnDefinitions();


