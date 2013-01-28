  IMPORT PB;  
              
  x:=DATASET(
     [{0,0,1,0},
      {0,1,2,0},
      {0,2,4,0},
      {1,0,3,0},
      {1,1,2,0},
      {1,2,1,0},
      {2,0,2,0},
      {2,1,4,0},
      {2,2,5,0},
      {3,0,3,0},
      {3,1,2,0},
      {3,2,1,0},
      {4,0,3,0},
      {4,1,2,0},
      {4,2,1,0}], PB.Types.Uint8LDatum);

  PB.Definitions();  // All the paperboat calls start with this initializer
  
  z:=PB.Karnagio(,x,).Nmf(
      ' --references_in=sparse$uint8$0'
      +' --w_factor_out=dense$double$1'
      +' --h_factor_out=dense$double$2'
      +' --k_rank=2'
      +' --iterations=100'        // number of iterations to run lbfgs method
      +' --epochs=100'            // number of epochs to run stochastic gradient
      +' --w_sparsity_factor=0.3' // these two parameters control the sparsity
      +' --h_sparsity_factor=0.0'   // of the w and h factors. High sparsity
                                    // has a better generalization error
      +' --sparse_mode=stoc_lbfgs'  // this option will run nmf with stocastic
                                    // gradient descent in the beginning and
                                    // then with lbfgs. You can run it only with 
                                    // stoc or lbfgs
      );
  z.call;
  OUTPUT(z.real_result);
  PB.UnDefinitions();

