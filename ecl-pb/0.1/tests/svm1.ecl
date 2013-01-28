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

   y:=DATASET(
     [{0,0,-1,1},
      {1,0,1,1},
      {2,0,-1,1},
      {3,2,1,1},
      {4,2,-1,1}], PB.Types.Int32LDatum);

  PB.Definitions();  // All the paperboat calls start with this initializer
  
  z:=PB.Karnagio(x,,y).Svm(
     ' --references_in=dense$double$0'
     +' --labels_in=dense$int32$1'
     +' --kernel=gaussian'  // we currently support only gaussian
     +' --iterations=10'    // it is important to set the number of iterations
     +' --regularization=3' // the typical L2 regularization for SVM
     +' --bandwidth=1'      // Your choice of the bandwidth
     +' --bandwidth_overload_factor=3' // Here is the critical part. This
                                       // parameter is our approximation factor
                                       // by increasing it you increase the
                                       // accuracy, but you make it slower
     +' --support_vectors_out=dense$double$3' // the support vectors the storage
                                              // and precision should match the 
                                              // ones of references_in
     +' --alphas_out=dense$double$4'
  );
  z.call;
  OUTPUT(z.real_result);
  
  PB.UnDefinitions();


