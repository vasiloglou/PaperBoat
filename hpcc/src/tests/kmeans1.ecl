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
  
  z:=PB.Karnagio(x,,).KMeans(
      '  --references_in=dense$double$0'
      +' --memberships_out=dense$uint32$1' // for every point we get the cluster
                                           // it belongs to
      +' --distortions_out=dense$double$2' // sum of squares for all points
                                           // in the same cluster
      +' --centroids_out=dense$double$3'   
      +' --k_clusters=2'
      +' --n_restarts=5'                   // you can execute kmeans several
                                           // times with different restarts 
      +' --initialization=random'          // this is how you initialize the
                                           // kmeans algorithm. You can use
                                           // kmeans++ as an option. You can
                                           // also predefine the initial
                                           // centroids by using the
                                           // --centroids_in option
      +' --algorithm=naive'                // for small number of clusters use
                                           // the naive option. If k is high
                                           // you should use tree option. 
                                           // Other options are online
                                           // online_tree and online_naive
                                           // The online option always has very
                                           // fast convergence to an acceptable
                                           // solution. The preceding method
                                           // will do the fine tuning
      +' --iterations=100'
      +' --randomize=0'                    // if you use online method it is
                                           // always useful to randomize the
                                           // data point order
      );
  z.call;
  OUTPUT(z.real_result);

  PB.UnDefinitions();

