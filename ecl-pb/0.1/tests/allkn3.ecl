IMPORT PB;


PB.Definitions();

DATASET(PB.Types.RealLDatum) xx := DATASET('3gaussians.thor', PB.Types.RealLDatum, CSV(SEPARATOR(','))); x:=xx(id<=10);
y:=DATASET(
[{0,0,-0.90265,1},
{0,1,-0.91719,1},
{0,2,0.74029,1},
{1,0,-0.079012,1},
{1,1,-0.79218,1},
{1,2,2.4023,1},
{2,0,-0.79442,1},
{2,1,0.40984,1},
{2,2,-0.65182,1},
{3,0,0.4469,1},
{3,1,0.86663,1},
{3,2,-0.15092,1},
{4,0,-0.52346,1},
{4,1,0.44427,1},
{4,2,0.33556,1}], PB.Types.RealLDatum);

w:=DATASET(
[{0,0,1,2},
 {1,0,2,2},
 {2,0,1,2},
 {3,0,2,2},
 {4,0,2,2}
], PB.Types.Int32LDatum);


z:=PB.Karnagio(x+y,,w).Allkn(' --references_in=dense$double$1 '
               +' --queries_in=dense$double$0 '
               +' --k_neighbors=1'
               +' --reference_labels_in=dense$int32$2'
               +' --method=nnclassification'
               +' --distances_out=dense$double$3'
               +' --indices_out=dense$uint32$4'
               +' --labels_out=dense$int32$5');

OUTPUT(z.call);
OUTPUT(z.real_result);
output(z.uint32_result);
OUTPUT(z.int32_result);

PB.UnDefinitions();
