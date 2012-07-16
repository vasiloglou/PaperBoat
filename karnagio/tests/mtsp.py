import os
import os.path
import threading
import test_suite
import optparse

parser = optparse.OptionParser();

parser.add_option("--version_dir", action="store", type="string", 
    dest="version_dir", default="../branches/0.6.4/", 
    help="the version directory to build/use the binaries")
parser.add_option("--logfile", action="store", type="string", 
    dest="logfile", default="test_results",
    help="The file to append the results")
(options, args)=parser.parse_args();

# open log file for appending output
fout=open(options.logfile, "a")

version_dir=options.version_dir
build_mode=["config-debug"]
bin_dir={"debug":(version_dir+"debug.build/bin"), \
    "release":(version_dir+"release.build/bin")}
targets="mtsp"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "mtsp", fout)

# Start the tests
for directory in bin_dir.values():
  mtsp1=directory+"/mtsp --references_augmented_data_in="+       \
      dataset_dir+"/mtsp/augmented.pb "     +       \
      "--queries_augmented_data_in="+                           \
      dataset_dir+"/climate.com/data/augmented_prediction.pb "+ \
      " --svd:svd_rank=2 "+                                     \
      " --svd:algorithm=covariance "+                           \
      " --timestamp_attribute=3 "+                              \
      " --references_in="+ dataset_dir=+"/mtsp/training.pb "+ \
      " --window=10 "+                                          \
      " --summary=svd "+                                        \
      " --time_lag=3 "+                                         \
      " --run_mode=train "+                                     \
      " --fixed_bw=0:10,10:1,2:2,3:2,3:5,4:1,5:1,6:1"

  os.system(mtsp + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, mtsp1, "SUCCESS"
  else:
    print >> fout, mtsp1, "FAILED"
    print ams1, "FAILED"
  os.remove("temp")
  if os.path.exists("clusters")==True:
    os.remove("clusters")
  if os.path.exists("memberships")==True:  
    os.remove("memberships")
 
print >> fout, "[mtsp] Test finished"
fout.close()
t.cancel()
