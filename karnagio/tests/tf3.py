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
targets="ams"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "tf3", fout)

# Start the tests
for directory in bin_dir.values():
  tf3=directory+"/tf3 --references_prefix_in="+          \
      dataset_dir+"/tf3sparse/tensor_sparse_100x50x3_ "+ \
      " --references_num_in=10 "+                        \
      " --method=parafac "+                              \
      " --algorithm=cwopt_sgd_lbfgs "+                   \
      " --lbfgs_max_line_searches=5 "+                   \
      " --a_factor_out=a_fac "+                          \
      " --b_factor_out=b_fac "+                          \
      " --c_factor_out=c_fac "+                          \
      " --lbfgs_iterations=10 "+                         \
      " --rank=5 "+                                      \
      " --lbfgs_rank=3 "+                                \
      " --a_regularization=0 "+                          \
      " --b_regularization=0 "+                          \
      " --c_regularization=0 "+                          \
      " --sgd_step0=0.1 "+                               \
      " --sgd_iterations=1 "+                            \
      " --sgd_epochs=1000 "

  os.system(tf3 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, tf3, "SUCCESS"
  else:
    print >> fout, tf3, "FAILED"
    print tf3, "FAILED"
  os.remove("temp")
  if os.path.exists("clusters")==True:
    os.remove("clusters")
  if os.path.exists("memberships")==True:  
    os.remove("memberships")
 
print >> fout, "[tf3] Test finished"
fout.close()
t.cancel()
