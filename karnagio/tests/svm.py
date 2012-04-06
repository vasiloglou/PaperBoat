import os
import os.path
import threading
import test_suite
import optparse

parser = optparse.OptionParser();

parser.add_option("--version_dir", action="store", type="string", 
    dest="version_dir", default="../branches/0.6.3/", 
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
targets="svm"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "svm", fout)

# Start the tests
for directory in bin_dir.values():
  svm1=directory+"/svm --references_in="\
      +dataset_dir+"/uci/adult_train_transformed_stratified_6k.txt "\
      + "--bandwidth=2 "                                            \
      + "--regularization=0.5 "                                     \
      + "--bandwidth_overload_factor=1.2 "                          \
      + "--iterations=1 "                                           \
      + "--support_vectors_out=support_vectors "                    \
      + "--alphas_out=alphas"
  os.system(svm1 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", ["support_vectors", "alphas"], []))==True:
    print >> fout, svm1, "SUCCESS"
  else:
    print >> fout, svm1, "FAILED"
    print svm1, "FAILED"
  os.remove("temp")
  if os.path.exists("support_vectors")==True:
    os.remove("support_vectors")
  else:
    print svm1, "(support_vectors) FAILED"
  if os.path.exists("alphas")==True:
    os.remove("alphas")
  else:
    print svm1, "(alphas) FAILED"

  svm2=directory+"/svm --references_in=" \
      +dataset_dir+"/uci/adult_train_transformed_stratified_6k.txt "  \
      +"  --support_vectors_out=support_vectors "                     \
      +"  --alphas_out=alphas     "                                   \
      +"  --kernel=gaussian       "                                   \
      +"  --bandwidth=2           "                                   \
      +"  --regularization=0.5    "                                   \
      +"  --bandwidth_overload_factor=2" 
  os.system(svm2 + "2>&1 > temp")
  if (test_suite.EvaluateRun("temp", ["support_vectors", "alphas"], []))==True:
    print >> fout, svm2, "SUCCESS"
  else:
    print >> fout, svm2, "FAILED"
    print >> svm2, "FAILED"
  os.remove("temp")
  if os.path.exists("support_vectors")==True:
    os.remove("support_vectors")
  else:
    print svm2, "(support_vectors) FAILED"
  if os.path.exists("alphas")==True:
    os.remove("alphas")
  else:
    print svm2, "(alphas) FAILED"

print >> fout, "[svm] Test finished"
fout.close()
t.cancel()
