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
targets="lasso"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "lasso", fout)

# Start the tests
for directory in bin_dir.values():
  lasso1=directory+"/lasso --references_in="+       \
      dataset_dir+"/random/random_1kx6.txt "+       \
      " --prediction_column=0"                      \
      +" --regularization=2"                        \
      +" --iterations=100"                          \
      +" --coefficients_out=coefficients"           \

  os.system(lasso1 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, lasso1, "SUCCESS"
  else:
    print >> fout, lasso1, "FAILED"
    print lasso1, "FAILED"

  os.remove("temp")
  if os.path.exists("coefficients")==True:
    os.remove("coefficients")
  else:
    print >>dout, lasso1, "(coefficients) FAILED"
 
