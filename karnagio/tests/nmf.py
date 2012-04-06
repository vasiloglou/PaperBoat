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
bin_dir={"debug":(version_dir+"debug.build/bin")}
targets="nmf"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "nmf", fout)

# Start the tests
for directory in bin_dir.values():
  nmf1=directory+"/nmf --references_in="+           \
      dataset_dir+"/random/random_1kx6.txt "+       \
      " --w_factor_out=w_factor"                    \
      +" --h_factor_out=h_factor"                   \
      +" --k_rank=3"
  os.system(nmf1 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, nmf1, "SUCCESS"
  else:
    print >> fout, nmf1, "FAILED"
    print nmf1, "FAILED"
  os.remove("temp")
  if os.path.exists("w_factor")==True:
    os.remove("w_factor")
  else:
    print >> fout, nmf1, "(w_factor) FAILED"
  if os.path.exists("h_factor")==True:
    os.remove("h_factor")
  else:
    print >> fout, nmf1, "(h_factor) FAILED"

  nmf2=directory+"/nmf --references_in="+           \
      dataset_dir+"/netflix/transformed_8.fl "+     \
      " --w_factor_out=w_factor"                    \
      +" --h_factor_out=h_factor"                   \
      +" --k_rank=3"                                \
      +" --iterations=10"                           \
      +" --epochs=100"                              \
      +" --w_sparsity_factor=0.3"                   \
      +" --h_sparsity_factor=0.0"                   \
      +" --sparse_mode=stoc_lbfgs"
  os.system(nmf2 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, nmf2, "SUCCESS"
  else:
    print >> fout, nmf2, "FAILED"
    print nmf2, "FAILED"

  os.remove("temp")
  if os.path.exists("w_factor")==True:
    os.remove("w_factor")
  else:
    print >> fout, nmf1, "(w_factor) FAILED"
  if os.path.exists("h_factor")==True:
    os.remove("h_factor")
  else:
    print >> fout, nmf1, "(h_factor) FAILED"
