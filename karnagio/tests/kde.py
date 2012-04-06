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

targets="kde"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "kde", fout)

# Start the tests
for directory in bin_dir.values():
  kde1=directory+"/kde --references_in="+            \
       dataset_dir+"/random/random_1kx6.txt "+       \
       " --kernel=gaussian"                          \
       +" --bandwidth=1"                             \
       +" --iterations=2"                            \
       +" --relative_error=0.1"                      \
       +" --algorithm=dual"                          \
       +" --densities_out=densities" 
  os.system(kde1 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, kde1, "SUCCESS"
  else:
    print >> fout, kde1, "FAILED"
    print kde1, "FAILED"
  os.remove("temp")
  if (os.path.exists("densities")==True):
    os.remove("densities")

  kde2=directory+"/kde --references_in="+                     \
       dataset_dir+"/random/random_1kx6.txt "                 \
       +" --queries_in="+dataset_dir+"/random/random_1kx6.txt"\
       +" --kernel=gaussian"                         \
       +" --bandwidth=1"                             \
       +" --relative_error=0.1"                      \
       +" --algorithm=dual"                          \
       +" --densities_out=densities"                 
  os.system(kde2 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, kde2, "SUCCESS"
  else:
    print >> fout, kde2, "FAILED"
    print kde2, "FAILED"

  os.remove("temp")
  if (os.path.exists("densities")==True):
    os.remove("densities")

  kde3=directory+"/kde --references_in="+                    \
       dataset_dir+"/random/random_1kx6.txt "                \
       +" --kernel=gaussian"                         \
       +" --bandwidth_selection=monte_carlo"         \
       +" --relative_error=0.1"                      \
       +" --algorithm=dual"                          \
       +" --densities_out=densities"                 
  os.system(kde3 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, kde3, "SUCCESS"
  else:
    print >> fout, kde3, "FAILED"
    print kde3, "FAILED"

  os.remove("temp")
  if (os.path.exists("densities")==True):
    os.remove("densities")


  kde4=directory+"/kde --references_in="+                      \
       dataset_dir+"/random/random_500x6_a.txt,"               \
       +dataset_dir+"/random/random_500x6_b.txt"               \
       +" --queries_in="+dataset_dir+"/random/random_1kx6.txt" \
       +" --priors=1,10"                                       \
       +" --auc=1"                                             \
       +" --kernel=gaussian"                         \
       +" --relative_error=0.1"                      \
       +" --algorithm=dual"                          \
       +" --densities_out=densities"                 \
       +" --results_out=results"
  os.system(kde1 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, kde4, "SUCCESS"
  else:
    print >> fout, kde4, "FAILED"
    print kde4, "FAILED"

  os.remove("temp")
  if (os.path.exists("densities")==True):
    os.remove("densities") 
  if (os.path.exists("results")==True):
    os.remove("results")

