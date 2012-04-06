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
targets="kmeans"
dataset_dir="./datasets"
test_dir=os.getcwd()

# Build targets
test_suite.Build(version_dir, test_dir, targets, fout);

# start the wtachdog timer that will stop the test in case
# something goes wrong with the tests and they deadlock
t=threading.Timer(120, test_suite.ExpireTest, "kmeans", fout)

# Start the tests
for directory in bin_dir.values():
  kmeans1=directory+"/kmeans --references_in="+       \
      dataset_dir+"/3gaussians/3gaussians.csv "+      \
      " --k_clusters=3"                               \
      +" --n_restarts=5"                              \
      +" --memberships_out=memberships"               \
      +" --distortions_out=distortions"               \
      +" --centroids_out=centroids"                   \
      +" --initialization=random"                     \
      +" --algorithm=naive"                           \
      +" --iterations=100"                            \
      +" --randomize=0"

  os.system(kmeans1 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, kmeans1, "SUCCESS"
  else:
    print >> fout, kmeans1, "FAILED"
    print kmeans1, "FAILED"

  os.remove("temp")
  if os.path.exists("memberships")==True:
    os.remove("memberships")
  else:
    print >> kmeans1, "(memberships) FAILED"
  if os.path.exists("distortions")==True:
    os.remove("distortions")
  else:
    print >> kmeans1, "(distortions) FAILED"


  kmeans2=directory+"/kmeans --references_in="+        \
      dataset_dir+"/3gaussians/3gaussians.csv "+       \
      "  --k_min=2"                                    \
      +" --k_max=10"                                   \
      +" --search_method=xmeans"                       \
      +" --n_restarts=5"                               \
      +" --memberships_out=memberships"                \
      +" --distortions_out=distortions"                \
      +" --centroids_out=centroids"                    \
      +" --initialization=random"                      \
      +" --algorithm=tree"                             \
      +" --iterations=100"                             \
      +" --randomize=0"         

  os.system(kmeans2 + " 2>&1 > temp")
  if (test_suite.EvaluateRun("temp", [], []))==True:
    print >> fout, kmeans2, "SUCCESS"
  else:
    print >> fout, kmeans2, "FAILED"
    print kmeans2, "FAILED"
  os.remove("temp")
  if os.path.exists("memberships")==True:
    os.remove("memberships")
  else:
    print >> kmeans1, "(memberships) FAILED"
  if os.path.exists("distortions")==True:
    os.remove("distortions")
  else:
    print >> kmeans1, "(distortions) FAILED"

