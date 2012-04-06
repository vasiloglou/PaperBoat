import os
import os.path

if os.path.exists("test_results")==True:
  os.remove("test_results")
version="../0.6.3/"
os.system("python allkn.py --version_dir="+version)
os.system("python kde.py --version_dir="+version)
os.system("python kmeans.py --version_dir="+version)
os.system("python lasso.py --version_dir="+version)
os.system("python nmf.py --version_dir="+version)
os.system("python npr.py --version_dir="+version)
os.system("python svd.py --version_dir="+version)
os.system("python svm.py --version_dir="+version)

fin=open("test_results", "r")
logs=fin.read()
ind=logs.find("FAILED");

