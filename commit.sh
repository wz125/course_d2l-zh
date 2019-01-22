
#conda config --prepend channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
#conda config --prepend channels http://mirrors.ustc.edu.cn/anaconda/pkgs/free/
#conda env create -f environment.yml
#source activate gluon
#

#env: jupyter
#jupyter notebook --generate-config
#$ ipython
#from notebook.auth import passwd
#passwd()
#Enter password:
#Verify password:
#Out[2]: 'sha1:6f6193fcfbd5:614c4ba185334868fc8bbce2e9890b3ef7d1a79b'  
#jupyter notebook

git add -u ./*
git commit -m "by wangzh  `date`"
git push origin master
