workDir=$(cd $(dirname $0); pwd)
codeDir=${workDir}/code
buildDir=${codeDir}/SAT/build 
mkdir ${workDir}/ans

logDir=${workDir}/log
mkdir ${logDir}
mkdir "${logDir}/solve"
mkdir "${logDir}/log"
mkdir "${logDir}/weight"



if [[ -d ${buildDir} ]]; then
    rm -rf ${buildDir}
    mkdir ${buildDir}
else
    mkdir ${buildDir}
fi
echo $codeDir
cd ${buildDir} || return 
cmake ../
make
