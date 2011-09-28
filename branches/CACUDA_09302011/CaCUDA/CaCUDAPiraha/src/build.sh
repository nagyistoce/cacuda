#! /bin/bash

distname="cacudapiraha"

mkdir -p bin
mkdir -p dist/${distname}

# compile cacuda parser and code generator
javac -Xlint:unchecked -d bin src/edu/lsu/cct/cacuda/*.java -cp piraha.jar

if [ $? -ne 0 ] ; then 
  echo "Error while compiling the java code" 1>&2
  exit -1
fi

# make a jar ball
pushd bin > /dev/null
jar cvf ../cacuda.jar `find edu -name \*.class`
popd > /dev/null

# make everything available for distribution
cp -a templates dist/${distname}
cp -a cacuda.jar piraha.jar dist/${distname} 
pushd dist > /dev/null
tar -zcf ${distname}.tgz ${distname} --remove-files
popd > /dev/null

