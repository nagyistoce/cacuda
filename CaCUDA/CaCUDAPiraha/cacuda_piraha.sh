#!/bin/bash

# Set up shell
set -x                          # Output commands
set -e                          # Abort on errors
have_jre=0

java -version 2> javacheck.out
grep 'java version' javacheck.out > /dev/null 2> /dev/null

if test $? = 0; then
    echo 'BEGIN MESSAGE'
    echo 'You have a java runtime environment installed but not a java compiler'
    echo 'The binary parser will be used'
    echo 'END MESSAGE'
    echo 'BEGIN DEFINE'
    echo "JAVA=`which java`"
    echo 'END DEFINE'
    have_jre=1
else
    echo 'BEGIN ERROR'
    echo 'You will need a java runtime environment to parse the CaCUDA ccl file. Abort !'
    echo 'END ERROR'      
fi

# set locations
    THORN=CaCUDAPiraha
    NAME=cacudapiraha
    CLASSPATH=${SCRATCH_BUILD}/build/${THORN}
    SRCDIR=$(dirname $0)
    JARS="cacuda.jar piraha.jar"


# prepare file structure
    rm -rf ${CLASSPATH}/
    mkdir -p ${CLASSPATH} 2>/dev/null
    
# clean up the dir and copy the parser over
    #set 1>&2
    #sleep 1000
# compile the parser      -a "0" = "`which javac >/dev/null 2>/dev/null; echo $?`" 
    
    if [ -e ${SRCDIR}/src/build.sh  -a 0 -eq `echo 'which javac >/dev/null 2>/dev/null; echo $?' | bash` ]; then 
      pushd ${SRCDIR}/src/ >/dev/null 2>/dev/null
      echo "BUILDING THE PIRAHA LIBRARY!!!!" 1>&2
      ./build.sh 1>&2
      mkdir ${CLASSPATH}/${NAME} 2>/dev/null
      cp $JARS ${CLASSPATH}/${NAME}
      cp dist/${NAME}.tgz ${SRCDIR}/dist
      popd >/dev/null 2>/dev/null
      pushd ${CLASSPATH} >/dev/null 2>/dev/null
    else
      echo "USING PREBUILD PIRAHA LIBRARY!!!!" 1>&2
      pushd ${CLASSPATH} >/dev/null 2>/dev/null
      cp -a ${SRCDIR}/dist/${NAME}.tgz ${CLASSPATH}
      tar -zxf ${CLASSPATH}/${NAME}.tgz > /dev/null 2> /dev/null
    fi

    if [ ! -d ${NAME}/templates ]; then
      echo 'Creating a templates dir ' 1>&2
      mkdir ${NAME}/templates
    fi

# copy the correct templates to compilation path 
    cp -rf ${SRCDIR}/dist/templates/*  ${CLASSPATH}/${NAME}/templates/
    
# now we iterate through all thorns and parse the cakernel.ccl file when we find it.
    cd ${NAME}
    echo 'BEGIN MESSAGE'
    CACUDACCL=`find ${SRCDIR}/../..  -maxdepth 3 -name cakernel.ccl -type f`
    for file in ${CACUDACCL}
    do
        java -cp ./cacuda.jar:./piraha.jar edu.lsu.cct.cacuda.CCLParser ${file} ./templates/deps.ccl        
    done
    cd ..
    popd > /dev/null 2> /dev/null
    echo 'END MESSAGE'

