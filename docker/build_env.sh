#!/bin/bash

if [ "${1}" = "-nc" ]; then
    cat auth.txt | xargs -n 2 sh -c 'docker-compose build --no-cache --build-arg ROOT_PASSWD=$0 --build-arg PASSWD=$1'
else
    cat auth.txt | xargs -n 2 sh -c 'docker-compose build --build-arg ROOT_PASSWD=$0 --build-arg PASSWD=$1'
fi
