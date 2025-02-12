#!/usr/bin/env bash

#
#  Start API server and workers as host processes. 
#  It assumes:
#  - a python virtual env (version >= 3.11) is setup in <PROJECT_HOME>/.venv
#  - .env files in each workers directory (e.g., src/poc/crewai) are setup with required API keys 
#

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PRJ_HOME_DIR=$(cd ${SCRIPT_DIR}/.. &> /dev/null && pwd)


set -x # echo so that users can understand what is happening

# Function to handle termination of processes on CTRL+C
cleanup() {
    # echo "Terminating background processes..."
    # kill "${api_server_pid}" "${lg_worker_pid}" \
    #      "${crew_worker_pid}" "${pdl_worker_pid}" \
    #      "${pdl_worker_fib_pid}" "${bee_worker_pid}" \
    #      "${ag_worker_pid}"
    echo "Stopping container processes..."
    docker rm -f redis  
    docker rm -f postgres  
    exit
}

# # Trap CTRL+C (SIGINT) and call cleanup function
# trap cleanup SIGINT

# wait until a container with specified name is started
wait_for_container_start() {
    container_name="$1"
    while true; do
        status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null)
        if [ "$status" == "running" ]; then
            echo "Container '$container_name' is running."
            break
        else
            echo "Waiting for container '$container_name' to start..."
            sleep 1
        fi
    done
}

:
: -------------------------------------------------------------------------
: "Start Redis and Postgres"
: 
docker ps  | grep redis > /dev/null
if [[ $? != 0 ]]; then
  echo "starting redis"
  docker run --name redis -p 6379:6379 -d redis
fi  

wait_for_container_start redis

docker ps  | grep postgres > /dev/null
if [[ $? != 0 ]]; then
  echo "starting postgres"
  docker run --name postgres -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d postgres
fi  

wait_for_container_start postgres

set -e # exit on error

