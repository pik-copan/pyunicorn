#!/bin/sh
set -e

# check cache
if [ ! -d "$HOME/igraph/lib" ]; then
    wget http://igraph.org/nightly/get/c/igraph-$IGRAPH.tar.gz
    tar -xzf igraph-$IGRAPH.tar.gz
    cd igraph-$IGRAPH && ./configure --prefix=$HOME/igraph && make && make install && cd ..
else
    echo "Using cached directory."
fi
