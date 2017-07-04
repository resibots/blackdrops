#!/bin/bash

git checkout -- .
git clean -df

cd deps/limbo
git checkout -- .
git clean -df
cd ../..

cd deps/libcmaes
git checkout -- .
git clean -df
cd ../..

cd deps/dart
git checkout -- .
git clean -df
cd ../..

cd deps/robot_dart
git checkout -- .
git clean -df
cd ../..