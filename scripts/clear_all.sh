#!/bin/bash

git checkout -- .
git clean -df

cd limbo
git checkout -- .
git clean -df
cd ..

cd libcmaes
git checkout -- .
git clean -df
cd ..

cd dart
git checkout -- .
git clean -df
cd ..

cd robot_dart
git checkout -- .
git clean -df
cd ..