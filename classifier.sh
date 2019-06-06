#!/bin/bash

if python3 ./final.py $1
 then exit 0 # Fake
 else exit 1 # Real
fi
