#!/bin/bash
mkdir -p /Data/amine.chraibi/msd
cd /Data/amine.chraibi/msd
kaggle datasets download -d caspervanengelenburg/modified-swiss-dwellings
unzip -o modified-swiss-dwellings.zip
rm modified-swiss-dwellings.zip
