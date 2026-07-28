# !/bin/bash

pip uninstall -y python-terrier pyterrier
pip install -r requirements.txt

conda install -y -c conda-forge openjdk=21

echo DONE