FROM continuumio/anaconda:2019.10

COPY environment.yml .

RUN /opt/conda/bin/conda env create -f environment.yml
