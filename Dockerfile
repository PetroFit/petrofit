#FROM jupyter/scipy-notebook:db3ee82ad08a
FROM jupyter/scipy-notebook:4d9c9bd9ced0
#FROM continuumio/miniconda3

#########
# Setup #
#########

ENV HOME /home/jovyan
WORKDIR $HOME

# Clone PetroFit repo
# RUN git clone https://github.com/PetroFit/petrofit.git

USER root
COPY --chown=$NB_USER:users . $HOME/petrofit/
USER $NB_USER

ENV PETROFITDIR $HOME/petrofit

########################################
# Setup Conda and Install Requirements #
########################################

RUN conda env create -f $PETROFITDIR/environment.yml
ENV PATH /opt/conda/envs/petrofit/bin:$PATH
RUN echo "source activate petrofit" >> $HOME/.bashrc

ENV CONDA_DEFAULT_ENV petrofit

WORKDIR $HOME

####################
# Install PetroFit #
####################

WORKDIR $PETROFITDIR

RUN python setup.py develop

WORKDIR $HOME

