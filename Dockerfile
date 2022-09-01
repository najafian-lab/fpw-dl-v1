# start with anaconda base and python3
FROM continuumio/anaconda3

# copy the deps definition file
COPY docker/environment.yml /deps/env.yml

# build/install dependencies
# note we split it up because the layer sizes are absolutely massive otherwise
# NOTE: had to get each layer under 3GB which is why you see this mess
WORKDIR /deps
RUN conda env create --file env.yml && conda clean -a
RUN conda install -n fpw-dl conda-forge::mahotas=1.4.* opencv=3.4.* scikit-image=0.15.*

# copy pip related deps
COPY docker/pipreqs.txt /deps/pipreqs.txt

# Make RUN commands use the new anaconda environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "fpw-dl", "/bin/bash", "-c"]

RUN python -m pip install --no-cache-dir "tensorflow-gpu>=1.15.0,<1.15.6"
RUN python -m pip install --no-cache-dir -r /deps/pipreqs.txt

# specify the worker we're in docker which will have some defaults/parameters changes
ENV IN_DOCKER=1

# copy main files to container
WORKDIR /src
COPY model /src/model
COPY najafian /src/najafian
COPY docker /src/docker
COPY analysis.py download.py configs.py eval.py figure.py configs.json LICENSE.md README.md /src/


# specify entrypoint
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "fpw-dl", "python" ]