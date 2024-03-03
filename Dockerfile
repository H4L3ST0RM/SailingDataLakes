FROM gcr.io/cloud-builders/git
ENTRYPOINT ["/bin/bash"]
COPY . /project
WORKDIR /project
RUN git submodule update --init --recursive
RUN git submodule update --remote --merge

FROM ghcr.io/getzola/zola:v0.18.0 as zola
COPY . /project
WORKDIR /project
RUN ["zola", "build"]

FROM ghcr.io/static-web-server/static-web-server:2
WORKDIR /
COPY --from=zola /project/public /public