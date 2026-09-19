# Build context: an isolated directory prepared by build-image.sh.
ARG BASE_IMAGE=lizenan1995/code-prover-lean@sha256:b4850e1e3dd01b81ae4973f9f10b3344e1320387333d0cf61098d94e8239cba1
FROM ${BASE_IMAGE}
USER root
RUN test "$(cat /task/lean-toolchain)" = 'leanprover/lean4:v4.28.0' && chmod a+rx /root
COPY bin/ /opt/codeprover/bin/
COPY lean/ /opt/codeprover/lean/
COPY phase.py /opt/codeprover/phase.py
ENV HOME=/tmp
ENV PATH=/root/.elan/toolchains/leanprover--lean4---v4.28.0/bin:/usr/local/bin:/usr/bin:/bin
USER 65532:65532
WORKDIR /work
