# GitHub Actions self-hosted runner, containerised.
#
# WHY A CONTAINER: the runner shares a headless Mac mini with other services
# (Grafana, storage). The container is how the runner gets bounded — see the
# cpu/memory limits in docker-compose.yml — so a runaway job cannot starve
# everything else on the box.
#
# WHAT THIS RUNNER ACTUALLY DOES: it does NOT run the test suite. That is
# ci.yml, which runs on GitHub-hosted ubuntu-latest. This runner serves
# deploy.yml / pod.yml / pipeline.yml, which ssh + rsync to the training pod.
# So openssh-client and rsync are LOAD-BEARING, not conveniences: without them
# scripts/pod_exec.sh and scripts/deploy_skybot.sh fail at the first step.
FROM ubuntu:22.04

ARG RUNNER_VERSION=2.337.0
ARG TARGETARCH=x64
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq sudo tar gzip \
        openssh-client rsync \
        libicu70 libssl3 \
    && rm -rf /var/lib/apt/lists/*

# The runner refuses to execute as root, by its own check.
RUN useradd -m -s /bin/bash runner
WORKDIR /home/runner

RUN curl -fsSL -o runner.tar.gz \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-${TARGETARCH}-${RUNNER_VERSION}.tar.gz" \
    && tar xzf runner.tar.gz \
    && rm runner.tar.gz \
    && ./bin/installdependencies.sh \
    && chown -R runner:runner /home/runner

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER runner
# StrictHostKeyChecking=no + /dev/null known_hosts: pod providers reuse IPs
# across pods, so the host key changes and plain ssh refuses outright. This is
# the same setting scripts/deploy_skybot.sh documents.
RUN mkdir -p /home/runner/.ssh && chmod 700 /home/runner/.ssh \
    && printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  ConnectTimeout 20\n' \
       > /home/runner/.ssh/config \
    && chmod 600 /home/runner/.ssh/config

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
