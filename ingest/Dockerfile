# Frame sampler image. Built by CI on version tags and published as
# ghcr.io/griffinseibold/crosswalk-ingest:<version>.
#
#   docker build -f ingest/Dockerfile -t crosswalk-ingest ingest/
#
# Builder: the official Rust image at the toolchain pinned in
# rust-toolchain.toml. Every dependency is pure Rust, so no -dev packages.
FROM rust:1.98.1-slim-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
# Fetch and compile dependencies in their own layer, so changing the
# sampler's code does not rebuild them.
RUN mkdir src && echo 'fn main() {}' > src/main.rs \
    && cargo build --release --locked \
    && rm -rf src
COPY src/ src/
COPY testdata/ testdata/
RUN touch src/main.rs && cargo build --release --locked

# Runtime: Debian trixie with its FFmpeg package (FFmpeg 7.1 in trixie).
# FFmpeg runs as a subprocess, so the decoder version is whatever this
# image's apt installs; the dated base tag keeps rebuilds reproducible.
FROM debian:trixie-20261020-slim
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent app
COPY --from=build /src/target/release/crosswalk-ingest /usr/local/bin/crosswalk-ingest
USER 10001:10001
EXPOSE 9100
ENTRYPOINT ["crosswalk-ingest"]
