FROM node:22-bookworm-slim

# Add build arguments for UID and GID with defaults
ARG USER_ID=1000
ARG GROUP_ID=1000

# Add support for danish language in OCR.
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive \
	apt-get install -qy \
	git \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER ${USER_ID}:${GROUP_ID}
