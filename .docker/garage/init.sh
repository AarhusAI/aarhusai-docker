#!/bin/sh
set -e

GARAGE_HOST="${GARAGE_RPC_HOST%%:*}"
unset GARAGE_RPC_HOST

echo "Waiting for Garage to be ready..."
until nc -z "$GARAGE_HOST" 3901 2>/dev/null; do
  sleep 1
done

# Resolve hostname to IP — Garage config requires a socket address, not a hostname
GARAGE_IP=$(getent hosts "$GARAGE_HOST" | awk '{print $1}')
sed -e "s|api_bind_addr = .*|api_bind_addr = \"${GARAGE_IP}:3903\"|" \
    -e "s|rpc_public_addr = .*|rpc_public_addr = \"${GARAGE_IP}:3901\"|" \
  /etc/garage.toml > /tmp/garage-remote.toml
export GARAGE_CONFIG_FILE=/tmp/garage-remote.toml
echo "Garage is reachable."

NODE_ID=$(/garage node id -q | cut -c1-16)
echo "Node ID: $NODE_ID"

# Layout: assign node if not already in the current layout
if ! /garage layout show 2>&1 | grep -q "$NODE_ID"; then
  echo "Assigning node to layout..."
  /garage layout assign -z dc1 -c 1G "$NODE_ID"
  /garage layout apply --version 1
  echo "Layout applied."
else
  echo "Node already in layout, skipping."
fi

# Key: import if not exists
if /garage key info openwebui-key >/dev/null 2>&1; then
  echo "Key 'openwebui-key' already exists, skipping."
else
  echo "Importing key 'openwebui-key'..."
  /garage key import --yes -n openwebui-key "$GARAGE_ACCESS_KEY" "$GARAGE_SECRET_KEY"
  echo "Key imported."
fi

# Bucket: create if not exists
if /garage bucket info "$GARAGE_BUCKET" >/dev/null 2>&1; then
  echo "Bucket '$GARAGE_BUCKET' already exists, skipping."
else
  echo "Creating bucket '$GARAGE_BUCKET'..."
  /garage bucket create "$GARAGE_BUCKET"
  /garage bucket allow "$GARAGE_BUCKET" --read --write --key openwebui-key
  echo "Bucket created and key granted access."
fi

echo "Garage init complete."
