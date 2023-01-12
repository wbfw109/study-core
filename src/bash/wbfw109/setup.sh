#!/bin/bash
poetry install --with web,db,vision
yarn install
yarn dlx @yarnpkg/sdks vscode # enable PnP in VS code
docker compose -f ref/dockerfiles/docker-compose.yml up -d
