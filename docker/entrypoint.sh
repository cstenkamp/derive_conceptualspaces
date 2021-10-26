#!/usr/bin/env bash

groupadd --force --gid ${APP_GID} appuser
useradd -c 'container user' -u $APP_UID -g $APP_GID appuser
chown -R $APP_UID:$APP_GID /app/data
exec $@

#TODO deleteme!