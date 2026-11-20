"""Entrypoint of the policy image.

The platform starts the container with no arguments and the single env var AUTH_TOKEN
(docs/nebius-competition.md:298-302); the server binds 0.0.0.0:8000 and gates both the
keepalive and the WebSocket session on the bearer token. An empty or non-printable
AUTH_TOKEN makes ModelServer refuse to start (positronic_model_server/server.py:110-111),
which is the documented behaviour.
"""

import logging
import os

from positronic_model_server.server import ModelServer
from positronic_model_server.server_wire import ServedHostPort
from positronic_model_server.websocket_wire import WebsocketWire

from scripted_policy import build_model

AUTH_TOKEN_ENV = 'AUTH_TOKEN'
HOST = '0.0.0.0'
PORT = 8000


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logger = logging.getLogger('policy_server')
    server = ModelServer(build_model, idle_timeout_min=None, auth_token=os.environ.get(AUTH_TOKEN_ENV))
    logger.info('Scripted pick-and-place policy listening on %s:%d (AUTH_TOKEN %s)',
                HOST, PORT, 'set' if os.environ.get(AUTH_TOKEN_ENV) else 'ABSENT — serving open')
    server.serve([WebsocketWire(ServedHostPort(HOST, PORT))])


if __name__ == '__main__':
    main()
