"""Deprecated!!!!!"""

"""Manages interactions with the DHT which acts as a 'bulletin board for the network"""

from __future__ import annotations
from typing import Dict, Any, Union, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from client.client import Client
    from server.server import Server


class DHT:
    _client_node: "Client" = None
    _server_nodes: List["Server"] = []
    # def __init__(
    #     self,
    #     client_node: "Client",
    #     servers_nodes: List[Server],
    # ) -> None:
    #     self.client_node = client_node
    #     self.server_nodes = servers_nodes

    @classmethod
    def register_client(cls, client: "Client") -> None:
        cls._client_node = client

    @classmethod
    def register_servers(cls, servers: List["Server"]) -> None:
        cls._server_nodes = servers

    @classmethod
    def find_chain_head(cls) -> Server:
        for server in cls._server_nodes:
            if server.is_head:
                return server
