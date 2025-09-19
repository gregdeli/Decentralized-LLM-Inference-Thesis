import asyncio
import logging
from typing import Dict, Any, Optional

from core.p2p.dht_manager import DHTManager

logger = logging.getLogger(__name__)

# Constants for keys on the DHT
HEAD_KEY = "chain_head"
TAIL_KEY = "chain_tail"
SERVER_INFO_PREFIX = "server_info_"
HEARTBEAT_INTERVAL_S = 30.0


class ChainManager:
    """Manages the creation and discovery of the server inference chain on the DHT."""

    def __init__(self, dht_manager: DHTManager):
        self.dht = dht_manager
        self.node_id = dht_manager.get_id()

    async def join_chain(self, self_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for a server node to join or form the inference chain.
        It determines if it's the first node or joining an existing chain.

        :param self_info: A dictionary with the server's data (e.g., {'layers': (0, 3), 'address': 'head-server:50051'}).
        :return: The updated info dictionary for this server.
        """
        logger.info(f"Node {self.node_id} attempting to join the chain...")
        head_id = await self.dht.get(HEAD_KEY)

        if head_id is None:
            logger.info("No existing chain found. Forming a new one.")
            await self._form_initial_chain(self_info)
        else:
            logger.info(f"Found existing chain with head {head_id}. Joining at the tail.")
            await self._join_existing_chain(self_info)

        return self_info

    async def _form_initial_chain(self, self_info: Dict[str, Any]):
        """Logic for the first server to establish the chain."""
        await self.dht.store(HEAD_KEY, self.node_id, HEARTBEAT_INTERVAL_S)
        await self.dht.store(TAIL_KEY, self.node_id, HEARTBEAT_INTERVAL_S)

        self_info["successor"] = None
        server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        await self.dht.store(server_key, self_info, HEARTBEAT_INTERVAL_S)
        logger.info(f"Node {self.node_id} is now the head and tail of the chain.")

    async def _join_existing_chain(self, self_info: Dict[str, Any]):
        """Logic for a new server to join an existing chain."""
        # Find the current tail
        tail_id = await self.dht.get(TAIL_KEY)
        if not tail_id:
            raise RuntimeError("Chain head exists, but tail was not found. The network is in an inconsistent state.")

        # Get the current tail's info to update its successor
        tail_server_key = f"{SERVER_INFO_PREFIX}{tail_id}"
        tail_info = await self.dht.get(tail_server_key)
        if not tail_info:
            raise RuntimeError(f"Could not retrieve info for tail node {tail_id}.")

        # Update the old tail to point to the new server node
        tail_info["successor"] = self.node_id
        await self.dht.store(tail_server_key, tail_info, HEARTBEAT_INTERVAL_S)
        logger.info(f"Updated previous tail {tail_id} to point to new node {self.node_id}.")

        # Store our own info and update the tail pointer to us
        self_info["successor"] = None
        server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        await self.dht.store(server_key, self_info, HEARTBEAT_INTERVAL_S)
        await self.dht.store(TAIL_KEY, self.node_id, HEARTBEAT_INTERVAL_S)
        logger.info(f"Node {self.node_id} has joined as the new tail.")

    async def find_head_server_info(self) -> Optional[Dict[str, Any]]:
        """
        Client-side function to find the head of the chain and get its connection info.

        :return: A dictionary containing the head server's info, or None if not found.
        """
        logger.info("Client searching for the head of the server chain...")
        head_id = await self.dht.get(HEAD_KEY)
        if not head_id:
            logger.warning("Could not find the head of the chain on the DHT.")
            return None

        head_server_key = f"{SERVER_INFO_PREFIX}{head_id}"
        head_info = await self.dht.get(head_server_key)
        if not head_info:
            logger.error(f"Found head ID {head_id} but could not retrieve its info.")
            return None

        logger.info(f"Found head server {head_id} with info: {head_info}")
        return head_info


# Example usage to demonstrate the flow
async def main():
    # --- Start Server 1 (Head) ---
    dht1 = DHTManager(host_maddrs=["/ip4/127.0.0.1/tcp/0"])
    await dht1.start()
    chain1 = ChainManager(dht1)

    server1_info = {"layers": (0, 3), "address": "localhost:50051"}
    await chain1.join_chain(server1_info)

    # --- Start Server 2 (Joiner) ---
    # It uses Server 1's address to connect
    initial_peers = dht1.dht.get_visible_maddrs()
    dht2 = DHTManager(host_maddrs=["/ip4/127.0.0.1/tcp/0"], initial_peers=initial_peers)
    await dht2.start()
    chain2 = ChainManager(dht2)

    server2_info = {"layers": (4, 7), "address": "localhost:50052"}
    await chain2.join_chain(server2_info)

    # --- Client discovers the head ---
    client_dht = DHTManager(host_maddrs=["/ip4/127.0.0.1/tcp/0"], initial_peers=initial_peers)
    await client_dht.start()
    client_chain_manager = ChainManager(client_dht)

    head_info = await client_chain_manager.find_head_server_info()
    if head_info:
        print(f"\nClient successfully found head server. Address: {head_info.get('address')}")
    else:
        print("\nClient could not find the head server.")

    # --- Shutdown ---
    await dht1.shutdown()
    await dht2.shutdown()
    await client_dht.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
