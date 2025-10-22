import logging
from typing import Dict, Any, Optional, Tuple, Union
import time

from core.p2p.dht_manager import DHTManager

logger = logging.getLogger(__name__)

# Constants for keys on the DHT
HEAD_KEY = "chain_head"
TAIL_KEY = "chain_tail"
TOTAL_LAYERS_KEY = "num_total_layers"
ALL_LAYERS_KEY = "all_layers_loaded"
SERVER_INFO_PREFIX = "server_info_"

EXPIRATION_S = 60.0
HEARTBEAT_INTERVAL_S = EXPIRATION_S / 2.0

DIGITS_SHOW = 12


class ChainManager:
    """Manages the creation and discovery of the server inference chain on the DHT."""

    def __init__(self, dht_manager: DHTManager):
        self.dht = dht_manager
        self.node_id = dht_manager.get_id()

    def join_chain(self, self_info: Dict[str, Any], num_layers: int, num_total_layers: int):
        """
        Main entry point for a server node to join or form the inference chain.
        It determines if it's the first node or joining an existing chain.

        :param:
          self_info: A dictionary with the server's data (e.g., {'address': 'head-server:5001'}).
          num_layers: The number of Transformers Layers the node can load. 
          num_total_layers: The number of total Transformer Layers of the LLM.
        :return: The updated info dictionary for this server.
        """
        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} attempting to join the chain...")
        head_id = self.dht.get(HEAD_KEY)

        if head_id is None:
            logger.info("No existing chain found. Forming a new one...")
            self._form_initial_chain(self_info, num_layers, num_total_layers)
        else:
            logger.info(f"Found existing chain with head {head_id[:DIGITS_SHOW]}. Joining at the tail...")
            self._join_existing_chain(self_info, num_layers, num_total_layers)
        
        logger.info(f"Self Info: {self._get_self_info()}")

    def _form_initial_chain(self, self_info: Dict[str, Any], num_layers:int, num_total_layers: int):
        """Logic for the first server to establish the chain."""
        # logger.info(f"store(\"{HEAD_KEY}\":{self.node_id})")
        self.dht.store(HEAD_KEY, self.node_id, EXPIRATION_S)

        # logger.info(f"store(\"{TAIL_KEY}\":{self.node_id})")
        self.dht.store(TAIL_KEY, self.node_id, EXPIRATION_S)

        self.dht.store(TOTAL_LAYERS_KEY, num_total_layers, EXPIRATION_S)

        self_info["successor"] = None

        # Don't exceed the the maximum layer index
        end_idx = num_layers - 1
        if end_idx >= num_total_layers:
            end_idx = num_total_layers - 1
        
        # Update the ALL_LAYERS_KEY if all layers have been loaded
        if end_idx == num_total_layers - 1:
            self.dht.store(ALL_LAYERS_KEY, True, EXPIRATION_S)

        self_info["layers_loaded"] = (0, end_idx)            

        server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(server_key, self_info, EXPIRATION_S)
        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} is now the head and tail of the chain.")

    def _join_existing_chain(self, self_info: Dict[str, Any], num_layers:int, num_total_layers: int):
        """Logic for a new server to join an existing chain."""
        # Find the current tail
        tail_id = self.dht.get(TAIL_KEY)
        if not tail_id:
            raise RuntimeError("Chain head exists, but tail was not found. The network is in an inconsistent state.")

        # Get the current tail's info to update its successor
        tail_server_key = f"{SERVER_INFO_PREFIX}{tail_id}"
        tail_info = self.dht.get(tail_server_key)
        if not tail_info:
            raise RuntimeError(f"Could not retrieve info for tail node {tail_id}.")

        # Get the previous tails layers_loaded to determine this nodes layer range
        start_idx = tail_info["layers_loaded"][1] + 1

        # If all the layers have already been loaded then dont add this node to the chain
        # and set it as a backup node 
        if start_idx >= num_total_layers:
            logger.warning("All the layers have already been loaded on the the previous tail")
            logger.warning("Setting this node as a backup node...")
            
            self_info["is_backup"] = True
            self_server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
            self.dht.store(self_server_key, self_info, EXPIRATION_S)
            return

        # Update the old tail to point to the new server node
        tail_info["successor"] = self.node_id
        self.dht.store(tail_server_key, tail_info, EXPIRATION_S)
        logger.info(f"Updated previous tail's ({tail_id[:DIGITS_SHOW]}) successor to point to new node {self.node_id[:DIGITS_SHOW]}.")

        # Store our own info and update the tail pointer to us
        self_info["successor"] = None

        end_idx = start_idx + num_layers - 1
        if end_idx >= num_total_layers:
            end_idx = num_total_layers - 1

        # Update the ALL_LAYERS_KEY if all layers have been loaded
        if end_idx == num_total_layers - 1:
            self.dht.store(ALL_LAYERS_KEY, True, EXPIRATION_S)

        self_info["layers_loaded"] = (start_idx, end_idx)

        # Store self info and update the chain_tail value
        self_server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_server_key, self_info, EXPIRATION_S)
        self.dht.store(TAIL_KEY, self.node_id, EXPIRATION_S)
        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} has joined as the new tail.")

        logger.info(f"Previous Tail Info: {tail_info}")

    def get_head_server_info(self, attempts:int = 5) -> Optional[Dict[str, Any]]:
        """
        Client-side function to find the head of the chain and get its connection info.

        :return: A dictionary containing the head server's info, or None if not found.
        """
        num_total_layers = self._get_num_total_layers()

        logger.info("Client searching for the head of the server chain...")
        head_id = self.dht.get(HEAD_KEY)
        if not head_id:
            logger.warning("Could not find the head of the chain on the DHT.")
            return None

        head_server_key = f"{SERVER_INFO_PREFIX}{head_id}"

        # If the head server doesn't hold all the layers, 
        # retry until its successor is not None
        for attempt in range(attempts):
            logger.info(f"Attempting to get head server info... (Attempt {attempt + 1})")
            head_info = self.dht.get(head_server_key)
            if not head_info:
                logger.error(f"Found head ID {head_id} but could not retrieve its info.")
                return None

            head_layers_loaded = head_info["layers_loaded"]
            head_successor = head_info["successor"]
            if head_layers_loaded[1] >= num_total_layers - 1 or head_successor is not None:
                logger.info(f"Found head server {head_id[:DIGITS_SHOW]} with info: {head_info}")
                return head_info
            logger.info(f"Attempt {attempt + 1}: Found head server but with layers_loaded < total and no successor.")
            logger.info(f"Retrying in 2 seconds...")
            time.sleep(2)

    def _get_self_info(self) -> Dict[str, Any]:
        """Get the server info dict for this node from the DHT"""
        self_info = self.dht.get(f"{SERVER_INFO_PREFIX}{self.node_id}")
        if not self_info:
            raise RuntimeError(f"Could not retrieve info for node {self.node_id}.")
        return self_info
    
    def _get_num_total_layers(self) -> int:
        num_total_layers = self.dht.get(TOTAL_LAYERS_KEY)
        if not num_total_layers:
            raise RuntimeError(f"Could not retrieve the total number of transformer layers.")
        return num_total_layers

    def get_layers_loaded(self) -> Tuple[int, int]:
        """Get the layers_loaded tuple for this node from the DHT"""
        self_info = self._get_self_info()
        return self_info.get("layers_loaded")
    
    def get_all_layers_loaded(self) -> Union[bool, None]:
            return self.dht.get(ALL_LAYERS_KEY)

    def get_successor_address(self) -> Optional[str]:
        self_info = self._get_self_info()
        successor_id = self_info.get("successor")

        if not successor_id:
            # If this node is the chain tail
            return None

        successor_info = self.dht.get(f"{SERVER_INFO_PREFIX}{successor_id}")
        if not successor_info:
            logger.error(f"Found successor ID {successor_id} but could not retrieve its info.")
            return None

        return successor_info.get("address")
    
    def _is_head(self) -> bool:
        """Check if this node is the head of the server chain"""
        head_id = self.dht.get(HEAD_KEY)
        if not head_id:
            return False 
        return head_id == self.node_id

    def is_tail(self) -> bool:
        tail_id = self.dht.get(TAIL_KEY)
        if not tail_id:
            raise RuntimeError("Tail server not found")

        return tail_id == self.node_id

    def is_backup(self) -> bool: 
        self_info = self._get_self_info()
        return bool(self_info.get("is_backup", False))
    
    def is_successor_alive(self) -> bool:
        """Check on this nodes successor"""
        self_info = self._get_self_info()
        if not self.is_tail():
            successor_id = self_info["successor"]
            successor_key = f"{SERVER_INFO_PREFIX}{successor_id}"
            successor_info = self.dht.get(successor_key)
            if not successor_info:
                logger.warning(f"This node's successor is DEAD.")
                return False
            return True # If successor info is retrieved successfully it is alive
        else:
            return True # This is the tail server


    def republish_keys(self) -> bool:
        """
        Periodically called to maintain the node's presence on the DHT and check chain integrity.

        :return: True if the successor was detected as dead during this check, False otherwise.
        """
        server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self_info = self._get_self_info()

        # Check if this node's successor is alive
        successor_dead = False
        if self.is_successor_alive():
            # If successor is alive or this node is the tail republish the "all_layers_loaded" key
            all_layers_loaded = self.get_all_layers_loaded() 
            if all_layers_loaded is not None:
                self.dht.store(ALL_LAYERS_KEY, all_layers_loaded, EXPIRATION_S)
        else:
            # If the successor is dead store "all_layers_loaded": False
            successor_dead = True
            self.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
            self_info["successor"] = None
            
        # Republish server info
        self.dht.store(server_key, self_info, EXPIRATION_S)

        # If this node is the head, republish the head key
        if self._is_head():
            self.dht.store(HEAD_KEY, self.node_id, EXPIRATION_S)
            # Also republish the num_total_layers key
            num_total_layers = self._get_num_total_layers() 
            if num_total_layers:
                    self.dht.store(TOTAL_LAYERS_KEY, num_total_layers, EXPIRATION_S)

        # If this node is the tail, republish the tail key
        if self.is_tail():
            self.dht.store(TAIL_KEY, self.node_id, EXPIRATION_S)
        
        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} republished its keys.")
        return successor_dead


# Example usage to demonstrate the flow
def main():
    # --- Start Server 1 (Head) ---
    dht1 = DHTManager(host_maddrs=["/ip4/127.0.0.1/tcp/0"])
    dht1.start()
    chain1 = ChainManager(dht1)

    server1_info = {"num_layers": 4, "layers": (0, 3), "address": "localhost:50051"}
    chain1.join_chain(server1_info)

    # --- Start Server 2 (Joiner) ---
    # It uses Server 1's address to connect
    initial_peers = dht1.dht.get_visible_maddrs()
    dht2 = DHTManager(host_maddrs=["/ip4/127.0.0.1/tcp/0"], initial_peers=initial_peers)
    dht2.start()
    chain2 = ChainManager(dht2)

    server2_info = {"num_layers": 4, "layers": (4, 7), "address": "localhost:50052"}
    chain2.join_chain(server2_info)

    # --- Client discovers the head ---
    client_dht = DHTManager(host_maddrs=["/ip4/127.0.0.1/tcp/0"], initial_peers=initial_peers)
    client_dht.start()
    client_chain_manager = ChainManager(client_dht)

    head_info = client_chain_manager.get_head_server_info()
    if head_info:
        print(f"\nClient successfully found head server. Address: {head_info.get('address')}")
    else:
        print("\nClient could not find the head server.")

    # --- Shutdown ---
    dht1.shutdown()
    dht2.shutdown()
    client_dht.shutdown()


if __name__ == "__main__":
    main()
