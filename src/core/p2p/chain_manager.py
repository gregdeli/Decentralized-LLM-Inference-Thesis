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

    def join_chain(self, self_info: Dict[str, Any], max_num_layers: int, num_total_layers: int):
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
            self._form_initial_chain(self_info, max_num_layers, num_total_layers)
        else:
            logger.info(f"Found existing chain with head {head_id[:DIGITS_SHOW]}. Joining at the tail...")
            self._join_existing_chain(self_info, max_num_layers, num_total_layers)

        logger.info(f"Self Info: {self._get_self_info()}")

    def _form_initial_chain(self, self_info: Dict[str, Any], max_num_layers: int, num_total_layers: int):
        """Logic for the first server to establish the chain."""
        # logger.info(f"store(\"{HEAD_KEY}\":{self.node_id})")
        self.dht.store(HEAD_KEY, self.node_id, EXPIRATION_S)

        # logger.info(f"store(\"{TAIL_KEY}\":{self.node_id})")
        self.dht.store(TAIL_KEY, self.node_id, EXPIRATION_S)

        self.dht.store(TOTAL_LAYERS_KEY, num_total_layers, EXPIRATION_S)

        self_info["successor"] = None

        # Don't exceed the the maximum layer index
        end_idx = max_num_layers - 1
        if end_idx >= num_total_layers:
            end_idx = num_total_layers - 1

        # Update the ALL_LAYERS_KEY if all layers have been loaded
        if end_idx == num_total_layers - 1:
            self.dht.store(ALL_LAYERS_KEY, True, EXPIRATION_S)

        self_info["layers_loaded"] = (0, end_idx)

        server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(server_key, self_info, EXPIRATION_S)
        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} is now the head and tail of the chain.")

    def _join_existing_chain(self, self_info: Dict[str, Any], max_num_layers: int, num_total_layers: int):
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

        end_idx = start_idx + max_num_layers - 1
        if end_idx >= num_total_layers:
            end_idx = num_total_layers - 1

        # Update the ALL_LAYERS_KEY if all layers have been loaded
        if end_idx == num_total_layers - 1:
            self.dht.store(ALL_LAYERS_KEY, True, EXPIRATION_S)

        # self_info["layers_loaded"] = (start_idx, end_idx)
        self_layers_loaded = (start_idx, end_idx)

        # Update the old tail to point to the new server node
        tail_info["successor"] = {"id": self.node_id, "address": self_info["address"], "layers_loaded": self_layers_loaded}
        self.dht.store(tail_server_key, tail_info, EXPIRATION_S)
        logger.info(f"Updated previous tail's ({tail_id[:DIGITS_SHOW]}) successor to point to new node {self.node_id[:DIGITS_SHOW]}.")

        # Store self info and update the chain_tail value
        self_info["successor"] = None
        self_info["layers_loaded"] = self_layers_loaded
        self_server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_server_key, self_info, EXPIRATION_S)
        self.dht.store(TAIL_KEY, self.node_id, EXPIRATION_S)
        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} has joined as the new tail.")

        logger.info(f"Previous Tail Info: {tail_info}")

    def get_head_server_info(self, attempts: int = 5) -> Optional[Dict[str, Any]]:
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

    def get_successor_address(self, attempts: int = 5) -> Optional[str]:
        if self.is_tail() or self.is_backup():
            return None

        for attempt in range(attempts):
            logger.info(f"Attempting to get successor address info... (Attempt {attempt + 1})")
            self_info = self._get_self_info()
            successor_data = self_info.get("successor")
            if successor_data:
                return successor_data.get("address")
            else:
                logger.info(f"Retrying in 2 seconds...")
                time.sleep(2)

        return None

    def _is_head(self) -> bool:
        """Check if this node is the head of the server chain"""
        head_id = self.dht.get(HEAD_KEY)
        if not head_id:
            return False
        return head_id == self.node_id

    def is_tail(self) -> bool:
        """Checks if this node is the TAIL"""
        tail_id = self.dht.get(TAIL_KEY)
        if not tail_id:
            raise RuntimeError("Tail server not found")

        num_total_layers = self._get_num_total_layers()
        layers_loaded = self.get_layers_loaded()

        if tail_id == self.node_id and layers_loaded[1] == num_total_layers - 1:
            return True

        return False
    
    def node_is_tail(self, node_id) -> bool:
        """Checks if a specific node is the tail"""
        tail_id = self.dht.get(TAIL_KEY)
        if not tail_id:
            raise RuntimeError("Tail server not found")

        # num_total_layers = self._get_num_total_layers()
        # layers_loaded = self.get_layers_loaded()

        if tail_id == node_id:# and layers_loaded[1] == num_total_layers - 1:
            return True

        return False

    def is_backup(self) -> bool:
        """Checks if this node is a backup node"""
        self_info = self._get_self_info()
        return bool(self_info.get("is_backup", False))

    def get_failed_successor_data(self) -> Optional[Dict[str, Any]]:
        """
        Gets the data of this node's successor from the DHT.
        This is called by the node when it actively detects its successor is dead.
        """
        self_info = self._get_self_info()
        successor_data = self_info.get("successor")
        if not successor_data:
            logger.warning(f"Node {self.node_id[:DIGITS_SHOW]} has no successor data.")
            return None
        
        # Check if the successor is the TAIL
        if self.node_is_tail(successor_data.get("id")):
            successor_data["was_tail"] = True
            return successor_data

        # Get the successor's successor id
        successor_key = f"{SERVER_INFO_PREFIX}{successor_data.get('id')}"
        successor_info = self.dht.get(successor_key)
        if not successor_info:
            logger.warning(f"Could not fetch info for successor {successor_data.get('id')[:DIGITS_SHOW]} from DHT. It may have just expired.")
            return None

        successor_2_data = successor_info.get("successor")
        if not successor_2_data:
            logger.warning(f"Node {successor_data.get('id')[:DIGITS_SHOW]} has no successor data.")
            return None

        # successor_2_id = successor_2_data.get("id")
        successor_data["successor"] = successor_2_data

        return successor_data

    def repair(self, layers_loaded: Tuple[int, int], successor_2_data: Optional[Dict[str, Any]], succ_was_tail: Optional[bool]):
        """
        Updates the DHT after a node has taken over the layers of a failed successor.

        :param layers_loaded: The new, expanded tuple of layers this node now holds.
        :param successor_2_data: The data dict of the new successor (the old successor's successor).
                                 Can be None if the failed node was the tail.
        """
        logger.info(f"Repairing chain on DHT. New layers: {layers_loaded}, New successor: {successor_2_data}")

        # Update this node's info
        self_info = self._get_self_info()
        self_info["layers_loaded"] = layers_loaded
        if successor_2_data:
            self_info["successor"] = successor_2_data
        
        # If the dead successor was the tail, set this node as the tail
        elif succ_was_tail:
            self_info["successor"] = None
            self.dht.store(TAIL_KEY, self.node_id, EXPIRATION_S)

        server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(server_key, self_info, EXPIRATION_S)

        # Set the "all_layers_loaded" key to True
        self.dht.store(ALL_LAYERS_KEY, True, EXPIRATION_S)

    def republish_keys(self):
        """
        Periodically called to maintain the node's presence on the DHT and check chain integrity.

        :return: The dead successor's data dictionary if detected, else None.
        """
        server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self_info = self._get_self_info()
            
        # Republish server info
        self.dht.store(server_key, self_info, EXPIRATION_S)

        # If this node is the head
        if self._is_head():
            # Republish the head key
            self.dht.store(HEAD_KEY, self.node_id, EXPIRATION_S)

            # Republish the num_total_layers key
            num_total_layers = self._get_num_total_layers()
            if num_total_layers:
                self.dht.store(TOTAL_LAYERS_KEY, num_total_layers, EXPIRATION_S)

            all_layers_loaded = self.get_all_layers_loaded()
            if all_layers_loaded is not None:
                self.dht.store(ALL_LAYERS_KEY, all_layers_loaded, EXPIRATION_S)

        # If this node is the tail, republish the tail key
        if self.is_tail():
            self.dht.store(TAIL_KEY, self.node_id, EXPIRATION_S)

        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} republished its keys.")

    def update_layers_loaded(self, new_layers: Tuple[int, int]):
        self_info = self._get_self_info()
        self_info["layers_loaded"] = new_layers
        server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(server_key, self_info, EXPIRATION_S)
    
    def print_chain_status(self):
        """
        Prints the full status of the inference chain from the DHT.
        """
        print("--------- Chain Status ---------")
        # Global keys
        head_id = self.dht.get(HEAD_KEY)
        tail_id = self.dht.get(TAIL_KEY)
        total_layers = self.dht.get(TOTAL_LAYERS_KEY)
        all_loaded = self.dht.get(ALL_LAYERS_KEY)

        print("Global Info:")
        print(f"  - Total Layers: {total_layers if total_layers is not None else 'Not Found'}")
        print(f"  - All Layers Loaded: {all_loaded if all_loaded is not None else 'Not Found'}")
        print(f"  - Head Node ID: {head_id[:DIGITS_SHOW] if head_id else 'Not Found'}")
        print(f"  - Tail Node ID: {tail_id[:DIGITS_SHOW] if tail_id else 'Not Found'}")

        if not head_id:
            print("\nChain is empty.")
            print("---------- End of Status ---------")
            return
        
        # Traverse chain and print server info
        print("\nServer Chain (from Head to Tail):")
        current_node_id = head_id
        counter = 1

        while current_node_id:
            server_key = f"{SERVER_INFO_PREFIX}{current_node_id}"
            info = self.dht.get(server_key)
            if not info:
                print(f"  {counter}. Node ID: {current_node_id[:DIGITS_SHOW]}")
                print("     [ERROR: Could not fetch info for this node. Chain traversal stopped.]")
                break

            # Extract info
            address = info.get("address")
            layers = info.get("layers_loaded")
            successor_data = info.get("successor")
            is_backup = info.get("is_backup", False)

            print(f"  {counter}. Node ID: {current_node_id[:DIGITS_SHOW]}")
            print(f"     - Address: {address}")
            print(f"     - Layers: {layers}")
            
            if successor_data:
                print(f"     - Successor: {successor_data.get('id')[:DIGITS_SHOW]}")
                current_node_id = successor_data.get("id")
            else:
                print("     - Successor: None")
                current_node_id = None # End of chain
            
            counter += 1
        
        print("--------- End of Status ---------")




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
