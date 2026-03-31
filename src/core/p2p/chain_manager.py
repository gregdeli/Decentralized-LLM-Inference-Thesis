import logging
from typing import Dict, Any, Optional, Tuple, Union, List
import time

from core.p2p.dht_manager import DHTManager

logger = logging.getLogger(__name__)

# Global keys on the DHT
HEAD_KEY = "chain_head"
TAIL_KEY = "chain_tail"
TOTAL_LAYERS_KEY = "num_total_layers"
ALL_LAYERS_KEY = "all_layers_loaded"
BACKUPS_KEY = "backup_nodes"
SERVER_INFO_PREFIX = "server_info_"

# EXPIRATION_S = 30.0 
EXPIRATION_S = 7200.0 
# HEARTBEAT_INTERVAL_S = EXPIRATION_S / 4.0 
HEARTBEAT_INTERVAL_S = 30.0

DIGITS_SHOW = 12


class ChainManager:
    """Manages the creation and discovery of the server inference chain on the DHT."""

    def __init__(self, dht_manager: DHTManager):
        self.dht = dht_manager
        self.node_id = dht_manager.get_id()

    def join_chain(self, self_info: Dict[str, Any], max_num_layers: int, num_total_layers: int) -> None:
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

        self.dht.store(BACKUPS_KEY, [], EXPIRATION_S)

        self_info["successor"] = None

        # Don't exceed the the maximum layer index
        end_idx = max_num_layers - 1
        if end_idx >= num_total_layers:
            end_idx = num_total_layers - 1

        self_info["layers"] = (0, end_idx)

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

        # Get the previous tails layers to determine this nodes layer range
        start_idx = tail_info["layers"][1] + 1

        # Backup Node, if all layers loaded
        if start_idx >= num_total_layers:
            logger.warning("All the layers have already been loaded on the the previous tail")
            logger.warning("Setting this node as a backup node...")

            self_info["is_backup"] = True
            self_server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
            self.dht.store(self_server_key, self_info, EXPIRATION_S)

            # Update the backup_nodes list
            backup_nodes = self.get_backup_nodes()
            backup_nodes.append(self.node_id)
            logger.info(f"backup_nodes list updated: {backup_nodes}")
            self.dht.store(BACKUPS_KEY, backup_nodes, EXPIRATION_S)
            return

        end_idx = start_idx + max_num_layers - 1
        if end_idx >= num_total_layers:
            end_idx = num_total_layers - 1

        self_layers = (start_idx, end_idx)

        # Update the old tail to point to the new server node
        tail_info["successor"] = {"id": self.node_id, "address": self_info["address"]}
        self.dht.store(tail_server_key, tail_info, EXPIRATION_S)
        logger.info(f"Updated previous tail's ({tail_id[:DIGITS_SHOW]}) successor to point to new node {self.node_id[:DIGITS_SHOW]}.")

        # Store self info and update the chain_tail value
        self_info["successor"] = None
        self_info["layers"] = self_layers
        self_server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_server_key, self_info, EXPIRATION_S)
        self.dht.store(TAIL_KEY, self.node_id, EXPIRATION_S)
        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} has joined as the new tail.")

        logger.info(f"Previous Tail Info: {tail_info}")

    def get_head_server_info(self) -> Optional[Dict[str, Any]]:
        """
        Client-side function to find the head of the chain and get its connection info.

        :return: A dictionary containing the head server's info, or None if not found.
        """

        logger.info("Client searching for the head of the server chain...")
        head_id = self.dht.get(HEAD_KEY)
        if not head_id:
            logger.warning("Could not find the head of the chain on the DHT.")
            return None

        head_server_key = f"{SERVER_INFO_PREFIX}{head_id}"

        logger.info(f"Attempting to get head server info...")
        head_info = self.dht.get(head_server_key)
        if not head_info:
            logger.error(f"Found head ID {head_id} but could not retrieve its info.")
            return None

        head_layers = head_info["layers"]
        head_successor = head_info["successor"]
        logger.info(f"Found head server {head_id[:DIGITS_SHOW]} with info: {head_info}")

        return head_info

    def _get_self_info(self) -> Dict[str, Any]:
        """Get the server info dict for this node from the DHT"""
        self_info = self.dht.get(f"{SERVER_INFO_PREFIX}{self.node_id}")
        if not self_info:
            raise RuntimeError(f"Could not retrieve info for node {self.node_id}.")
        return self_info

    def get_server_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.dht.get(f"{SERVER_INFO_PREFIX}{node_id}")

    def _get_num_total_layers(self) -> int:
        num_total_layers = self.dht.get(TOTAL_LAYERS_KEY)
        if not num_total_layers:
            raise RuntimeError(f"Could not retrieve the total number of transformer layers.")
        return num_total_layers

    def get_layers(self) -> Optional[Tuple[int, int]]:
        """Get the layers tuple for this node from the DHT"""
        self_info = self._get_self_info()
        return self_info.get("layers")

    def get_all_layers_loaded(self) -> Optional[bool]:
        return self.dht.get(ALL_LAYERS_KEY)

    def get_backup_nodes(self) -> Optional[List[str]]:
        return self.dht.get(BACKUPS_KEY)

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

        if tail_id == self.node_id: 
            return True

        return False

    def node_is_tail(self, node_id: str) -> bool:
        """Checks if a specific node is the tail"""
        tail_id = self.dht.get(TAIL_KEY)
        if not tail_id:
            raise RuntimeError("Tail server not found")

        if tail_id == node_id:  
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

        # Get the failed successor's layers
        successor_key = f"{SERVER_INFO_PREFIX}{successor_data.get('id')}"
        successor_info = self.dht.get(successor_key)
        if not successor_info:
            logger.warning(f"Could not fetch info for successor {successor_data.get('id')[:DIGITS_SHOW]} from DHT. It may have just expired.")
            return None

        successor_data["layers"] = successor_info.get("layers")

        # Check if the successor is the TAIL
        if self.node_is_tail(successor_data.get("id")):
            successor_data["was_tail"] = True
            return successor_data

        # Get the successor's successor data
        successor_2_data = successor_info.get("successor")
        if not successor_2_data:
            logger.warning(f"Node {successor_data.get('id')[:DIGITS_SHOW]} has no successor data.")
            return None

        successor_data["successor"] = successor_2_data
        return successor_data

    def repair(
            self, 
            layers: Tuple[int, int], 
            successor_2_data: Optional[Dict[str, Any]], 
            succ_was_tail: Optional[bool], 
            replacement_node_id: Optional[str] = None,
            ):
        """
        Updates the DHT after a node has taken over the layers of a failed successor.

        :param layers: The new, expanded tuple of layers this node now holds.
        :param successor_2_data: The data dict of the new successor (the old successor's successor).
                                 Can be None if the failed node was the tail.
        """
        logger.info(f"Repairing chain on DHT. New layers: {layers}, New successor: {successor_2_data}")

        self_info = self._get_self_info()

        if replacement_node_id:
            logger.info(f"With replacement backup node: {replacement_node_id}")
            replacement_info = self.get_server_info(replacement_node_id)
            replacement_info["is_backup"] = False

            # Remove the replacement node from the backup_nodes list
            backup_nodes = self.get_backup_nodes()
            backup_nodes.remove(replacement_node_id)
            self.dht.store(BACKUPS_KEY, backup_nodes, EXPIRATION_S)
            
            # Also update this nodes successor to be the replacement
            new_successor_data = {"id": replacement_node_id, "address": replacement_info.get("address")}
            self.update_successor(new_successor_data)
        else:
            replacement_node_id = self.node_id
            replacement_info = self_info

        # Update the replacement node's info
        replacement_info["layers"] = layers
        if successor_2_data:
            replacement_info["successor"] = successor_2_data

        # If the dead successor was the tail, set this node as the tail
        elif succ_was_tail:
            replacement_info["successor"] = None
            self.dht.store(TAIL_KEY, replacement_node_id, EXPIRATION_S)

        server_key = f"{SERVER_INFO_PREFIX}{replacement_node_id}"
        self.dht.store(server_key, replacement_info, EXPIRATION_S)

        self.update_all_layer_loaded()

        # Rebublish keys since the tail node could have died and heartbeat task would fail
        self.republish_keys()
    
    def evaluate_takeover_eligibility(self) -> Optional[Dict[str, Any]]:
        """
        Executed by a backup node, this method compare the processing_rate and memory_limit metrics of the backup node 
        and all the nodes in the active chain, to determine if a node takeover should take place.

        Returns: the weaker nodes' info if one is found, otherwise None
        """
        self_info = self._get_self_info()
        self_proc_rate = self_info.get("processing_rate")
        self_device = self_info.get("device")
        self_mem_limit = self_info.get("vram_limit") if self_device == "cuda" else self_info.get("memory_limit")

        head_id = self.dht.get(HEAD_KEY)
        current_node_id = head_id

        # Iterate through the active chain nodes
        while current_node_id:
            server_info = self.get_server_info(current_node_id)
            if not server_info:
                break

            target_successor_data = server_info.get("successor")
            next_node_id = target_successor_data.get("id") if target_successor_data else None

            target_proc_rate = server_info.get("processing_rate")
            target_device = server_info.get("device")
            target_mem_limit = server_info.get("vram_limit") if target_device == "cuda" else server_info.get("memory_limit")

            if not target_proc_rate:
                logger.info(f"No processing rate recorded for node: {current_node_id[:DIGITS_SHOW]}")
                current_node_id = next_node_id
                continue
            
            if (self_proc_rate > target_proc_rate) and (self_mem_limit and target_mem_limit and self_mem_limit >= target_mem_limit):
                return server_info

            current_node_id = next_node_id
        
        return None
                
            


    def republish_keys(self):
        """
        Periodically called to maintain the node's presence on the DHT and check chain integrity.

        :return: The dead successor's data dictionary if detected, else None.
        """
        server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self_info = self._get_self_info()

        # Republish this servers' info
        self.dht.store(server_key, self_info, EXPIRATION_S)

        # If this node is the head
        if self._is_head():
            # Republish the head key
            self.dht.store(HEAD_KEY, self.node_id, EXPIRATION_S)

            # Republish the num_total_layers key
            num_total_layers = self._get_num_total_layers()
            if num_total_layers:
                self.dht.store(TOTAL_LAYERS_KEY, num_total_layers, EXPIRATION_S)

            self.update_all_layer_loaded()

            backup_nodes = self.get_backup_nodes()
            if backup_nodes is not None:
                self.dht.store(BACKUPS_KEY, backup_nodes, EXPIRATION_S)

        # If this node is the tail, republish the tail key
        if self.is_tail():
            self.dht.store(TAIL_KEY, self.node_id, EXPIRATION_S)

        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} republished its keys.")

    def update_layers(self, new_layers: Tuple[int, int]):
        self_info = self._get_self_info()
        self_info["layers"] = new_layers
        server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(server_key, self_info, EXPIRATION_S)

    def get_chain_info(self) -> Optional[Dict[str, Any]]:
        # Global keys
        head_id = self.dht.get(HEAD_KEY)
        tail_id = self.dht.get(TAIL_KEY)
        total_layers = self.dht.get(TOTAL_LAYERS_KEY)
        all_loaded = self.dht.get(ALL_LAYERS_KEY)
        backup_nodes = self.dht.get(BACKUPS_KEY)

        if not head_id:
            return None

        chain_info = {
            HEAD_KEY: head_id,
            TAIL_KEY: tail_id,
            TOTAL_LAYERS_KEY: total_layers,
            ALL_LAYERS_KEY: all_loaded,
        }

        # Traverse chain and print server info
        server_list = []
        current_node_id = head_id

        while current_node_id:
            server_info = self.get_server_info(current_node_id)
            if not server_info:
                break

            server_info["id"] = current_node_id

            server_list.append(server_info)

            successor_data = server_info.get("successor")

            if successor_data:
                current_node_id = successor_data.get("id")
            else:
                current_node_id = None  # End of chain

        chain_info["servers"] = server_list

        # Iterate through the backup_nodes list
        backup_nodes_info = []
        if backup_nodes:
            for id in backup_nodes:
                server_info = self.get_server_info(id)
                if server_info:
                    server_info["id"] = id
                    backup_nodes_info.append(server_info)

        
        chain_info[BACKUPS_KEY] = backup_nodes_info
        return chain_info

    def update_chain_tail(self, node_id: str):
        """Updates the chain_tail key with the given node id"""
        self.dht.store(TAIL_KEY, node_id, EXPIRATION_S)

    def update_all_layer_loaded(self):
        """Checks the layers_loaded subkey in all the server nodes in the chain"""
        # Global keys
        head_id = self.dht.get(HEAD_KEY)

        if not head_id:
            return

        # Traverse chain and check the layers_loaded subkey
        current_node_id = head_id

        while current_node_id:
            server_info = self.get_server_info(current_node_id)
            if not server_info:
                break

            layers_loaded = server_info.get("layers_loaded", False)

            if layers_loaded:
                successor_data = server_info.get("successor")

                if successor_data:
                    current_node_id = successor_data.get("id")
                elif self.node_is_tail(current_node_id):
                    layers = server_info.get("layers")
                    total_layers = self._get_num_total_layers()
                    if layers[1] == total_layers - 1:
                        self.dht.store(ALL_LAYERS_KEY, True, EXPIRATION_S)
                    current_node_id = None  # End of chain

            else:
                self.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
                break

    def update_layers_loaded(self, layers_loaded: bool):
        """Subkey that that shows if all the server's assigned layers have been loaded"""
        self_info = self._get_self_info()
        self_info["layers_loaded"] = layers_loaded

        self_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_key, self_info, EXPIRATION_S)

    def update_layers(self, layers: Tuple[int, int]):
        self_info = self._get_self_info()
        self_info["layers"] = layers

        self_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_key, self_info, EXPIRATION_S)

    def update_device(self, device: str):
        self_info = self._get_self_info()
        self_info["device"] = device

        self_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_key, self_info, EXPIRATION_S)

    def update_memory(self, mem_usage: float, mem_limit: float, avail_mem: float):
        self_info = self._get_self_info()
        self_info["memory_usage"] = mem_usage
        self_info["memory_limit"] = mem_limit
        self_info["available_memory"] = avail_mem

        self_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_key, self_info, EXPIRATION_S)

    def update_vram(self, vram_usage: float, vram_limit: float, avail_vram: float):
        self_info = self._get_self_info()
        self_info["vram_usage"] = vram_usage
        self_info["vram_limit"] = vram_limit
        self_info["available_vram"] = avail_vram

        self_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_key, self_info, EXPIRATION_S)

    def update_successor(self, new_successor_data: str = None):
        self_info = self._get_self_info()
        self_info["successor"] = new_successor_data

        self_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_key, self_info, EXPIRATION_S)

    def update_processing_rate(self, processing_rate: float = 0.0):
        self_info = self._get_self_info()
        self_info["processing_rate"] = processing_rate

        self_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_key, self_info, EXPIRATION_S)

    def update_inference_delay(self, inference_delay: float = 0.0):
        self_info = self._get_self_info()
        self_info["inference_delay"] = inference_delay

        self_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_key, self_info, EXPIRATION_S)

    def update_grpc_overhead(self, grpc_overhead: float = 0.0):
        self_info = self._get_self_info()
        self_info["grpc_overhead"] = grpc_overhead

        self_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_key, self_info, EXPIRATION_S)
    
    def become_backup(self):
        self_info = self._get_self_info()
        self_info["is_backup"] = True
        self_info["successor"] = None
        self_info["layers"] = None
        self_info["layers_loaded"] = False

        self_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self.dht.store(self_key, self_info, EXPIRATION_S)

        backups_list = self.get_backup_nodes()
        backups_list.append(self.node_id)

        self.dht.store(BACKUPS_KEY, backups_list, EXPIRATION_S)

        



