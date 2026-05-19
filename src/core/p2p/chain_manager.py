import logging
from typing import Dict, Any, Optional, Tuple, Union, List
import time
import random
from functools import partial
import grpc
from enum import Enum

from hivemind.utils import ValueWithExpiration, get_dht_time
from hivemind.dht import DHTID, DHT, DHTNode

from core.constants import *
from core.p2p.dht_manager import DHTManager

from core.remote import nodeservice_pb2, nodeservice_pb2_grpc
from core.remote.utils import create_grpc_channel

logger = logging.getLogger(__name__)

# Global keys on the DHT
HEAD_KEY = "chain_head"
TAIL_KEY = "chain_tail"
LEADER_KEY = "chain_leader"
TOTAL_LAYERS_KEY = "num_total_layers"
TOTAL_PARAMS_KEY = "num_total_params"
ALL_LAYERS_KEY = "all_layers_loaded"
BACKUPS_KEY = "backup_nodes"
CLIENTS_KEY = "client_nodes"
SERVER_INFO_PREFIX = "server_info_" # "server_info_(node_id)"
NUM_CLIENTS_KEY = "num_clients"

TOKENS_GENERATED_KEY = "global_tokens_generated"
THROUGHPUT_KEY = "chain_throughput_" # "chain_throughput_1", "chain_throughput_2"
THROUGHPUT_VERSION_KEY ="chain_throughput_version" 

# EXPIRATION_S = 30.0
EXPIRATION_S = 7200.0
# HEARTBEAT_INTERVAL_S = EXPIRATION_S / 4.0
HEARTBEAT_INTERVAL_S = 15.0
# HEARTBEAT_INTERVAL_S = 7200.0
THROUGHPUT_EXPIRATION_S = 7200.0
# THROUGHPUT_EXPIRATION_S = 60.0

DIGITS_SHOW = 12

STATUS_KEY = "chain_status"


class ChainStatus(str, Enum):
    READY = "READY"  # all layers loaded is true
    UNREADY = "UNREADY"  # not all layers loaded
    RUNNING = "RUNNING"
    NONE = "NONE"  # while inference is running
    PROFILING = "PROFILING"
    REPAIRING = "REPAIRING"
    REALLOCATING = "REALLOCATING"
    TAKEOVER = "TAKEOVER"


class ChainManager:
    """Manages the creation and discovery of the server inference chain on the DHT."""

    def __init__(self, dht_manager: DHTManager, is_client=False):
        self.dht = dht_manager
        self.node_id = dht_manager.get_id()
        self.is_client = is_client

    def join_chain(
        self, 
        self_info: Dict[str, Any], 
        max_num_layers: int, 
        output_layer_memory_tle,
        num_total_layers: int, 
        num_total_params: int,
    ) -> None:
        """
        Main entry point for a server node to join or form the inference chain.
        It determines if it's the first node or joining an existing chain.

        :param:
          self_info: A dictionary with the server's data (e.g., {'address': 'head-server:5001'}).
          num_layers: The number of Transformers Layers the node can load.
          num_total_layers: The number of total Transformer Layers of the LLM.
        :return: The updated info dictionary for this server.
        """
        # chain_leader = self.get_chain_leader()
        # while chain_leader is None or chain_leader != self.node_id:
        #     sleep_time = random.uniform(0.5, 3.0)
        #     time.sleep(sleep_time)
        #     logger.info(f"A different chain leader is joining or there is no chain leader. Sleeping for {sleep_time} seconds...")
        #     # self.become_chain_leader()
        #     self.dht.store(LEADER_KEY, self.node_id, EXPIRATION_S)

        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} attempting to join the chain...")
        head_id = self.dht.get(HEAD_KEY)

        if head_id is None:
            logger.info("No existing chain found. Forming a new one...")
            self._form_initial_chain(
                self_info, 
                max_num_layers, 
                output_layer_memory_tle,
                num_total_layers, 
                num_total_params,
            )
        else:
            logger.info(
                f"Found existing chain with head {head_id[:DIGITS_SHOW]}. Joining at the tail..."
            )
            self._join_existing_chain(
                self_info, 
                max_num_layers, 
                output_layer_memory_tle,
                num_total_layers,
            )

        logger.info(f"Self Info: {self.get_self_info()}")
        
        # Unset the chain leader so that another node can join
        # self.dht.store(LEADER_KEY, None, EXPIRATION_S)

    def _form_initial_chain(
        self, 
        self_info: Dict[str, Any], 
        max_num_layers: int, 
        output_layer_memory_tle: float,
        num_total_layers: int, 
        num_total_params: int,
    ):
        """Logic for the first server to establish the chain."""
        self.dht.store(HEAD_KEY, self.node_id, EXPIRATION_S)

        self.dht.store(TAIL_KEY, self.node_id, EXPIRATION_S)

        self.dht.store(TOTAL_LAYERS_KEY, num_total_layers, EXPIRATION_S)

        self.dht.store(TOTAL_PARAMS_KEY, num_total_params, EXPIRATION_S)

        self.become_chain_leader()

        # if not self.get_backup_nodes():
        #     self.dht.store(BACKUPS_KEY, [], EXPIRATION_S)
        
        if not self.get_client_nodes():
            self.dht.store(CLIENTS_KEY, [], EXPIRATION_S)

        self.update_chain_status(ChainStatus.UNREADY)

        if not self.dht.get(TOKENS_GENERATED_KEY):
            self.dht.store(TOKENS_GENERATED_KEY, 0, THROUGHPUT_EXPIRATION_S)
        
        if not self.dht.get(THROUGHPUT_VERSION_KEY):
            self.dht.store(THROUGHPUT_VERSION_KEY, 0, THROUGHPUT_EXPIRATION_S)  

        self_info["successor"] = None

        target_num_layers = max_num_layers

        # Don't exceed the the maximum layer index
        end_idx = target_num_layers - 1
        if end_idx >= num_total_layers:
            end_idx = num_total_layers - 1

        self_info["layers"] = (0, end_idx)

        # If all the transformer layers fit on this node
        if end_idx == num_total_layers - 1:
            target_num_layers = end_idx + 1
            remaining_num_layers = max_num_layers - target_num_layers

            # Check if the final output layer can be loaded as well
            if remaining_num_layers >= output_layer_memory_tle:
                self_info["load_output_layer"] = True


        # This node could have been a backup
        if self_info.get("is_backup"):
            self_info["is_backup"] = False

            # Remove the node from the backup_nodes list
            # backup_nodes = self.get_backup_nodes()
            # backup_nodes.remove(self.node_id)
            # self.dht.store(BACKUPS_KEY, backup_nodes, EXPIRATION_S)
            self.dht.store(key=BACKUPS_KEY, subkey=self.node_id, value=False, expiration_s=EXPIRATION_S)

        self._update_server_info(self.node_id, self_info)

        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} is now the head and tail of the chain.")

    def _join_existing_chain(
        self, 
        self_info: Dict[str, Any], 
        max_num_layers: int,
        output_layer_memory_tle: float, 
        num_total_layers: int, 
    ):
        """Logic for a new server to join an existing chain."""
        # Find the current tail
        tail_id = self.dht.get(TAIL_KEY)
        if not tail_id:
            raise RuntimeError(
                "Chain head exists, but tail was not found. The network is in an inconsistent state."
            )

        # Get the current tail's info to update its successor
        tail_info = self.get_server_info(tail_id)
        if not tail_info:
            raise RuntimeError(f"Could not retrieve info for tail node {tail_id}.")

        # Make it a Backup Node, if all layers loaded
        if tail_info.get("output_layer_loaded") or tail_info.get("load_output_layer"):
            logger.warning("All the layers have already been loaded on the the previous tail")
            logger.warning("Setting this node as a backup node...")

            if not self_info.get("is_backup"):
                self.make_node_backup(server_info=self_info)
            return

        # Get the previous tails layers to determine this nodes layer range
        start_idx = tail_info["layers"][1] + 1

        self_layers = None
        end_idx = num_total_layers - 1

        target_num_layers = 0

        # If transformer layers are left unloaded
        if start_idx < num_total_layers:
            end_idx = start_idx + max_num_layers - 1

            if end_idx >= num_total_layers:
                end_idx = num_total_layers - 1
            
            target_num_layers = end_idx - start_idx + 1

            self_layers = (start_idx, end_idx)

        remaining_num_layers = max_num_layers - target_num_layers

        # Check if the final output layer can be loaded 
        if (end_idx == num_total_layers - 1) and remaining_num_layers >= output_layer_memory_tle:  #and (max_num_params >= final_output_params):
            self_info["load_output_layer"] = True

        # If at this point the node is not supposed to load any layers make it a backup
        if not self_layers and not self_info.get("load_output_layer"):
            self.make_node_backup(server_info=self_info)
            return

        # Update the old tail to point to the new server node
        tail_info["successor"] = {"id": self.node_id, "address": self_info["address"]}
        self._update_server_info(tail_id, tail_info)
        logger.info(
            f"Updated previous tail's ({tail_id[:DIGITS_SHOW]}) successor to point to new node {self.node_id[:DIGITS_SHOW]}."
        )

        # Store self info and update the chain_tail value
        self_info["successor"] = None
        self_info["layers"] = self_layers

        # This node could be trying to join at the tail after being a backup
        if self_info.get("is_backup"):
            self_info["is_backup"] = False

            # Remove the node from the backup_nodes list
            # backup_nodes = self.get_backup_nodes()
            # backup_nodes.remove(self.node_id)
            # self.dht.store(BACKUPS_KEY, backup_nodes, EXPIRATION_S)
            self.dht.store(key=BACKUPS_KEY, subkey=self.node_id, value=False, expiration_s=EXPIRATION_S)


        self._update_server_info(self.node_id, self_info)

        # Become the new TAIL
        self.dht.store(TAIL_KEY, self.node_id, EXPIRATION_S)

        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} has joined as the new tail.")

        logger.info(f"Previous Tail Info: {tail_info}")

    @staticmethod
    async def _find_random_server(
        dht: DHT, 
        node: DHTNode, 
        client_nodes: List[str], 
        max_attempts: int
    ) -> Optional[str]:
        """
        Static coroutine executed inside the DHT background process.
        """
        for _ in range(max_attempts):
            random_target = DHTID.generate()

            result = await node.find_nearest_nodes(
                queries=[random_target],
                k_nearest=5,
                exclude_self=True
            )

            if random_target in result and result[random_target]:
                # Extract peer IDs from the result dictionary
                nearest_peers = list(result[random_target].values())

                for peer_id in nearest_peers:
                    peer_id_str = peer_id.to_string()

                    if peer_id_str not in client_nodes:
                        return peer_id_str
        return None

    # ---- Getters ----

    # ----- Global Keys -----

    def get_chain_leader(self) -> Optional[str]:
        return self.dht.get(LEADER_KEY)

    def get_head_id(self) -> str:
        return self.dht.get(TAIL_KEY)

    def get_tail_id(self) -> str:
        return self.dht.get(TAIL_KEY)

    def get_chain_status(self) -> Optional[ChainStatus]:
        status_value = self.dht.get(STATUS_KEY)
        if status_value is not None:
            try:
                return ChainStatus(status_value)
            except ValueError:
                logger.error(f"Unknown status '{status_value}' found on DHT.")
                return None
        return ChainStatus("NONE")

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

        # logger.info(f"Found head server {head_id[:DIGITS_SHOW]} with info: {head_info}")

        return head_info
    
    def _get_num_total_layers(self) -> int:
        num_total_layers = self.dht.get(TOTAL_LAYERS_KEY)
        if not num_total_layers:
            raise RuntimeError(f"Could not retrieve the total number of transformer layers.")
        return num_total_layers
    
    def _get_num_total_params(self) -> int:
        num_total_params = self.dht.get(TOTAL_PARAMS_KEY)
        if not num_total_params:
            raise RuntimeError(f"Could not retrieve the total number of parameters.")
        return num_total_params
    
    def get_all_layers_loaded(self) -> Optional[bool]:
        return self.dht.get(ALL_LAYERS_KEY)

    def get_backup_nodes(self) -> Optional[List[str]]:
        backups_dict = self.dht.get(BACKUPS_KEY)
        if backups_dict is None:
            return None

        backups_list = []
        for backup_id, is_backup in backups_dict.items():
            if is_backup.value:
                backups_list.append(backup_id)
        
        return backups_list

    def get_client_nodes(self) -> Optional[List[str]]:
        return self.dht.get(CLIENTS_KEY)

    def get_chain_throughput(self) -> Optional[Dict[float, ValueWithExpiration]]:
        version = self.dht.get(THROUGHPUT_VERSION_KEY)
        key = f"{THROUGHPUT_KEY}{version}"
        chain_throughput = self.dht.get(key)
        return chain_throughput
    
    def get_random_server_node_id(self, max_attempts: int = 5) -> Optional[str]:
        """
        Selects a random server node (non-client) from the DHT
        """
        client_nodes = self.get_client_nodes()
        if client_nodes is None:
            client_nodes = []

        coro = partial(
            self._find_random_server,
            client_nodes=client_nodes,
            max_attempts=max_attempts
        )
        
        random_server_id = self.dht.dht.run_coroutine(coro)

        return random_server_id
        

    # ----- Server Info Subkeys -----

    def get_self_info(self) -> Dict[str, Any]:
        """Get the server info dict for this node from the DHT"""
        self_info = self.dht.get(f"{SERVER_INFO_PREFIX}{self.node_id}")
        return self_info

    def get_successor_data(self) -> Optional[Dict[str, Any]]:
        """Get the successor data for this node from the DHT"""
        self_info = self.get_self_info()
        return self_info.get("successor")

    def get_successor_info(self) -> Optional[Dict[str, Any]]:
        """Get this nodes' successor info dict from the DHT"""
        succ_data = self.get_successor_data()
        succ_info = None
        if succ_data:
            succ_info = self.get_server_info(succ_data.get("id"))

        return succ_info

    def get_server_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.dht.get(f"{SERVER_INFO_PREFIX}{node_id}")

    def get_layers(self) -> Optional[Tuple[int, int]]:
        """Get the layers tuple for this node from the DHT"""
        self_info = self.get_self_info()
        return self_info.get("layers")

    def get_load_output_layer(self) -> bool:
        self_info = self.get_self_info()
        return self_info.get("load_output_layer", False)

    def get_output_layer_loaded(self) -> bool:
        self_info = self.get_self_info()
        return self_info.get("output_layer_loaded", False)

    def get_successor_address(self, attempts: int = 5) -> Optional[str]:
        if self.is_tail() or self.is_backup():
            return None

        for attempt in range(attempts):
            logger.info(f"Attempting to get successor address info... (Attempt {attempt + 1})")
            self_info = self.get_self_info()
            successor_data = self_info.get("successor")
            if successor_data:
                return successor_data.get("address")
            else:
                logger.info(f"Retrying in 2 seconds...")
                time.sleep(2)

        return None

    def get_chain_info(self) -> Optional[Dict[str, Any]]:
        """
        Collect and return chain topology and node state from the DHT.

        Returns:
            A dictionary with keys:
            - HEAD_KEY
            - TAIL_KEY
            - TOTAL_LAYERS_KEY
            - ALL_LAYERS_KEY
            - "servers": list of active chained server info
            - BACKUPS_KEY: list of backup server info
            or None if no chain head is available.
        """
        # Global keys
        head_id = self.dht.get(HEAD_KEY)
        tail_id = self.dht.get(TAIL_KEY)
        total_layers = self.dht.get(TOTAL_LAYERS_KEY)
        total_params = self.dht.get(TOTAL_PARAMS_KEY)
        all_loaded = self.dht.get(ALL_LAYERS_KEY)
        backup_nodes = self.get_backup_nodes()
        current_status = self.get_chain_status()
        current_num_clients = self.dht.get(NUM_CLIENTS_KEY)
        client_nodes = self.get_client_nodes()

        # if not head_id:
        #     return None

        chain_info = {
            HEAD_KEY: head_id,
            TAIL_KEY: tail_id,
            TOTAL_LAYERS_KEY: total_layers,
            TOTAL_PARAMS_KEY: total_params,
            ALL_LAYERS_KEY: all_loaded,
            STATUS_KEY: current_status,
            NUM_CLIENTS_KEY: current_num_clients,
            CLIENTS_KEY: client_nodes
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

    def gather_total_rate(self) -> float:
        """Add the partial processing rates of all server nodes"""
        head_id = self.dht.get(HEAD_KEY)

        total_rate = 0
        current_node_id = head_id
        while current_node_id:
            server_info = self.get_server_info(current_node_id)
            if not server_info:
                break

            partial_rate = server_info.get("processing_rate")
            total_rate += partial_rate

            successor_data = server_info.get("successor")
            if successor_data:
                current_node_id = successor_data.get("id")
            else:
                current_node_id = None  # End of chain

        return total_rate

    def gather_total_memory(
            self,
            layer_mem_size: float,
            output_layer_mem_size: float,
        ) -> float:
        """Add the availbale memory sizes of all active nodes"""
        head_id = self.dht.get(HEAD_KEY)

        total_memory = 0
        current_node_id = head_id
        while current_node_id:
            server_info = self.get_server_info(current_node_id)
            if not server_info:
                break

            available_vram = server_info.get("availbale_vram")
            available_memory = server_info.get("available_memory")
            available = available_vram if available_vram is not None else available_memory

            # vram_usage = server_info.get("vram_usage")
            # memory_usage = server_info.get("memory_usage")
            # in_use = vram_usage if vram_usage is not None else memory_usage
            
            # Calculate the memory used by any layers that are currently loaded
            layers = server_info.get("layers")
            layers_loaded = server_info.get("layers_loaded", False)
            output_layer_loaded = server_info.get("output_layer_loaded", False)

            current_num_layers = 0
            if layers is not None and layers_loaded:
                current_num_layers = layers[1] - layers[0] + 1
            
            in_use = current_num_layers * layer_mem_size
            if output_layer_loaded:
                in_use += output_layer_mem_size

            total_memory += in_use + available - RESERVED_MEM_MB

            successor_data = server_info.get("successor")
            if successor_data:
                current_node_id = successor_data.get("id")
            else:
                current_node_id = None  # End of chain

        return total_memory
    
    def get_tail_output_temporal_tle(self):
        tail_id = self.get_tail_id()
        tail_info = self.get_server_info(tail_id)

        return tail_info.get("output_layer_temporal_tle")

    def is_head(self) -> Optional[bool]:
        """Check if this node is the head of the server chain"""
        head_id = self.dht.get(HEAD_KEY)
        if not head_id:
            # If the head_id is None then it just not have been republished, check layers
            self_info = self.get_self_info()
            my_layers = self_info.get("layers")
            layers_loaded = self_info.get("layers_loaded", False)
            if my_layers and my_layers[0] == 0 and layers_loaded and self.get_chain_status() == ChainStatus.UNREADY:
                return True
            
            return None
        return head_id == self.node_id

    def is_tail(self) -> Optional[bool]:
        """Checks if this node is the TAIL"""
        tail_id = self.dht.get(TAIL_KEY)
        if not tail_id:
            return None

        if tail_id == self.node_id:
            return True

        return False

    def is_backup(self) -> bool:
        """Checks if this node is a backup node"""
        self_info = self.get_self_info()
        return bool(self_info.get("is_backup", False))

    def node_is_tail(self, node_id: str) -> bool:
        """Checks if a specific node is the tail"""
        tail_id = self.dht.get(TAIL_KEY)
        if not tail_id:
            raise RuntimeError("Tail server not found")

        if tail_id == node_id:
            return True

        return False

    def node_is_head(self, node_id: str) -> bool:
        """Checks if a specific node is the head"""
        head_id = self.dht.get(HEAD_KEY)
        if not head_id:
            raise RuntimeError("Head server not found")

        if head_id == node_id:
            return True

        return False

    def node_is_backup(self, node_id: str) -> bool:
        """Checks if a specific node is a backup"""
        server_info = self.get_server_info(node_id)
        return bool(server_info.get("is_backup", False))

    def repair(
        self,
        new_layers: Tuple[int, int],
        replacement_info: Dict[str, Any],
        replacee_info: Dict[str, Any],
        make_replacee_backup: Optional[bool] = False,
        replacement_load_output_layer: Optional[bool] = False,
        replacee_was_head: Optional[bool] = False,
        replacee_was_tail: Optional[bool] = False,
        replacee_pred_info: Optional[Dict[str, Any]] = None,
    ):
        logger.info(f"Repairing chain on DHT...")
        replacement_node_id = replacement_info.get("id")
        replacee_node_id = replacee_info.get("id")

        # Update the replacement node's layers and successor
        replacement_info["layers"] = new_layers

        replacement_info["load_output_layer"] = replacement_load_output_layer

        # If the replacee's successor is the replacement node,
        # then the replacement's successor stays the same
        # Otherwise the replacement's successor becomes the replacee's successor
        original_replacement_succ = replacement_info.get("successor")
        replacee_successor_data = replacee_info.get("successor")
        replacement_info["successor"] = replacee_successor_data

        if replacee_successor_data and replacee_successor_data.get("id") == replacement_node_id:
            replacement_info["successor"] = original_replacement_succ

        if replacee_was_head:
            self.dht.store(HEAD_KEY, replacement_node_id, EXPIRATION_S)

        if replacee_was_tail:
            self.dht.store(TAIL_KEY, replacement_node_id, EXPIRATION_S)

        # If the replacement node was a backup make it active
        if replacement_info.get("is_backup"):
            logger.info(f"With replacement Backup Node: {replacement_node_id}")
            replacement_info["is_backup"] = False
            self.remove_node_from_backups(replacement_info.get("id"))

        # Make the replacee a backup (Opportunistic Takeover and Reallocation Takeover)
        if make_replacee_backup:
            logger.info("Setting the replacee node as a backup...")
            self.make_node_backup(replacee_node_id)

        # Update the replacee's predecessor's successor to point to the replacement node
        if replacee_pred_info:
            replacee_pred_info["successor"] = {
                "id": replacement_node_id,
                "address": replacement_info.get("address"),
            }
            logger.info(
                "Updating the replacee's predeseccor's successor to point to the replacement..."
            )
            self._update_server_info(replacee_pred_info.get("id"), replacee_pred_info)

        self._update_server_info(replacement_node_id, replacement_info)

        self.update_all_layers_loaded()

        if not self.is_client:
            self.republish_keys()

    def evaluate_takeover_eligibility(
        self,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Executed by a backup node, this method compare the processing_rate and memory_limit metrics of the backup node
        and all the nodes in the active chain, to determine if a node takeover should take place.

        Returns: the first weaker nodes' info if one is found along with its predecessors' info, otherwise (None, None)
        """
        self_info = self.get_self_info()
        self_proc_rate = self_info.get("processing_rate")
        self_device = self_info.get("device")
        self_mem_limit = (
            self_info.get("vram_limit") if self_device == "cuda" else self_info.get("memory_limit")
        )

        head_id = self.dht.get(HEAD_KEY)
        current_node_id = head_id

        # Iterate through the active chain nodes
        predecessor_info = None
        while current_node_id:
            server_info = self.get_server_info(current_node_id)
            if not server_info:
                break

            target_successor_data = server_info.get("successor")
            next_node_id = target_successor_data.get("id") if target_successor_data else None

            target_proc_rate = server_info.get("processing_rate")
            target_device = server_info.get("device")
            target_mem_limit = (
                server_info.get("vram_limit")
                if target_device == "cuda"
                else server_info.get("memory_limit")
            )

            if not target_proc_rate:
                logger.info(
                    f"No processing rate recorded for node: {current_node_id[:DIGITS_SHOW]}"
                )
                current_node_id = next_node_id
                continue

            if (self_proc_rate > target_proc_rate * OPPRTUNISTIC_TAKEOVER_MULT_THRESHOLD) and (
                self_mem_limit and target_mem_limit and self_mem_limit >= target_mem_limit
            ):
                return server_info, predecessor_info

            # Assign predecessor_info and go to the next node id
            predecessor_info = server_info.copy()
            current_node_id = next_node_id

        return (None, None)

    def republish_keys(self):
        """
        Periodically called to maintain the node's presence on the DHT and check chain integrity.
        """
        server_key = f"{SERVER_INFO_PREFIX}{self.node_id}"
        self_info = self.get_self_info()

        # Republish this servers' info
        self.dht.store(server_key, self_info, EXPIRATION_S)

        # Set the chain leader 
        # if self.dht.get(LEADER_KEY) is None:
        self.become_chain_leader()

        # Republish the num_total_layers key
        num_total_layers = self._get_num_total_layers()
        if num_total_layers:
            self.dht.store(TOTAL_LAYERS_KEY, num_total_layers, EXPIRATION_S)
        
        # Republish the num_total_params key
        num_total_params = self._get_num_total_params()
        if num_total_params:
            self.dht.store(TOTAL_PARAMS_KEY, num_total_params, EXPIRATION_S)
        
        # Republish the num_clients key
        # num_clients = self.dht.get(NUM_CLIENTS_KEY)
        # if num_clients:
        #     self.dht.store(NUM_CLIENTS_KEY, num_clients, EXPIRATION_S)

        # Republish the client_nodes list
        # client_nodes = self.get_client_nodes()
        # if client_nodes:
        #     self.dht.store(CLIENTS_KEY, client_nodes, EXPIRATION_S)

        # Update the all_layers_loaded and chain_status keys
        self.update_all_layers_loaded()

        # Republish chain_status
        current_status = self.get_chain_status()
        if current_status:
            if current_status in (ChainStatus.UNREADY, ChainStatus.READY, ChainStatus.NONE) and self.get_all_layers_loaded():
                self.update_chain_status(ChainStatus.READY)
            # else:
            #     self.update_chain_status(current_status)

        # Republish backup nodes 
        # backup_nodes = self.get_backup_nodes()
        # if backup_nodes is not None:
        #     self.dht.store(BACKUPS_KEY, backup_nodes, EXPIRATION_S)
        if self.is_backup():
            self.dht.store(key=BACKUPS_KEY, subkey=self.node_id, value=True, expiration_s=EXPIRATION_S)

        # If this node is the head
        if self.is_head():
            # Republish the head key
            self.dht.store(HEAD_KEY, self.node_id, EXPIRATION_S)
            logger.info("HEAD Node Key republished.")

        # If this node is the tail, republish the tail key
        if self.is_tail():
            self.dht.store(TAIL_KEY, self.node_id, EXPIRATION_S)
            logger.info("HEAD Node Key republished.")

        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} republished its keys.")

    
    def republish_client_keys(self):
        # Republish the num_clients key
        num_clients = self.dht.get(NUM_CLIENTS_KEY)
        if num_clients:
            self.dht.store(NUM_CLIENTS_KEY, num_clients, EXPIRATION_S)

        # Republish the client_nodes list
        client_nodes = self.get_client_nodes()
        if client_nodes:
            self.dht.store(CLIENTS_KEY, client_nodes, EXPIRATION_S)
        
        logger.info(f"Node {self.node_id[:DIGITS_SHOW]} republished the client keys.")


    # ---- Global key update methods ----

    def increment_num_clients(self):
        """Called by a client when in joins the DHT"""
        current_num_clients = self.dht.get(NUM_CLIENTS_KEY)
        if current_num_clients:
            new_num_clients = current_num_clients + 1
            self.dht.store(NUM_CLIENTS_KEY, new_num_clients, EXPIRATION_S)
            logger.info(f"Num Clientes updated to {new_num_clients}...")
        else:
            # This is the first client
            self.dht.store(NUM_CLIENTS_KEY, 1, EXPIRATION_S)

    def decrement_num_clients(self):
        """Called by a client when in leaves the DHT"""
        current_num_clients = self.dht.get(NUM_CLIENTS_KEY)
        if current_num_clients:
            new_num_clients = current_num_clients - 1
            self.dht.store(NUM_CLIENTS_KEY, new_num_clients, EXPIRATION_S)
            logger.info(f"Num Clientes updated to {new_num_clients}...")
        else:
            # This is the first client
            self.dht.store(NUM_CLIENTS_KEY, 1, EXPIRATION_S)
    
    def add_node_to_clients(self, client_node_id: str):
        """Appends a client_node id to the client_nodes list"""
        client_nodes = self.get_client_nodes()

        if client_nodes is None:
            client_nodes = []

        if client_node_id not in client_nodes:
            client_nodes.append(client_node_id)
        
        self.dht.store(CLIENTS_KEY, client_nodes, EXPIRATION_S)
    
    def become_chain_leader(self, force: bool = False):
        logger.info("Attempting to become the chain leader...")
        if self.get_chain_leader() is None or force:
            self.dht.store(LEADER_KEY, self.node_id, EXPIRATION_S)
            logger.info(f"Node {self.node_id[:DIGITS_SHOW]} is now the leader of the chain.")
            return 
        logger.info(f"A node is already the chain leader.")

    def update_chain_status(self, status: ChainStatus):
        logger.info(f"Attempting to update chain status to {status.value}...")
        leader_id = self.dht.get(LEADER_KEY)
        if leader_id is not None and leader_id == self.node_id:
            self.dht.store(STATUS_KEY, status.value, 2 * EXPIRATION_S)
            logger.info(f"Chain status updated to: {status.name}")
            return
        elif leader_id is None:
            logger.info("No chain leader found.")
            if not self.is_client:
                self.become_chain_leader(force=True)
                self.update_chain_status(status)
            return            
        else:
            # If a different node is the chain leader send an update chain status request
            logger.info("A different node is the chain leader. Sending a request to update the chain status...")
            leader_info = self.get_server_info(leader_id)
            leader_addr = leader_info.get("address")
            
            if leader_info is not None and leader_addr is not None:
                try:
                    channel = create_grpc_channel(leader_addr)
                    leader_stub = nodeservice_pb2_grpc.NodeServiceStub(channel)

                    request = nodeservice_pb2.UpdateChainStatusRequest(status=status.value)
                    leader_stub.UpdateChainStatus(request, timeout=5)
                    return
                except grpc.RpcError as e:
                    if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                        logger.warning("Leader failure during Update Chain Status request.")
            else:
                logger.info("Leader address not foud.")
            
            if not self.is_client:
                logger.info("Becoming chain leader...")
                self.become_chain_leader(force=True)
                self.update_chain_status(status)
            else:
                server_id = self.get_random_server_node_id()
                self.dht.store(LEADER_KEY, server_id, EXPIRATION_S)
                self.update_chain_status(status)

    def update_chain_tail(self, node_id: str):
        """Updates the chain_tail key with the given node id"""
        self.dht.store(TAIL_KEY, node_id, EXPIRATION_S)

    def update_chain_head(self, node_id: str):
        self.dht.store(HEAD_KEY, node_id, EXPIRATION_S)

    def update_all_layers_loaded(self):
        """Checks the layers_loaded subkey in all the server nodes in the chain"""
        # Global keys
        head_id = self.dht.get(HEAD_KEY)

        if not head_id:
            self.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
            # self.update_chain_status(ChainStatus.UNREADY)
            return

        # Traverse chain and check the layers_loaded subkey
        current_node_id = head_id

        while current_node_id:
            server_info = self.get_server_info(current_node_id)
            if not server_info:
                break
            
            # Check if the HEAD is assigned the first layer
            if self.node_is_head(current_node_id):
                head_layers = server_info.get("layers")
                if head_layers[0] != 0:
                    self.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
                    return

            layers_loaded = server_info.get("layers_loaded", False)
            output_layer_loaded = server_info.get("output_layer_loaded", False)

            if layers_loaded or output_layer_loaded:
                successor_data = server_info.get("successor")

                if successor_data:
                    current_node_id = successor_data.get("id")
                
                elif self.node_is_tail(current_node_id):
                    # layers = server_info.get("layers")
                    # total_layers = self._get_num_total_layers()
                    if output_layer_loaded:
                        self.dht.store(ALL_LAYERS_KEY, True, EXPIRATION_S)

                        # current_chain_status = self.get_chain_status()
                        # if current_chain_status in (ChainStatus.UNREADY, ChainStatus.RUNNING):
                        #     self.update_chain_status(ChainStatus.READY)
                        # else:
                        #     self.update_chain_status(current_chain_status)
                        break

                    # If the tail doesn't have the final layer loaded
                    self.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
                    # self.update_chain_status(ChainStatus.UNREADY)
                    break  # current_node_id = None  # End of chain
            else:
                self.dht.store(ALL_LAYERS_KEY, False, EXPIRATION_S)
                # self.update_chain_status(ChainStatus.UNREADY)
                break

    def update_chain_throughput(self):
        """Increments the global tokens generated and adds a timestamp subkey to the chain_throughput key"""
        tokens_generated = self.dht.get(TOKENS_GENERATED_KEY)
        if tokens_generated is None:
            tokens_generated = 0
        tokens_generated += 1
        self.dht.store(TOKENS_GENERATED_KEY, tokens_generated, THROUGHPUT_EXPIRATION_S)

        version = self.dht.get(THROUGHPUT_VERSION_KEY)
        key = f"{THROUGHPUT_KEY}{version}"
        self.dht.store(key=key, subkey=time.perf_counter(), value=tokens_generated, expiration_s=THROUGHPUT_EXPIRATION_S)
    
    # @staticmethod
    # async def _store_many(dht_daemon, dht_node: DHTNode, keys, values, expiration_time, subkeys):
    #     # This runs inside the background DHT process
    #     await dht_node.store_many(
    #         keys=keys,
    #         values=values,
    #         expiration_time=expiration_time,
    #         subkeys=subkeys
    #     )
    #     return True

    def clear_chain_throughput(self):
        self.dht.store(TOKENS_GENERATED_KEY, 0, THROUGHPUT_EXPIRATION_S)

        current_version = self.dht.get(THROUGHPUT_VERSION_KEY)
        self.dht.store(THROUGHPUT_VERSION_KEY, current_version + 1, THROUGHPUT_EXPIRATION_S)

        # throughput_dict = self.dht.get(THROUGHPUT_KEY)
        # if throughput_dict:
        #     timestamps = list(throughput_dict.keys())

        #     keys=[THROUGHPUT_KEY] * len(timestamps)
        #     values = [0] * len(timestamps)
        #     expiration_time = get_dht_time()

        #     subkeys=timestamps

        #     coro_partial = partial(
        #         self._store_many, 
        #         keys=keys, 
        #         values=values, 
        #         expiration_time=expiration_time, 
        #         subkeys=subkeys
        # )

        #     self.dht.dht.run_coroutine(coro_partial)
        
        # throughput_dict = self.dht.get(THROUGHPUT_KEY)
        # pass

    # ---- Server info subkey update methods ----

    def _update_server_info(self, node_id: int, updated_server_info: Dict[str, Any]):
        """Helper that updates the server_info_(node_id) key of a specific node."""
        server_key = f"{SERVER_INFO_PREFIX}{node_id}"
        self.dht.store(server_key, updated_server_info, EXPIRATION_S)

    def update_layers(self, new_layers: Tuple[int, int]):
        self_info = self.get_self_info()
        self_info["layers"] = new_layers
        self._update_server_info(self.node_id, self_info)

    def update_layers_loaded(self, layers_loaded: bool):
        """Subkey that that shows if all the server's assigned layers have been loaded"""
        self_info = self.get_self_info()
        self_info["layers_loaded"] = layers_loaded
        self._update_server_info(self.node_id, self_info)

    def update_load_output_layer(self, load_output_layer: bool):
        self_info = self.get_self_info()
        self_info["load_output_layer"] = load_output_layer
        self._update_server_info(self.node_id, self_info)

    def update_output_layer_loaded(self, output_layer_loaded: bool):
        self_info = self.get_self_info()
        self_info["output_layer_loaded"] = output_layer_loaded
        self._update_server_info(self.node_id, self_info)

    def update_output_layer_temporal_tle(self, output_layer_temporal_tle: float):
        self_info = self.get_self_info()
        self_info["output_layer_temporal_tle"] = output_layer_temporal_tle
        self._update_server_info(self.node_id, self_info)
    
    def update_output_layer_memory_tle(self, output_layer_memory_tle: float):
        self_info = self.get_self_info()
        self_info["output_layer_memory_tle"] = output_layer_memory_tle
        self._update_server_info(self.node_id, self_info)

    def update_device(self, device: str):
        self_info = self.get_self_info()
        self_info["device"] = device
        self._update_server_info(self.node_id, self_info)

    def update_memory(self, mem_usage: float, mem_limit: float, avail_mem: float):
        self_info = self.get_self_info()
        self_info["memory_usage"] = mem_usage
        self_info["memory_limit"] = mem_limit
        self_info["available_memory"] = avail_mem
        self._update_server_info(self.node_id, self_info)

    def update_vram(self, vram_usage: float, vram_limit: float, avail_vram: float):
        self_info = self.get_self_info()
        self_info["vram_usage"] = vram_usage
        self_info["vram_limit"] = vram_limit
        self_info["available_vram"] = avail_vram
        self._update_server_info(self.node_id, self_info)

    def update_kv_cache_size(self, kv_cache_memories: Dict[str, float]):
        """kv_cache_memories = {"client-addr-1": 100.23 MB, "client-addr-2": 100.23 MB}"""
        self_info = self.get_self_info()
        self_info["kv_cache_memories"] = kv_cache_memories
        self._update_server_info(self.node_id, self_info)

    def update_successor(self, new_successor_data: str = None):
        self_info = self.get_self_info()
        self_info["successor"] = new_successor_data
        self._update_server_info(self.node_id, self_info)

    def update_processing_rate(self, processing_rate: int = 0):
        self_info = self.get_self_info()
        self_info["processing_rate"] = processing_rate
        self._update_server_info(self.node_id, self_info)

    def update_inference_delay(self, inference_delay: float = 0.0):
        self_info = self.get_self_info()
        self_info["inference_delay"] = inference_delay
        self._update_server_info(self.node_id, self_info)

    def update_grpc_overhead(self, grpc_overhead: float = 0.0):
        self_info = self.get_self_info()
        self_info["grpc_overhead"] = grpc_overhead
        self._update_server_info(self.node_id, self_info)

    def make_node_backup(
        self, node_id: Optional[str] = None, server_info: Optional[Dict[str, Any]] = None
    ):
        """Turn an active server node into a backup."""
        if node_id:
            server_info = self.get_server_info(node_id)
        else:
            node_id = server_info.get("id")
        
        server_info["is_backup"] = True
        server_info["successor"] = None
        server_info["layers"] = None
        server_info["layers_loaded"] = False
        server_info["load_output_layer"] = False
        server_info["output_layer_loaded"] = False

        self._update_server_info(node_id, server_info)

        # backups_list = self.get_backup_nodes()
        # backups_list.append(node_id)
        # self.dht.store(BACKUPS_KEY, backups_list, EXPIRATION_S)
        self.dht.store(key=BACKUPS_KEY, subkey=node_id, value=True, expiration_s=EXPIRATION_S)

        logger.info(
            f"Node: {node_id[:DIGITS_SHOW]} was removed from the active chain and became a backup."
        )

    def remove_node_from_backups(self, backup_node_id: str):
        """If node_id was a backup, remove it from the DHT backups list"""
        backup_info = self.get_server_info(backup_node_id)
        if not backup_info:
            return

        logger.info(f"Making Node {backup_node_id[:DIGITS_SHOW]} active...")
        if self.node_is_backup(backup_node_id):
            # backup_info["is_backup"] = False
            # self._update_server_info(backup_node_id, backup_info)

            # Remove the node from the backup_nodes list
            # backup_nodes = self.get_backup_nodes()
            # backup_nodes.remove(backup_node_id)
            # self.dht.store(BACKUPS_KEY, backup_nodes, EXPIRATION_S)
            self.dht.store(key=BACKUPS_KEY, subkey=backup_node_id, value=False, expiration_s=EXPIRATION_S)
            return

        logger.info(f"Node: {backup_node_id[:DIGITS_SHOW]} was not a backup")

    def remove_node_from_clients(self, client_node_id: str):
        """Remove a client node from the client_nodes list"""

        logger.info(f"Removing Client {client_node_id[:DIGITS_SHOW]} from client_nodes list...")

        # Remove the node from the client_nodes list
        client_nodes = self.get_client_nodes()
        if client_node_id in client_nodes:
            client_nodes.remove(client_node_id)
            self.dht.store(CLIENTS_KEY, client_nodes, EXPIRATION_S)
            return

        logger.info(f"Node: {client_node_id[:DIGITS_SHOW]} was not in the client_nodes list.")
