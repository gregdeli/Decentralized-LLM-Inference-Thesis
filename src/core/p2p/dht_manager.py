import logging
from typing import Optional, List, Any, Dict, Union
import hivemind
from functools import partial
from hivemind.dht import DHT, DHTNode
from hivemind.utils.timed_storage import get_dht_time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DHTManager:
    """High-level interface to the hivemind DHT for P2P coordination"""

    def __init__(self, host_maddrs: List[str], initial_peers: Optional[List[str]] = None):
        """
        Initializes and starts a DHT node.

        :param host_maddrs: List of multi-addresses to listen on (e.g., ["/ip4/0.0.0.0/tcp/0"]).
        :param initial_peers: List of multi-addresses of existing peers to connect to. If None, starts a new DHT.
        """
        self.dht: Optional[hivemind.DHT] = None
        self.host_maddrs = host_maddrs
        self.initial_peers = initial_peers

    def start(self):
        """Asynchronously starts the DHT node."""
        logger.info("Starting DHT node...")
        self.dht = hivemind.DHT(host_maddrs=self.host_maddrs, initial_peers=self.initial_peers, start=True)
        logger.info(f"DHT node started. Visible address: {self.dht.get_visible_maddrs()[0]}")

    def get_id(self) -> str:
        """Returns the unique PeerID of this DHT node as a string."""
        if not self.dht:
            raise RuntimeError("DHT has not been started. Call start() first.")
        return self.dht.peer_id.to_string()

    def store(self, key: str, value: Any, expiration_s: float, subkey: Optional[str] = None, return_future: Optional[bool] = False) -> Union[bool,Any]:
        """
        Stores a key-value pair on the DHT with a given expiration time.
        """
        if not self.dht:
            raise RuntimeError("DHT has not been started.")
        
        expiration_time = get_dht_time() + expiration_s


        return self.dht.store(key=key, subkey=subkey, value=value, expiration_time=expiration_time, return_future=return_future)

    @staticmethod
    async def _store_many_task(dht: DHT, node: DHTNode, keys: List[str], subkeys: List[str], values: List[Any], expiration_time: float):
        return await node.store_many(
            keys=keys,
            subkeys=subkeys,
            values=values,
            expiration_time=expiration_time
        )

    def store_many(self, key: str, subkeys_values: Dict[str, Any], expiration_s: float) -> bool:
        """
        Stores multiple subkeys for a given key in a single bulk DHT call
        """
        if not self.dht:
            raise RuntimeError("DHT has not been started.")

        expiration_time = get_dht_time() + expiration_s
        keys = [key] * len(subkeys_values)
        subkeys = list(subkeys_values.keys())
        values = list(subkeys_values.values())

        coro = partial(
            self._store_many_task, keys=keys, subkeys=subkeys, values=values, expiration_time=expiration_time
        )

        result_dict = self.dht.run_coroutine(coro)
        
        # store_many returns a Dict mapping (key, subkey) -> bool
        if not result_dict:
            return False
            
        return all(result_dict.values())
        

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves a value from the DHT by its key.
        """
        if not self.dht:
            raise RuntimeError("DHT has not been started.")
        result = self.dht.get(key, latest=True)
        if result is None:
            return None
        return result.value

    def get_visible_maddrs(self) -> List[str]:
        if self.dht is None:
            raise RuntimeError("DHT has not been started.")
        return self.dht.get_visible_maddrs()

    def shutdown(self):
        """Shuts down the DHT node gracefully."""
        if self.dht:
            logger.info("Shutting down DHT node...")
            self.dht.shutdown()
            self.dht = None
            logger.info("DHT node shut down.")

