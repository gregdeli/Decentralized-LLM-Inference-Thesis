import logging
from typing import Optional, List, Any
import hivemind

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

    async def start(self):
        """Asynchronously starts the DHT node."""
        logger.info("Starting DHT node...")
        self.dht = await hivemind.DHT(host_maddrs=self.host_maddrs, initial_peers=self.initial_peers, start=True)
        logger.info(f"DHT node started. Visible addresses: {self.dht.get_visible_maddrs()}")

    def get_id(self) -> str:
        """Returns the unique PeerID of this DHT node as a string."""
        if not self.dht:
            raise RuntimeError("DHT has not been started. Call start() first.")
        return self.dht.peer_id.to_string()

    async def store(self, key: str, value: Any, expiration_s: float) -> bool:
        """
        Stores a key-value pair on the DHT with a given expiration time.
        """
        if not self.dht:
            raise RuntimeError("DHT has not been started.")
        expiration_time = hivemind.get_dht_time() + expiration_s
        return await self.dht.store(key, value, expiration_time)

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieves a value from the DHT by its key.
        """
        if not self.dht:
            raise RuntimeError("DHT has not been started.")
        result = await self.dht.get(key, latest=True)
        if result is None:
            return None
        return result.value

    async def shutdown(self):
        """Shuts down the DHT node gracefully."""
        if self.dht:
            logger.info("Shutting down DHT node...")
            await self.dht.shutdown()
            self.dht = None
            logger.info("DHT node shut down.")
