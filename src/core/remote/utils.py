import logging
import time
import grpc
import socket

from core.remote import nodeservice_pb2_grpc, nodeservice_pb2

from core.constants import *

logger = logging.getLogger(__name__)

"""-------------- GRPC Connection Utils --------------"""

class CustomRpcError(grpc.RpcError, grpc.Call):
    """
    A custom gRPC error that implements both RpcError and Call interfaces.
    """
    def __init__(self, code, details=""):
        super().__init__(details)
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details
    

def get_free_port() -> int:
    """Finds and returns an available network port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def get_bootstrap_peer_address(address: str, attempts: int = 5) -> str | None:
    """
    Connects to a bootstrap gRPC server, retrieves its P2P multiaddress, and closes the connection.
    """
    for attempt in range(attempts):
        try:
            logger.info(
                f"Attempting to discover bootstrap peer at {address} (Attempt {attempt + 1})..."
            )
            with grpc.insecure_channel(address) as channel:
                stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
                response = stub.GetPeerMultiaddr(nodeservice_pb2.Empty(), timeout=GRPC_REQUEST_TIMEOUT)

                if response and response.multiaddr:
                    return response.multiaddr

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                logger.info("Bootstrap node not ready yet, retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"An unexpected gRPC error occurred while contacting bootstrap node: {e}")
                return None

    logger.error(
        f"FATAL: Could not connect to bootstrap node at {address} after {attempts} attempts."
    )
    return None


"""-------------- UDP BOOTSTRAP NODE ADDRESS DISCOVERY --------------"""

UDP_PORT = 9999


def discover_bootstrap_node_address(timeout: float = 5.0, is_client: bool = False) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)

    while True:
        try:
            # Broadcast the discovery message
            sock.sendto(b"DISCOVER_BOOTSTRAP", ("255.255.255.255", UDP_PORT))

            # Wait for the response
            data, addr = sock.recvfrom(1024)
            bootstrap_addr = data.decode()
            logger.info(f"Discovered bootstrap node: {bootstrap_addr}")

            return bootstrap_addr
        except socket.timeout:
            logger.error("No bootstrap node found.")

            # If this is a server node and no bootstrap is found then
            # the DHT hasn't been created and we should return.
            # Otherwise if this is a client it should keep retrying
            if not is_client:
                break


def create_grpc_channel(address: str) -> grpc.Channel:
    channel = grpc.insecure_channel(
        address,
        # options=[
        #     ("grpc.default_compression_algorithm", grpc.Compression.Deflate),  # Deflate, Gzip
        #     ("grpc.default_compression_level", 3),  # 0: None, 1: Low, 2: Med, 3: High
        # ],
    )
    return channel
