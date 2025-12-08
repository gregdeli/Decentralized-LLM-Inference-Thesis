import logging
import time
import grpc
import socket

from core.remote import nodeservice_pb2_grpc, nodeservice_pb2

logger = logging.getLogger(__name__)

"""-------------- GRPC Connection Utils --------------"""


def get_bootstrap_peer_address(address: str, attempts: int = 5) -> str | None:
    """
    Connects to a bootstrap gRPC server, retrieves its P2P multiaddress, and closes the connection.
    """
    for attempt in range(attempts):
        try:
            logger.info(f"Attempting to discover bootstrap peer at {address} (Attempt {attempt + 1})...")
            with grpc.insecure_channel(address) as channel:
                stub = nodeservice_pb2_grpc.NodeServiceStub(channel)
                response = stub.GetPeerMultiaddr(nodeservice_pb2.Empty(), timeout=5)

                if response and response.multiaddr:
                    return response.multiaddr

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                logger.info("Bootstrap node not ready yet, retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"An unexpected gRPC error occurred while contacting bootstrap node: {e}")
                return None

    logger.error(f"FATAL: Could not connect to bootstrap node at {address} after {attempts} attempts.")
    return None


"""-------------- UDP BOOTSTRAP NODE ADDRESS DISCOVERY --------------"""

UDP_PORT = 9999


def discover_bootstrap_node_address(timeout: float = 5.0) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)

    sock.sendto(b"DISCOVER_BOOTSTRAP", ("255.255.255.255", UDP_PORT))

    try:
        data, addr = sock.recvfrom(1024)
        bootstrap_addr = data.decode()
        logger.info(f"Discovered bootstrap node: {bootstrap_addr}")
        return bootstrap_addr
    except socket.timeout:
        logger.error("No bootstrap node found")
