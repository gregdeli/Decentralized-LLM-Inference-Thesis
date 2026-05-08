import asyncio
import os
import grpc
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import List, Dict, Any

from nicegui import ui, run, app
from client.client import Client
from core.remote.utils import (
    get_bootstrap_peer_address,
    discover_bootstrap_node_address,
    get_ip_address,
    get_free_port
)
from core.p2p.chain_manager import (
    ChainStatus,
    HEAD_KEY,
    TAIL_KEY,
    TOTAL_LAYERS_KEY,
    TOTAL_PARAMS_KEY,
    ALL_LAYERS_KEY,
    BACKUPS_KEY,
    STATUS_KEY,
    NUM_CLIENTS_KEY,
    DIGITS_SHOW,
)

logger = logging.getLogger(__name__)


class UIReferences:
    def __init__(self):
        self.global_labels = {}  # Global info labels
        self.node_labels = {}  # Label references for each node id
        self.last_topology = None  # Signature of the current chain structure


class AppState:
    def __init__(self):
        self.client: Client = None
        self.is_generating = False
        self.ui = UIReferences()
        self.max_new_tokens = 500
        self.stream = True


state = AppState()

# GPRC_PORT = 5001


async def initialize_client():
    load_dotenv()

    model_path_str = os.getenv("MODEL_PATH")

    my_ip = os.getenv("IP")
    if not my_ip:
        my_ip = get_ip_address()
    host_maddrs = f"/ip4/{my_ip}/tcp/0"

    bootstrap_addr = os.getenv("BOOTSTRAP_NODE_ADDR")

    if not bootstrap_addr:
        bootstrap_addr = discover_bootstrap_node_address(is_client=True)

    if not bootstrap_addr:
        logger.error("Failed to find bootstrap node.")
        return

    bootstrap_peer_addr = get_bootstrap_peer_address(bootstrap_addr, attempts=5)
    initial_peers = [bootstrap_peer_addr] if bootstrap_peer_addr else None

    # Initialize the Client Node
    grpc_port = os.getenv("GRPC_PORT")
    grpc_port = grpc_port if grpc_port else get_free_port()
    grpc_addr = grpc_addr = f"{my_ip}:{grpc_port}"

    state.client = Client(
        model_path=Path(model_path_str),
        host_maddrs=[host_maddrs],
        initial_peers=initial_peers,
        grpc_addr=grpc_addr,
    )
    logger.info("Client initialized.")


# ----- UI Components -----
def render_server_card(node_id: str, role: str, info: Dict[str, Any]) -> Dict[str, ui.label]:
    """
    Renders a single server node card on the Left Sidebar.
    Roles: Head, Tail, Intermediate, Head_Tail
    """
    labels = {}

    color = (
        "green-100"
        if role == "Head" or role == "Head_Tail"
        else (
            "blue-100" if role == "Tail" else "orange-100" if role == "Intermediate" else "gray-100"
        )
    )

    with ui.card().classes(f"w-full p-0 bg-{color} gap-2"):
        with ui.expansion().classes("w-full font-mono text-sm") as expasion:
            with expasion.add_slot('header'):
                with ui.column().classes("gap-2 w-full"):
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label(role).classes("font-bold uppercase text-gray-600")
                    ui.label(f"Hostname: {info.get('hostname', 'N/A')}")
                    
                    # Layers 
                    if "layers" in info:
                        layers = info.get("layers")
                        if layers:
                            num_layers = layers[1] - layers[0] + 1
                            labels["layers"] = ui.label(
                                f"Layers: [{layers[0]} - {layers[1]}] | Count: {num_layers}"
                            ).classes("font-mono text-sm")
                    
                    # Output Layer Loaded
                    output_layer_loaded = info.get("output_layer_loaded", False)
                    labels["output_layer_loaded"] = ui.label(f"Output Layer Loaded: {output_layer_loaded}").classes(
                        "font-mono text-sm"
                    )

            # ID
            ui.label(f"ID: {node_id[:DIGITS_SHOW]}...").classes("font-mono text-sm")

            # Layers Loaded (bool)
            layers_loaded = info.get("layers_loaded", False)
            labels["layers_loaded"] = ui.label(f"Layers Loaded: {layers_loaded}").classes(
                "font-mono text-sm"
            )

            if "device" in info:
                ui.label(f'Device: {info["device"]}').classes("font-mono text-sm")

            if "processing_rate" in info:
                processing_rate = info.get("processing_rate", 0.0)
                labels["processing_rate"] = ui.label(
                    f"Processing Rate: {processing_rate / 1000000000:.2f}B params/sec"
                ).classes("font-mono text-sm")

            if "inference_delay" in info:
                inference_latency = info.get("inference_delay", 0.0)
                labels["inference_delay"] = ui.label(
                    f"Inference Delay: {inference_latency:.6f}s"
                ).classes("font-mono text-sm")

            if "grpc_overhead" in info:
                grpc_overhead = info.get("grpc_overhead", 0.0)
                labels["grpc_overhead"] = ui.label(f"GRPC Overhead: {grpc_overhead:.6f}s").classes(
                    "font-mono text-sm"
                )

            if "memory_usage" in info and "memory_limit" in info:
                mem_usage = info["memory_usage"]
                mem_limit = info["memory_limit"]
                labels["memory"] = ui.label(
                    f"Memory Usage: {int(mem_usage)}/{int(mem_limit)} MB"
                ).classes("font-mono text-sm")

            if "available_memory" in info:
                avail_mem = info["available_memory"]
                labels["available_memory"] = ui.label(f"Available Memory: {int(avail_mem)} MB").classes(
                    "font-mono text-sm"
                )

            if "vram_usage" in info and "vram_limit" in info:
                vram_usage = info["vram_usage"]
                vram_limit = info["vram_limit"]
                labels["vram"] = ui.label(
                    f"VRAM Usage: {int(vram_usage)}/{int(vram_limit)} MB"
                ).classes("font-mono text-sm")

            if "available_vram" in info:
                avail_vram = info["available_vram"]
                labels["available_vram"] = ui.label(f"Available VRAM: {int(avail_vram)} MB").classes(
                    "font-mono text-sm"
                )
            
            if "kv_cache_memories" in info:
                kv_cache_memories = info.get("kv_cache_memories")
                if kv_cache_memories:
                    for client_addr, kv_cache_size in kv_cache_memories.items():
                        labels[client_addr] = ui.label(f"{client_addr} KV Cache Size: {int(kv_cache_size)} MB").classes(
                            "font-mono tesxt-sm"
                        )

                # labels["kv_cache_size"] = ui.label(f"KV Cache Size: {int(info.get('kv_cache_size'))} MB").classes(
                #     "font-mono tesxt-sm"
                # )

            if "address" in info:
                ui.label(f'Address: {info["address"]}').classes("font-mono text-sm")

    return labels


async def refresh_chain_view(chain_container: ui.column, full_rebuild: bool = False):
    """
    Fetches DHT data and updates the Left Sidebar.
    """
    if not state.client:
        return

    # Fetch chain info
    # try:
    # chain_info = await asyncio.to_thread(state.client.chain.get_chain_info)
    chain_info = await run.io_bound(state.client.chain.get_chain_info)
    # except Exception as e:
    #     logger.error(f"Error fetching chain info: {e}")

    if not chain_info:
        chain_container.clear()
        with chain_container:
            ui.label("No chain found.").classes("text-2xl text-red-600")
            return

    # Extract topology signature
    current_topology = []

    # Process Active Chain
    server_list = chain_info.get("servers", [])
    for s in server_list:
        role = (
            "Head"
            if s["id"] == chain_info[HEAD_KEY]
            else "Tail" if s["id"] == chain_info[TAIL_KEY] else "Intemediate"
        )
        current_topology.append((s["id"], role, len(s)))

    # Process Backups
    backups_list = chain_info.get(BACKUPS_KEY, [])
    for b in backups_list:
        current_topology.append((b["id"], "Backup", len(b)))

    # Check if the topology matches the previous state
    topology_changed = (state.ui.last_topology != current_topology) or (full_rebuild)

    if not topology_changed:
        # --- Update in place ---

        # Global Keys
        if "total_params" in state.ui.global_labels:
            state.ui.global_labels["total_params"].text = (
                f"Total Params: {chain_info.get(TOTAL_PARAMS_KEY) / 1000000000:.2f}B"
            )

        if "total_layers" in state.ui.global_labels:
            state.ui.global_labels["total_layers"].text = (
                f"Total Transformer Layers: {chain_info.get(TOTAL_LAYERS_KEY)}"
            )
        
        if "num_clients" in state.ui.global_labels:
            state.ui.global_labels["num_clients"].text = (
                f"Num Clients: {chain_info.get(NUM_CLIENTS_KEY)}"
            )

        if "all_loaded" in state.ui.global_labels:
            all_loaded = chain_info.get(ALL_LAYERS_KEY, False)
            state.ui.global_labels["all_loaded"].text = f"All Layers Loaded: {all_loaded}"
            state.ui.global_labels["all_loaded"].classes(
                replace=f"text-lg {'text-green-600' if all_loaded else 'text-red-600'}"
            )

        if "chain_status" in state.ui.global_labels:
            current_status = chain_info.get(STATUS_KEY)
            state.ui.global_labels["chain_status"].text = f"Chain Status: {current_status.value}"
            status_color = (
                "text-green-600"
                if current_status == ChainStatus.READY
                else "text-red-600" if current_status == ChainStatus.UNREADY else "text-orange-600"
            )
            state.ui.global_labels["chain_status"].classes(replace=f"text-lg {status_color}")

        # Node Info
        def update_node_labels(nodes_list: List[Dict[str, Any]]):
            for node_info in nodes_list:
                node_id = node_info["id"]
                if node_id in state.ui.node_labels:
                    labels = state.ui.node_labels[node_id]

                    # Update Layers
                    if "layers" in labels and "layers" in node_info:
                        l = node_info["layers"]
                        labels["layers"].text = (
                            f"Layers: [{l[0]} - {l[1]}] | Count: {l[1] - l[0] + 1}"
                        )

                    # Update Loaded Status
                    if "layers_loaded" in labels:
                        labels["layers_loaded"].text = (
                            f"Layers Loaded: {node_info.get('layers_loaded', False)}"
                        )

                    if "output_layer_loaded" in labels:
                        labels["output_layer_loaded"].text = (
                            f"Output Layer Loaded: {node_info.get('output_layer_loaded', False)}"
                        )

                    if "processing_rate" in labels:
                        labels["processing_rate"].text = (
                            f"Processing Rate: {node_info.get('processing_rate', 0) / 1000000000:.2f}B params/sec"
                        )

                    if "inference_delay" in labels:
                        labels["inference_delay"].text = (
                            f"Inference Delay: {node_info.get('inference_delay', 0.0):.6f}s"
                        )

                    if "grpc_overhead" in labels:
                        labels["grpc_overhead"].text = (
                            f"GRPC Overhead: {node_info.get('grpc_overhead', 0.0):.6f}s"
                        )

                    # Update Memory
                    if (
                        "memory" in labels
                        and "memory_usage" in node_info
                        and "memory_limit" in node_info
                    ):
                        labels["memory"].text = (
                            f"Memory Usage: {int(node_info['memory_usage'])}/{int(node_info['memory_limit'])} MB"
                        )

                    if "available_memory" in labels and "available_memory" in node_info:
                        labels["available_memory"].text = (
                            f"Available Memory: {int(node_info['available_memory'])} MB"
                        )

                    # Update VRAM
                    if "vram" in labels and "vram_usage" in node_info and "vram_limit" in node_info:
                        labels["vram"].text = (
                            f"VRAM Usage: {int(node_info['vram_usage'])}/{int(node_info['vram_limit'])} MB"
                        )

                    if "available_vram" in labels and "available_vram" in node_info:
                        labels["available_vram"].text = (
                            f"Available VRAM: {int(node_info['available_vram'])} MB"
                        )

                    # if "kv_cache_memories" in node_info:
                    #     kv_cache_memories = node_info.get("kv_cache_memories")
                    #     if kv_cache_memories:
                    #         for client_addr, kv_cache_size in kv_cache_memories.items():
                    #             labels[client_addr].text = f"{client_addr} KV Cache Size: {int(kv_cache_size)} MB"
                        # labels["kv_cache_size"].text = f"KV Cache Size: {int(node_info.get('kv_cache_size'))} MB"

        update_node_labels(server_list)
        update_node_labels(backups_list)

    else:
        # --- Full Rebuild (Topology changed) ---

        chain_container.clear()
        state.ui.global_labels = {}
        state.ui.node_labels = {}

        spinner = ui.spinner().props("size=lg")
        with chain_container:
            spinner
        await asyncio.sleep(0.005)

        with chain_container:
            # Layer Status
            ui.label("Global Keys").classes("font-bold text-xl")
            total_layers = chain_info.get(TOTAL_LAYERS_KEY)
            total_params = chain_info.get(TOTAL_PARAMS_KEY)
            current_status = chain_info.get(STATUS_KEY)
            all_loaded = chain_info.get(ALL_LAYERS_KEY, False)
            num_clients = chain_info.get(NUM_CLIENTS_KEY)

            state.ui.global_labels["total_params"] = ui.label(
                f"Total Params: {total_params / 1000000000:.2f}B"
            ).classes("text-lg")

            state.ui.global_labels["total_layers"] = ui.label(
                f"Total Transformer Layers: {total_layers}"
            ).classes("text-lg")

            state.ui.global_labels["num_clients"] = ui.label(
                f"Num Clients: {num_clients}"
            ).classes("text-lg")

            state.ui.global_labels["all_loaded"] = ui.label(
                f"All Layers Loaded: {all_loaded}"
            ).classes(f"text-lg {'text-green-600' if all_loaded else 'text-red-600'}")

            status_color = (
                "text-green-600"
                if current_status == ChainStatus.READY
                else "text-red-600" if current_status == ChainStatus.UNREADY else "text-orange-600"
            )
            state.ui.global_labels["chain_status"] = ui.label(
                f"Chain Status: {current_status.value}"
            ).classes(f"text-lg {status_color}")

            # Active Chain Info
            ui.label("Active Chain").classes("font-bold text-xl")
            for server_info in server_list:
                node_id = server_info["id"]
                role = (
                    "Head_Tail"
                    if node_id == chain_info[HEAD_KEY] and node_id == chain_info[TAIL_KEY]
                    else (
                        "Head"
                        if node_id == chain_info[HEAD_KEY]
                        else "Tail" if node_id == chain_info[TAIL_KEY] else "Intermediate"
                    )
                )

                labels = render_server_card(node_id, role, server_info)
                state.ui.node_labels[node_id] = labels

            # Backup Nodes Info
            ui.label("Backup Nodes").classes("font-bold text-xl")
            for backup_info in backups_list:
                node_id = backup_info["id"]
                role = "Backup"

                labels = render_server_card(node_id, role, backup_info)
                state.ui.node_labels[node_id] = labels

            chain_container.remove(spinner)

    state.ui.last_topology = current_topology


def refresh_client_stats(client_stats_container: ui.row):
    client_stats_container.clear()
    with client_stats_container:
        ui.label("Client").classes("text-lg font-bold")
        ui.label(f"Address: {state.client.grpc_addr}").classes("")
        ui.separator()
        ui.label(f"Initial Inference Delay: {state.client.initial_inference_delay:.6f}s")
        ui.label(f"Serialization Delay: {state.client.serialization_delay:.6f}s")
        ui.label(f"Head Communication Latency: {state.client.head_communication_latency:.6f}s")
        ui.label(f"Deserialization Delay: {state.client.deserialization_delay:.6f}s")
        # ui.label(f"Final Inference Delay: {state.client.final_inference_delay:.6f}s")
        # ui.label(f"Logit Sampling Delay: {state.client.sample_delay:.6f}s")
        ui.label(f"Token Decoding Delay: {state.client.decode_delay:.6f}s")
        ui.label(f"Yield Delay: {state.client.yield_delay:.6f}s")


def get_next_token(generator):
    try:
        return next(generator)
    except StopIteration:
        return None


async def update_response_message(response_message: ui.chat_message, text: str):
    response_message.clear()
    with response_message:
        ui.markdown(text)
    await asyncio.sleep(0.001)

    # Scroll to bottom
    ui.run_javascript(
        'var el = document.getElementById("chat-container"); if (el) el.scrollTop = el.scrollHeight'
    )


async def generate(
    input_element: ui.input,
    chat_container: ui.column,
    stats_container: ui.column,
    chain_container: ui.column,
    send_btn: ui.button,
    stop_btn: ui.button,
):
    if state.is_generating:
        ui.notify("Please wait for the text generation to end.")
        return

    prompt = input_element.value
    if not prompt or not state.client:
        return

    input_element.value = ""
    state.is_generating = True

    send_btn.visible = False
    stop_btn.visible = True

    # Chat container
    with chat_container:
        ui.chat_message(text=prompt, name="You", sent=True)
        response_message = ui.chat_message(name="Distributed LLM", sent=False)
        spinner = ui.spinner()

    await asyncio.sleep(0.005)

    token_generator = await run.io_bound(
        state.client.generate, prompt, max_new_tokens=int(state.max_new_tokens), stream=state.stream
    )

    chat_container.remove(spinner)

    full_response = ""

    # If stream = False
    if isinstance(token_generator, str):
        full_response = token_generator
        await update_response_message(response_message, full_response)

    else:
        try:
            if token_generator:
                while True:
                    if not state.is_generating:
                        state.client.chain.update_chain_status(ChainStatus.READY)
                        break

                    token = await run.io_bound(get_next_token, token_generator)

                    if token is None:
                        break

                    full_response += token
                    await update_response_message(response_message, full_response)
        except (RuntimeError, ValueError, AttributeError, grpc.RpcError) as e:
            state.is_generating = False
            ui.notify(f"Generation Failed: {str(e)}", type="negative")
            send_btn.visible = True
            stop_btn.visible = False
            return

    state.is_generating = False
    send_btn.visible = True
    stop_btn.visible = False

    # Refresh chain info
    await refresh_chain_view(chain_container)

    # Update Stats
    stats = state.client.last_inference_stats
    num_tokens = stats.get("num_tokens_generated")
    latency = stats.get("latency")
    throughput = stats.get("throughput")

    stats_container.clear()
    with stats_container:
        ui.label("Generation").classes("text-lg font-bold")
        ui.separator()
        ui.label(f"Tokens Generated: {num_tokens} tokens")
        ui.label(f"Generation Time: {latency:.2f}s")
        ui.label(f"Throughput: {throughput:.2f} tokens/sec")
        ui.label(f"Total Rate: {state.client.total_rate / 1000000000:.2f}B params/sec")

        if state.client.total_rate > 0:
            ui.button(
                "Trigger Reallocation",
                on_click=lambda: trigger_reallocation(stats_container, chain_container),
            ).classes("w-full")


def stop_generation():
    """Signals the generation loop to stop."""
    if state.is_generating:
        state.is_generating = False
        ui.notify("Stopping generation...")


async def trigger_reallocation(stats_container: ui.column, chain_container: ui.column):

    spinner = ui.spinner().props("size=lg")
    with stats_container:
        spinner

    ui.notify("Triggering layer reallocation.")
    await asyncio.sleep(0.005)

    await asyncio.to_thread(state.client.trigger_reallocation)

    ui.notify("Reallocation complete.")
    stats_container.remove(spinner)

    await refresh_chain_view(chain_container)


def clear_chat(chat_container: ui.column):
    if state.client:
        state.client.chat_history = []
    chat_container.clear()
    ui.notify("Chat History Cleared...")


def run_background_init():
    asyncio.create_task(initialize_client())


app.on_startup(run_background_init)


# ----- Main Layout -----
@ui.page("/")
async def main_page():
    # Global style
    ui.query("#c3").classes("p-0")

    with ui.row().classes("w-full h-screen gap-0 flex-nowrap"):

        # Left Sidebar: Server Chain (fixed width)
        with ui.column().classes("w-1/4 h-full border-r border-gray-200 p-4 overflow-y-auto"):
            ui.label("Server Chain Info").classes("text-2xl font-bold")

            chain_container = ui.column().classes("w-full gap-2")

            await refresh_chain_view(chain_container, full_rebuild=True)

            ui.button(
                "Refresh",
                icon="refresh",
                on_click=lambda: refresh_chain_view(chain_container, full_rebuild=True),
            ).classes("w-full")

            # Backgroud auto-refresh
            ui.timer(1.0, lambda: refresh_chain_view(chain_container))

        # Center: Chat Area (Flexible Width)
        with ui.column().classes("flex-1 h-full relative p-4"):
            chat_container = (
                ui.column()
                .classes(
                    "w-full mx-auto flex-grow p-2 items-stretch overflow-y-auto overflow-x-hidden"
                )
                .props('id="chat-container"')
            )

            # Input Area
            with ui.row().classes("w-full bg-white p-4 items-center gap-2"):

                # Generation Settings Row
                with ui.column().classes("items-start"):
                    ui.number("Max Tokens", min=1, max=8192, step=1, format="%d").bind_value(
                        state, "max_new_tokens"
                    ).props("dense").classes("w-full")

                    ui.checkbox("Stream").bind_value(state, "stream").props(
                        "dense size=sm"
                    ).classes("text-xs")

                msg_input = (
                    ui.input(placeholder="Enter prompt...")
                    .classes("flex-1")
                    .props("outlined rounded")
                )

                send_btn = ui.button(
                    icon="send",
                    on_click=lambda: generate(
                        msg_input,
                        chat_container,
                        generation_stats_container,
                        chain_container,
                        send_btn,
                        stop_btn,
                    ),
                ).props("flat round color=primary")
                stop_btn = (
                    ui.button(icon="stop", on_click=stop_generation)
                    .props("flat round color=primary")
                    .classes("hidden")
                )
                stop_btn.visible = False

                # Clear Chat button
                ui.button(icon="delete", on_click=lambda: clear_chat(chat_container)).props(
                    "flat round color=negative"
                ).tooltip("Clear Chat History")

                # Bind Enter key
                msg_input.on(
                    "keydown.enter",
                    lambda: generate(
                        msg_input,
                        chat_container,
                        generation_stats_container,
                        chain_container,
                        send_btn,
                        stop_btn,
                    ),
                )

        # Right Sidebar: Stats (Fixed Width)
        with ui.column().classes("w-1/4 h-full border-l border-gray-200 p-4"):
            ui.label("Stats").classes("text-2xl font-bold")
            generation_stats_container = ui.column().classes("w-full gap-2")
            with generation_stats_container:
                ui.label("Waiting for inference...").classes("text-gray-400 italic")

            client_stats_container = ui.column().classes("w-full gap-2")

            ui.timer(1.0, lambda: refresh_client_stats(client_stats_container))

async def shutdown_client():
    """Handles graceful termination of background processes."""
    logger.info("Shutting down application and releasing ports...")
    if state.client:
        # Stop any active generation loops
        state.is_generating = False 

        state.client.shutdown()

app.on_shutdown(shutdown_client)


ui.run(title="Distributed LLM Client", port=8080, host="0.0.0.0")
