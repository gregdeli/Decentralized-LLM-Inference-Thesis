import asyncio
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import List, Dict, Any

from nicegui import ui, app
from client.client import Client
from core.remote.utils import get_bootstrap_peer_address, discover_bootstrap_node_address, get_ip_address
from core.p2p.chain_manager import HEAD_KEY, TAIL_KEY, TOTAL_LAYERS_KEY, ALL_LAYERS_KEY, BACKUPS_KEY, DIGITS_SHOW

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self):
        self.client: Client = None
        self.is_generating = False


state = AppState()

GPRC_PORT = 5001

async def initialize_client():
    load_dotenv()

    model_path_str = os.getenv("MODEL_PATH")

    my_ip = os.getenv("IP")
    if not my_ip:
        my_ip = get_ip_address()
    host_maddrs = f"/ip4/{my_ip}/tcp/0"

    bootstrap_addr = os.getenv("BOOTSTRAP_NODE_ADDR")

    if not bootstrap_addr:
        bootstrap_addr = discover_bootstrap_node_address()

    if not bootstrap_addr:
        logger.error("Failed to find bootstrap node.")
        return

    bootstrap_peer_addr = get_bootstrap_peer_address(bootstrap_addr, attempts=5)
    initial_peers = [bootstrap_peer_addr] if bootstrap_peer_addr else None

    # Initialize the Client Node
    grpc_addr = grpc_addr = f"{my_ip}:{GPRC_PORT}"
    state.client = Client(
        model_path=Path(model_path_str),
        host_maddrs=[host_maddrs],
        initial_peers=initial_peers,
        grpc_addr = grpc_addr
    )
    logger.info("Client initialized.")

# ----- UI Components -----
def render_server_card(node_id: str, role: str, info: Dict[str, Any]):
    """
    Renders a single server node card on the Left Sidebar.
    Roles: Head, Tail, Intermidiate,
    """
    color = "green-100" if role == "Head" else "blue-100" if role == "Tail" else "orange-100" if role == "Intermidiate" else "gray-100"

    with ui.card().classes(f"w-full p-2 bg-{color} gap-2"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(role).classes("font-bold text-xs uppercase text-gray-600")
            ui.icon("dns", color="gray").classes("text-sm")

        ui.label(f"ID: {node_id[:DIGITS_SHOW]}...").classes("font-mono text-sm")

        if "layers" in info:
            layers = info["layers"]
            num_layers = layers[1] - layers[0] + 1
            ui.label(f"Layers: [{layers[0]} - {layers[1]}] | Count: {num_layers}").classes("font-mono text-sm")

        layers_loaded = info.get("layers_loaded", False)
        ui.label(f"Layers Loaded: {layers_loaded}").classes("font-mono text-sm")

        if "device" in info:
            ui.label(f'Device: {info["device"]}').classes("font-mono text-sm")

        if "memory_usage" in info and "memory_limit" in info:
            mem_usage = info["memory_usage"]
            mem_limit = info["memory_limit"]
            ui.label(f"Memory Usage: {int(mem_usage)}/{int(mem_limit)} MB").classes("font-mono text-sm")

        if "available_memory" in info:
            avail_mem = info["available_memory"]
            ui.label(f"Available Memory: {int(avail_mem)} MB").classes("font-mono text-sm")

        if "vram_usage" in info and "vram_limit" in info:
            vram_usage = info["vram_usage"]
            vram_limit = info["vram_limit"]
            ui.label(f"VRAM Usage: {int(vram_usage)}/{int(vram_limit)} MB").classes("font-mono text-sm")

        if "available_vram" in info:
            avail_vram = info["available_vram"]
            ui.label(f"Available VRAM: {int(avail_vram)} MB").classes("font-mono text-sm")

        if "address" in info:
            ui.label(f'Address: {info["address"]}').classes("font-mono text-sm")


async def refresh_chain_view(chain_container: ui.column):
    """
    Fetches DHT data and updates the Left Sidebar.
    """
    if not state.client:
        return

    chain_container.clear()
    with chain_container:
        spinner = ui.spinner().props("size=lg")

    await asyncio.sleep(0.005)

    chain_info = state.client.chain.get_chain_info()

    with chain_container:
        if not chain_info:
            ui.label("No chain info found.")
            chain_container.remove(spinner)
            return

        # Layer Status
        ui.label("Global Keys").classes("font-bold text-lg")
        total_layers = chain_info.get(TOTAL_LAYERS_KEY)
        all_loaded = chain_info.get(ALL_LAYERS_KEY, False)
        ui.label(f"Total Layers: {total_layers}")
        ui.label(f"All Layers Loaded: {all_loaded}").classes("text-green-600" if all_loaded else "text-red-600")

        # Chain Info
        ui.label("Active Chain").classes("font-bold text-lg")
        if not chain_info:
            ui.label("No Chain Found").classes("text-red-500 italic")
            return

        for server_info in chain_info["servers"]:
            node_id = server_info["id"]
            role = "Intermidiate"
            if node_id == chain_info[HEAD_KEY]:
                role = "Head"
            elif node_id == chain_info[TAIL_KEY]:
                role = "Tail"

            render_server_card(node_id, role, server_info)

        # Backup Nodes
        ui.label("Backup Nodes").classes("font-bold text-lg")

        for backup_info in chain_info[BACKUPS_KEY]:
            node_id = backup_info["id"]
            role = "Backup"

            render_server_card(node_id, role, backup_info)
    
    chain_container.remove(spinner)


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

    token_generator = state.client.generate(prompt, max_new_tokens=500, stream=True)

    chat_container.remove(spinner)

    full_response = ""
    token_count = 0
    try:
        if token_generator:
            for token in token_generator:
                if not state.is_generating:
                    break
                full_response += token
                response_message.clear()
                with response_message:
                    ui.markdown(full_response)
                token_count += 1
                await asyncio.sleep(0.01)

                # Scroll to bottom
                ui.run_javascript('var el = document.getElementById("chat-container"); if (el) el.scrollTop = el.scrollHeight')
    except (RuntimeError, AttributeError) as e:
        state.is_generating = False
        ui.notify(f"Generation Failed: {str(e)}", type="negative")
        send_btn.visible = True
        stop_btn.visible = False
        return

    state.is_generating = False
    send_btn.visible = True
    stop_btn.visible = False

    # Update Stats
    stats = state.client.last_inference_stats
    latency = stats["latency"]
    throughput = stats["throughput"]

    stats_container.clear()
    with stats_container:
        ui.label("Performance").classes("text-lg font-bold")
        ui.separator()
        ui.label(f"Generation Time: {latency:.2f}s")
        ui.label(f"Throughput: {throughput:.2f} tokens/sec")
        ui.label(f"Total Rate: {state.client.total_rate:.2f} layers/sec")

        if state.client.total_rate > 0:
            ui.button("Trigger Reallocation", on_click=lambda: trigger_reallocation(stats_container, chain_container)).classes("w-full")


def stop_generation():
    """Signals the generation loop to stop."""
    if state.is_generating:
        state.is_generating = False
        ui.notify("Stopping generation...")


async def trigger_reallocation(stats_container: ui.column, chain_container: ui.column):

    with stats_container:
        spinner = ui.spinner().props("size=lg")

    ui.notify("Triggering layer reallocation.")
    await asyncio.sleep(0.005)

    state.client.trigger_reallocation()

    ui.notify("Reallocation complete.")
    stats_container.remove(spinner)

    await refresh_chain_view(chain_container)


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

            await refresh_chain_view(chain_container)

            ui.button("Refresh", icon="refresh", on_click=lambda: refresh_chain_view(chain_container)).classes("w-full")

        # Center: Chat Area (Flexible Width)
        with ui.column().classes("flex-1 h-full relative p-4"):
            chat_container = (
                ui.column().classes("w-full mx-auto flex-grow p-2 items-stretch overflow-y-auto overflow-x-hidden").props('id="chat-container"')
            )

            # Input Area
            with ui.row().classes("w-full bg-white p-4 items-center gap-2"):
                msg_input = ui.input(placeholder="Enter prompt...").classes("flex-1").props("outlined rounded")

                send_btn = ui.button(
                    icon="send", on_click=lambda: generate(msg_input, chat_container, stats_container, chain_container, send_btn, stop_btn)
                ).props("flat round color=primary")
                stop_btn = ui.button(icon="stop", on_click=stop_generation).props("flat round color=primary").classes("hidden")
                stop_btn.visible = False

                # Bind Enter key
                msg_input.on("keydown.enter", lambda: generate(msg_input, chat_container, stats_container, chain_container, send_btn, stop_btn))

        # Right Sidebar: Stats (Fixed Width)
        with ui.column().classes("w-1/4 h-full border-l border-gray-200 p-4"):
            ui.label("Stats").classes("text-2xl font-bold")
            stats_container = ui.column().classes("w-full gap-2")
            with stats_container:
                ui.label("Waiting for inference...").classes("text-gray-400 italic")


ui.run(title="Distributed LLM Client", port=8080, host="0.0.0.0")
