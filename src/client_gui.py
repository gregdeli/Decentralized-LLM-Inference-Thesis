import asyncio
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from nicegui import ui, app
from client.client import Client
from core.remote.utils import get_bootstrap_peer_address
from core.p2p.chain_manager import HEAD_KEY, TAIL_KEY, TOTAL_LAYERS_KEY, ALL_LAYERS_KEY, DIGITS_SHOW


class AppState:
    def __init__(self):
        self.client: Client = None
        self.is_generating = False


state = AppState()


async def initialize_client():
    model_path_str = os.getenv("MODEL_PATH", "/models/Llama-3.2-1B-Instruct")
    host_maddrs = os.getenv("HOST_MADDRS", "/ip4/0.0.0.0/tcp/0")
    bootstrap_addr = os.getenv("BOOTSTRAP_NODE_ADDR", "tail-server:5001")

    # Discovery logic from distributed_chatbot.py
    bootstrap_peer_addr = get_bootstrap_peer_address(bootstrap_addr, attempts=5)
    initial_peers = [bootstrap_peer_addr] if bootstrap_peer_addr else None

    # Initialize your actual Client
    state.client = Client(
        model_path=Path(model_path_str),
        host_maddrs=[host_maddrs],
        initial_peers=initial_peers,
    )
    # ui.notify(f"Client initialized. Peer ID: {state.client.chain.node_id}")


# ----- UI Components -----
def render_server_card(node_id: str, role: str, info: Dict[str, Any]):
    """
    Renders a single server node card on the Left Sidebar.
    Roles: Head, Tail, Intermidiate,
    """
    color = "green-100" if role == "Head" else "blue-100" if role == "Tail" else "gray-100"

    with ui.card().classes(f"w-full p-2 bg-{color}"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(role).classes("font-bold text-xs uppercase text-gray-600")
            ui.icon("dns", color="gray").classes("text-sm")

        ui.label(f"ID: {node_id[:DIGITS_SHOW]}...").classes("font-mono text-xs")

        if "layers_loaded" in info:
            layers = info["layers_loaded"]
            ui.label(f"Layers: {layers[0]} - {layers[1]}").classes("text-xs text-blue-800")

        # Display Address
        if "address" in info:
            ui.label(info["address"]).classes("text-[10px] text-gray-500 break-all")


def refresh_chain_view(container: ui.column):
    """
    Fetches DHT data and updates the Left Sidebar.
    """
    if not state.client:
        return

    container.clear()
    chain_info = state.client.chain.get_chain_info()

    with container:
        ui.label("Active Chain").classes("text-sm font-bold mt-2 text-gray-500")

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

        # Backup Nodes and Layer Status
        ui.label("Network Status").classes("text-sm font-bold mt-4 text-gray-500")
        total_layers = chain_info.get(TOTAL_LAYERS_KEY)
        all_loaded = chain_info.get(ALL_LAYERS_KEY)
        ui.label(f"Total Layers: {total_layers}")
        ui.label(f"All Loaded: {all_loaded}").classes("text-green-600" if all_loaded else "text-red-600")


async def generate(input_element: ui.input, chat_container: ui.column, stats_container: ui.column):
    prompt = input_element.value
    if not prompt or not state.client:
        return

    input_element.value = ""
    state.is_generating = True

    # Chat container
    with chat_container:
        ui.chat_message(text=prompt, name="You", sent=True)
        response_message = ui.chat_message(name="Distributed LLM", sent=False)
        spinner = ui.spinner()

    await asyncio.sleep(0.005)

    full_response = ""
    # start_time = time.perf_counter()

    token_generator = state.client.generate(prompt, max_new_tokens=250, stream=True)

    chat_container.remove(spinner)

    token_count = 0
    if token_generator:
        for token in token_generator:
            full_response += token
            response_message.clear()
            # response_markdown.set_content(full_response)
            with response_message:
                ui.markdown(full_response)
            token_count += 1
            await asyncio.sleep(0.005)

    state.is_generating = False

    # Update Stats
    # elapsed = time.perf_counter() - start_time
    # tps = token_count / elapsed if elapsed > 0 else 0 # Thoughput (tokens/sec)

    try:
        stats_container.clear()
        with stats_container:
            ui.label("Performance").classes("text-lg font-bold")
            ui.separator()
            # ui.label(f"Throughput: {tps:.2f} tok/s").classes('text-xl text-blue-600 font-mono')
            # ui.label(f"Last Latency: {elapsed:.2f}s")
            ui.label(f"Total Rate: {state.client.total_rate:.2f} layers/sec").classes("text-gray-500")

            if state.client.total_rate > 0:
                ui.button("Trigger Reallocation", on_click=lambda: state.client.trigger_reallocation()).classes("mt-4 bg-orange-500 text-white")
    except RuntimeError:
        return


# ----- Main Layout -----
@ui.page("/")
async def main_page():
    if not state.client:
        await initialize_client()

    # Apply global styles
    ui.query("body").classes("bg-slate-50 p-0 m-0 overflow-hidden")

    with ui.row().classes("w-full h-screen gap-0"):

        # Left Sidebar: Server Chain (fixed width)
        with ui.column().classes("w-1/4 h-full bg-white border-r border-gray-200 p-4 overflow-y-auto"):
            ui.label("Server Chain").classes("text-xl font-bold mb-4 text-slate-800")

            chain_container = ui.column().classes("w-full gap-2")

            refresh_chain_view(chain_container)

            ui.button("Refresh", icon="refresh", on_click=lambda: refresh_chain_view(chain_container)).classes("w-full mt-4 bg-slate-700 text-white")

        # Center: Chat Area (Flexible Width)
        with ui.column().classes("flex-1 h-full relative p-6"):
            chat_container = ui.column().classes("w-full mx-auto flex-grow items-stretch overflow-y-auto overflow-x-hidden")

            # Input Area
            with ui.row().classes("w-full bg-white p-4 border-t border-gray-200 items-center gap-2"):
                msg_input = ui.input(placeholder="Enter prompt...").classes("flex-1").props("outlined rounded")
                send_btn = ui.button(icon="send", on_click=lambda: generate(msg_input, chat_container, stats_container)).props(
                    "flat round color=primary"
                )

                # Bind Enter key
                msg_input.on("keydown.enter", lambda: generate(msg_input, chat_container, stats_container))

        # Right Sidebar: Stats (Fixed Width)
        with ui.column().classes("w-1/5 h-full bg-white border-l border-gray-200 p-4"):
            ui.label("Stats").classes("text-xl font-bold mb-4 text-slate-800")
            stats_container = ui.column().classes("w-full gap-2")
            with stats_container:
                ui.label("Waiting for inference...").classes("text-gray-400 italic")


ui.run(title="Distributed LLM Client", port=8080)
