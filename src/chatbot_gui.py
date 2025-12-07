import os
import asyncio
from pathlib import Path

from nicegui import ui
import html

from client.client import Client
from core.remote.utils import get_bootstrap_peer_address


def root():
    model_path_str = os.getenv("MODEL_PATH")
    model_path = Path(model_path_str)

    host_maddrs = os.getenv("HOST_MADDRS")
    bootstrap_node_addr_str = os.getenv("BOOTSTRAP_NODE_ADDR")

    # Connect to the bootstrap node to get its p2p Multiaddress
    bootstrap_peer_addr = get_bootstrap_peer_address(bootstrap_node_addr_str, attempts=10)

    if not bootstrap_peer_addr:
        return

    initial_peers = [bootstrap_peer_addr]
    print(f"Successfully discovered bootstrap peer: {initial_peers[0]}")

    client = Client(
        model_path=model_path,
        host_maddrs=[host_maddrs],
        initial_peers=initial_peers,
    )

    async def generate():
        prompt = text.value
        text.value = ""

        with message_container:
            ui.chat_message(text=prompt, name="You", sent=True)
            response_message = ui.chat_message(name="Distributed LLM", sent=False)
            spinner = ui.spinner(type="dots")
        await asyncio.sleep(0.005)

        token_generator = client.generate(prompt, max_new_tokens=250, stream=True)

        message_container.remove(spinner)
        response_text = ""
        for token in token_generator:
            response_text += token
            response_message.clear()
            with response_message:
                # ui.html(response_text.replace("\n", "<br>"), sanitize=False)
                ui.markdown(response_text)
            await asyncio.sleep(0.005)
            # await asyncio.sleep(0)

            # Scroll to bottom
            ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")

    message_container = ui.column().classes("w-full max-w-2xl mx-auto flex-grow items-stretch")

    with ui.footer().classes("bg-white"):
        with ui.column().classes("w-full max-w-3xl mx-auto my-6"):
            with ui.row().classes("w-full no-wrap items-center"):
                text = (
                    ui.input(placeholder="Enter prompt")
                    .props("rounded outlined input-class=mx-3")
                    .classes("w-full self-center")
                    .style("border-width: 3px;")
                    .on("keydown.enter", generate)
                )

                with text.add_slot("append"):
                    ui.icon("send").classes("cursor-pointer").on("click", generate)


ui.run(root, title="Distributed LLM Chat")
