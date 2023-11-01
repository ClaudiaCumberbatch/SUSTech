import asyncio
import websockets

async def echo(uri):
    async with websockets.connect(uri) as websocket:
        while True:
            message = input("Write down your message:")
            await websocket.send(message)
            print("<", message)
            recv_text = await websocket.recv()
            print("> {}".format(recv_text))

asyncio.get_event_loop().run_until_complete(
    echo('ws://127.0.0.1:8766'))