import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        message = "I got your message: {}".format(message)
        await websocket.send(message) # 一对一通信

asyncio.get_event_loop().run_until_complete(websockets.serve(echo, '127.0.0.1', 8766))
asyncio.get_event_loop().run_forever()