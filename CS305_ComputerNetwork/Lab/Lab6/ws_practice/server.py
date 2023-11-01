import asyncio
import websockets

connected_clients = set()

class DanmakuServer:
    def __init__(self):
        # TODO: define your variables needed in this class
        # 创建用户列表
        connected_clients = set()
        # raise NotImplementedError

    async def reply(self, websocket):
        # TODO: design your reply method
        # 识别用户，向所有用户发送
        connected_clients.add(websocket) 
        print("add websocket", websocket) 
        async for message in websocket:
            for client in connected_clients:
                await client.send(message)  
        # raise NotImplementedError


if __name__ == "__main__":
    server = DanmakuServer()
    asyncio.get_event_loop().run_until_complete(
        websockets.serve(server.reply, 'localhost', 8765))
    asyncio.get_event_loop().run_forever()
