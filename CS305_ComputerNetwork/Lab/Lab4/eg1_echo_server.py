import socket

def echo():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 5555))
    sock.listen(10) # 监听10个客户
    sock.settimeout(0.5)
    while True:
        try:
            conn, address = sock.accept() # 阻塞式，等待连接
            while True:
                data = conn.recv(2048)
                if data and data != b'exit': # bytes类型
                    conn.send(data) # 将收到的信息回传
                    print(data)
                else:
                    conn.close()
                    break
        except socket.timeout:
            continue
if __name__ == "__main__":
    try:
        echo()
    except KeyboardInterrupt:
        pass