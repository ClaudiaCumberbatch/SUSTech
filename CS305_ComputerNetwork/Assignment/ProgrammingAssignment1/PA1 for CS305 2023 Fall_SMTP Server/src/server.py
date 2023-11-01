from __future__ import annotations

from argparse import ArgumentParser
from queue import Queue
import socket
from socketserver import ThreadingTCPServer, BaseRequestHandler
from threading import Thread
from email.mime.text import MIMEText
import re
import tomli
import hashlib

# 创建一个SHA-256哈希对象
hash_object = hashlib.sha256()


def student_id() -> int:
    return 12110644  # TODO: replace with your SID


parser = ArgumentParser()
parser.add_argument('--name', '-n', type=str, required=True)
parser.add_argument('--smtp', '-s', type=int)
parser.add_argument('--pop', '-p', type=int)

args = parser.parse_args()

with open('data/config.toml', 'rb') as f:
    _config = tomli.load(f)
    SMTP_PORT = args.smtp or int(_config['server'][args.name]['smtp'])
    POP_PORT = args.pop or int(_config['server'][args.name]['pop'])
    ACCOUNTS = _config['accounts'][args.name]
    MAILBOXES = {account: [] for account in ACCOUNTS.keys()}
    DELETE_INDEXES = {account: [] for account in ACCOUNTS.keys()}
    ID_LIST = []
    # print('ACCOUNTS: ', ACCOUNTS)
    # print('MAILBOX: ', MAILBOXES)

with open('data/fdns.toml', 'rb') as f:
    FDNS = tomli.load(f)

ThreadingTCPServer.allow_reuse_address = True


def fdns_query(domain: str, type_: str) -> str | None:
    domain = domain.rstrip('.') + '.'
    return FDNS[type_][domain]

class MAIL():
    def __init__(self, data):
        self.data = data
        hash_object.update(data)
        # 获取哈希值的十六进制表示
        self.id = hash_object.hexdigest()

class POP3Server(BaseRequestHandler):
    def handle(self):
        conn = self.request # 用于与客户端通信的新socket.socket对象
        conn.sendall(b'+OK POP3 Server Ready\r\n')

        while True:
            command = conn.recv(1024).strip().decode('utf-8')
            if not command:
                break
            print('Received:', command)
            # 用户名
            if command.lower().startswith('user'):
                username = command.split(' ')[1]
                if username in MAILBOXES: 
                    conn.sendall(b'+OK\r\n')
                    print('Response: +OK\n')
                else: 
                    conn.sendall(b'-ERR Account never registered!\r\n')
            # 密码
            elif command.lower().startswith('pass'):
                password = command.split(' ')[1]
                if ACCOUNTS[username] == password:
                    conn.sendall(b'+OK user successfully logged on\r\n')
                    print('Response: +OK\n')
                else:
                    conn.sendall(b'-ERR Wrong password!\r\n')
            # stat
            elif command.lower()=='stat':
                l = 0
                for mail in MAILBOXES[username]:
                    l += len(mail.data)
                conn.sendall(f'+OK {len(MAILBOXES[username])} {l}\r\n'.encode('utf-8'))
                print('Response: +OK\n')
            # list
            elif command.lower().startswith('list'):
                index = command.split(" ")[-1]
                if index.lower()=='list': # index not specified, returns all seperately
                    msgs = MAILBOXES[username] 
                    conn.sendall(f'+OK {len(msgs) - len(DELETE_INDEXES[username])} messages\r\n'.encode('utf-8'))
                    for i, msg in enumerate(msgs):
                        if i+1 in DELETE_INDEXES[username]: continue
                        conn.sendall(f'{i+1} {len(msg.data)}\r\n'.encode('utf-8'))
                    conn.sendall(b'.\r\n')
                    print('Response: +OK\n')
                else: 
                    index = int(index)
                    if index > len(MAILBOXES[username]) or index < 1 or index in DELETE_INDEXES[username]:
                        conn.sendall(b'-ERR Index out of range!\r\n')
                    else:
                        conn.sendall(f'+OK {index-1} {len(MAILBOXES[username][index-1].data)}\n\r'.encode('utf-8'))
                        print('Response: +OK\n')
            # retr
            elif command.lower().startswith('retr'):
                index = int(command.split(" ")[-1])
                if index > len(MAILBOXES[username]) or index < 1:
                    conn.sendall(b'-ERR Index out of range!\r\n')
                else:
                    conn.sendall(b'+OK %d octets\r\n' % len(MAILBOXES[username][index - 1].data))
                    conn.sendall(MAILBOXES[username][index-1].data)
                    # conn.sendall(b'\r\n.\r\n')
                    print('Response: +OK\n')
            # dele
            elif command.lower().startswith('dele'):
                index = int(command.split(" ")[-1])
                if index > len(MAILBOXES[username]):
                    conn.sendall(b'-ERR Index out of range!\r\n')
                else:
                    DELETE_INDEXES[username].append(index)
                    conn.sendall(f'+OK message {index} marked for deletion\r\n'.encode('utf-8'))
                    print('Response: +OK\n')
            # rset
            elif command.lower()=='rset':
                DELETE_INDEXES[username].clear()
                conn.sendall(b'+OK revoke all DELE commands\r\n')
                print('Response: +OK\n')
            # noop
            elif command.lower()=='noop':
                conn.sendall(b'+OK\r\n')
                print('Response: +OK\n')
            # quit
            elif command.lower()=='quit':
                DELETE_INDEXES[username].sort(reverse=True)
                for index in DELETE_INDEXES[username]:
                    MAILBOXES[username].pop(index)
                self.request.sendall(b"+OK\r\n")
                print('Response: +OK\n')
                break
            # uidl
            elif command.lower() == 'uidl':
                self.request.sendall(b"+OK Unique ID listing follows\r\n")
                for i, mail in enumerate(MAILBOXES[username], 1):
                    response = f"{i} {mail.id}\r\n"
                    conn.sendall(response.encode('utf-8'))
                conn.sendall(b".\r\n")
                print('Response: +OK\n')
            elif command.lower().startswith("uidl"):
                index = int(command.split(" ")[-1])
                if index > len(MAILBOXES[username]) or index < 1 or index in DELETE_INDEXES[username]:
                    conn.sendall(b'-ERR Index out of range!\r\n')
                else:
                    conn.sendall(f'+OK {index} {MAILBOXES[username][index-1].id}\r\n'.encode('utf-8'))
                    print('Response: +OK\n')
            # top
            elif command.lower().startswith("top"):
                msg = int(command.split(" ")[1])
                lines = int(command.split(" ")[2])
                if msg > len(MAILBOXES[username]) or msg < 1 or msg in DELETE_INDEXES[username] or lines < 0:
                    conn.sendall(b'-ERR Index out of range!\r\n')
                else:
                    conn.sendall(b'+OK\r\n')
                    # send it all
                    if lines > len(MAILBOXES[username][msg-1].data.split(b"\n")):
                        conn.sendall(MAILBOXES[username][msg-1].data)
                    # send top lines of the mail
                    else:
                        for line in MAILBOXES[username][msg-1].data.split(b"\n")[0:lines]:
                            conn.sendall(line)
                        conn.sendall(b"\r\n.\r\n")
                    print('Response: +OK\n')
            else:
                conn.sendall(b'-ERR unknown error\r\n')


class SMTPServer(BaseRequestHandler):
    def sendmail(self, sender, receiver, data):
        mx = fdns_query(receiver.split("@")[1], 'MX')
        port = int(fdns_query(mx, 'P'))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', port))
        # sock.connect(('10.32.8.214', port))
        # sock.connect(('10.32.40.94', port))
        d = sock.recv(1024)
        print("after connect,", d)
        sock.sendall(b'HELO smtp\n')
        d = sock.recv(1024)
        print("after helo,", d)
        sock.sendall(b'MAIL FROM: <' + sender.encode('utf-8') + b'>\n')
        d =sock.recv(1024)
        print("after sender", d)
        sock.sendall(b'RCPT TO: <' + receiver.encode('utf-8') + b'>\n')
        d =sock.recv(1024)
        print("after receiver", d)
        sock.sendall(b'DATA\n')
        d =sock.recv(1024)
        print("after data", d)
        sock.sendall(data)
        d =sock.recv(1024)
        print("after real data", d)
        sock.sendall(b'QUIT\n')
        d =sock.recv(1024)
        print("after quit", d)
        sock.close()
        return

    def handle(self):
        conn = self.request
        conn.sendall(b'220 SMTP Server Ready\r\n')
        from_agent = False
        wrong_name = False

        while True:
            command = conn.recv(1024).strip().decode('utf-8')
            if not command:
                break

            print('Received:', command)
            if command.lower().startswith('ehlo') or command.lower().startswith('helo'):
                msg = '250 HELLO'
                conn.sendall(f'{msg}\r\n'.encode('utf-8'))
                print('Response:', msg, '\n')
            elif command.lower().startswith('mail'):
                sender = re.search(r'<(.*?)>', command).group(1)
                # same port number, connection from agent, check sender
                print(sender)
                mx = fdns_query(sender.split("@")[1], 'MX')
                if SMTP_PORT==int(fdns_query(mx, 'P')):
                    from_agent = True
                    if sender not in MAILBOXES: 
                        conn.sendall(b'553 Command stopped because the mailbox name doesn\'t exist\r\n')
                        print('Response: 553')
                        break
                else: from_agent = False
                conn.sendall(f'250 Sender <{sender}> OK\r\n'.encode('utf-8'))
                print('Response: 250 OK\n')
            elif command.lower().startswith('rcpt'):
                recipient = re.search(r'<(.*?)>', command).group(1)
                # connection from SMTP server, check reveiver
                if not from_agent:
                    if recipient not in MAILBOXES:
                        print("wrong name!")
                        wrong_name = True
                        conn.sendall(b'553 Command stopped because the mailbox name doesn\'t exist\r\n')
                        print('Response: 553')
                        continue
                    else: wrong_name = False
                conn.sendall(f'250 Recipient <{recipient}> OK\r\n'.encode('utf-8'))
                print('Response: 250 OK\n')
            elif command.lower().startswith('data'):
                msg = '354 Start mail input; end with <CRLF>.<CRLF>\r\n'
                conn.sendall(msg.encode('utf-8'))
                print('Response:', msg)
                data = b''
                while True:
                    line = conn.recv(1024)
                    data += line
                    if b'\r\n.\r\n' in data:
                        break
                print('data:', data)
                if wrong_name:
                    # 用socket来发送原始邮件
                    self.sendmail(recipient, sender, data)
                    conn.sendall(b'250 Message accepted for delivery\r\n')
                    print('Response: 250 OK\n')
                    continue
                        
                # 判断是否同一个domain
                mx = fdns_query(recipient.split("@")[1], 'MX')
                if SMTP_PORT==int(fdns_query(mx, 'P')):
                    MAILBOXES[recipient].append(MAIL(data))
                else:
                    # 用socket来实际发送邮件
                    self.sendmail(sender, recipient, data)
                
                conn.sendall(b'250 Message accepted for delivery\r\n')
                print('Response: 250 OK\n')
            elif command.lower() == 'quit':
                conn.sendall(b'221 Bye\r\n')
                print('Response: 221 Bye\n')
                break
            elif command.lower() == 'help':
                commands = [
                    "HELO",
                    "EHLO",
                    "MAIL FROM",
                    "RCPT TO",
                    "DATA",
                    "QUIT",
                    "VRFY"
                ]
                response = "Supported commands: " + ", ".join(commands)
                conn.sendall (214, response.encode('utf-8'))
                print('Response: 214\n')
            else:
                conn.sendall(b'544 Transaction failed without additional details\r\n')
                print('unknown err\n')


if __name__ == '__main__':
    if student_id() % 10000 == 0:
        raise ValueError('Invalid student ID')

    smtp_server = ThreadingTCPServer(('', SMTP_PORT), SMTPServer)
    pop_server = ThreadingTCPServer(('', POP_PORT), POP3Server)
    Thread(target=smtp_server.serve_forever).start()
    Thread(target=pop_server.serve_forever).start()
