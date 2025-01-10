# Work Log

## Environment

403forbidden, 手动从网站上下载镜像https://hub.docker.com/r/handsonsecurity/seed-ubuntu/tags
docker pull handsonsecurity/seed-ubuntu:small-arm
再sudo docker compose build

## Useful Commands

左：
sudo docker compose build

sudo docker compose down

sudo docker compose up


右：
apt-get update && apt-get install -y netcat

sudo docker ps --format "{{.ID}} {{.Names}}"

sudo docker exec -it 038e74913a44 /bin/bash

echo hello | nc 10.9.0.5 9090

echo $(cat badfile) | nc 10.9.0.5 9090

nc 10.9.0.5 9090 < ./badfile 


外：
docker cp badfile 0a0de3ce2a2f:/fmt



## Scratch

(gdb) x/40x 0x0000ffffffffef68
0xffffffffef68: 0x0001f0cc      0x00000004      0xffff0000      0xffffff80
0xffffffffef78: 0x00435830      0x00000000      0x00492240      0x00000000
0xffffffffef88: 0x00000000      0x00000000      0x00000000      0x00000000




这是一个简单hello
server-10.9.0.5  | Got a connection from 10.9.0.6
server-10.9.0.5  | Starting format
server-10.9.0.5  | The input buffer's address:    0x0000fffffffff2d8
server-10.9.0.5  | The secret message's address:  0x0000000000458408
server-10.9.0.5  | The target variable's address: 0x0000000000492048
server-10.9.0.5  | Waiting for user input ......
server-10.9.0.5  | Received 6 bytes.
server-10.9.0.5  | Frame Pointer (inside dummy):      0x0000fffffffff200
server-10.9.0.5  | Content of fp+8 (inside dummy):      0x0000000000000000
server-10.9.0.5  | Content of fp+16 (inside dummy):      0x000000000049b926
server-10.9.0.5  | Content of fp-8 (inside dummy):      0x0000fffffffff2a0
server-10.9.0.5  | Content of fp-16 (inside dummy):      0xf00ff00ff00ff00f
server-10.9.0.5  | Frame Pointer (inside myprintf):      0x0000fffffffff1d0
server-10.9.0.5  | Content of fp+8 (inside myprintf):      0xf00ff00ff00ff00f
server-10.9.0.5  | Content of fp+16 (inside myprintf):      0x0000000000000000
server-10.9.0.5  | Content of fp-8 (inside myprintf):      0x00000000ff000000
server-10.9.0.5  | Content of fp-16 (inside myprintf):      0xf00ff00ff00ff00f
server-10.9.0.5  | The target variable's value (before): 0x1122334455667788
server-10.9.0.5  | hello
server-10.9.0.5  | x30 (inside myprintf):      0x00000000004007b0
server-10.9.0.5  | The target variable's value (after):  0x1122334455667788
server-10.9.0.5  | x30 (inside dummy):      0x00000000004009f0
server-10.9.0.5  | x30 (inside main):      0x00000000004008ac
server-10.9.0.5  | (^_^)(^_^)  Returned properly (^_^)(^_^)