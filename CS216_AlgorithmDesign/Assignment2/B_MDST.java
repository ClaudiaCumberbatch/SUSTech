package lab5;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class B_MDST {
    public static ArrayList<b_node> R = new ArrayList<>();
    public static void main(String[] args) throws IOException{
//        Reader input = new Reader();
        Scanner input = new Scanner(System.in);
        int n = input.nextInt();
        int m = input.nextInt();
        b_node[] nodes = new b_node[n+1];
        for (int i = 0; i < n; i++){
            b_node nd = new b_node(i);
            nodes[i] = nd;
        }
        int indexCounter = n;
        for (int i = 0; i < m; i++){
            int s = input.nextInt();
            int t = input.nextInt();
            int weight = input.nextInt();
            b_edge eg = new b_edge(nodes[s], nodes[t], weight);
            nodes[t].Proot = merge(nodes[t].Proot, eg);
        }
        b_node root = new b_node(indexCounter++);
        nodes[n] = root;
        for (int i = 0; i < n; i++){
//            b_edge eg = new b_edge(root, nodes[i], Integer.MAX_VALUE);
            b_edge eg = new b_edge(root, nodes[i], 1000000);
            nodes[i].Proot = merge(nodes[i].Proot, eg);
        }
        // 无差别加边
        for (int i = 0; i < n; i++){
//            b_edge eg = new b_edge(nodes[i], nodes[i+1], Integer.MAX_VALUE);
            b_edge eg = new b_edge(nodes[i], nodes[i+1], 1000000);
            nodes[i+1].Proot = merge(nodes[i+1].Proot, eg);
        }
        // contract
        b_node a = nodes[n];
        int trueRootIndex = -1;
        while (a.Proot != null){
            b_edge uv = a.Proot;
            pushDown(uv);
            a.Proot = merge(a.Proot.lson, a.Proot.rson);
            while(find(uv.inNode) == find(uv.outNode)){ // 去自环
                uv = a.Proot;
                if (uv == null) break;
                pushDown(uv);
                a.Proot = merge(a.Proot.lson, a.Proot.rson);
            }
            if (uv == null) break;
            b_node u = uv.outNode;
            b_node b = find(u);
            if (a != b){
                a.in = uv;
                a.prev = b;
                if (b == root) {
                    trueRootIndex = a.in.inNode.index;
                }
                if (b.in == null){ // path extended
                    a = b;
                }else{ // new cycle formed
                    b_node c = new b_node(indexCounter++);
                    while(a.parent == null){
                        a.parent = c; // super-vertex
                        a.ans = c;
//                        System.out.printf("node %d in weight %d\n", a.index, a.in.weight);
                        if (a.Proot != null){
                            a.Proot.lazy += a.in.weight;
                            a.Proot.weight -= a.in.weight;
                        }
                        c.children.add(a);
                        // meld
                        c.Proot = merge(c.Proot, a.Proot);
                        a = a.prev;
                        if (a.parent != null) a = find(a);
                        if (a == c) break;
                    }
//                    a = a.parent;
                }
            }
        }
        // expand
        dismantle(root);
//        System.out.println("------------expand period-------------");
        while(R.size() > 0){
            b_node c = R.get(R.size()-1);
            R.remove(R.size()-1);
            b_edge uv = c.in;
            b_node v = uv.inNode;
            v.in = uv;
//            System.out.printf("node %d in weight %d\n", v.index, uv.weight);
            dismantle(v);
        }
        // calculate result
        int result = 0;
        boolean flag = false;
        for (int i = 0; i < n; i++){
//            System.out.printf("node %d weight %d\n", i, nodes[i].in.originalWeight);
            if (i == trueRootIndex) continue;
            if (nodes[i].in.originalWeight > 10000){
                System.out.println("impossible");
                flag = true;
                break;
            }
            if (nodes[i].prev != root) result += nodes[i].in.originalWeight;
        }
        if (!flag) System.out.printf("%d %d\n", result, trueRootIndex);
    }
    static b_node find(b_node u){
        b_node v = u;
        while (u.ans != null){
            u = u.ans;
        }
        while (v.ans != null){
            b_node t = v.ans;
            v.ans = u;
            v = t;
        }
        return u;
    }
    static void pushDown(b_edge root){
        b_edge left = root.lson;
        if (left != null){
            left.weight -= root.lazy; //lazy是个正数
            left.lazy += root.lazy;
        }
        b_edge right = root.rson;
        if (right != null){
            right.weight -= root.lazy;
            right.lazy += root.lazy;
        }
        root.lazy = 0;
    }
    static b_edge merge(b_edge e1, b_edge e2){
        if (e1 == null) return e2;
        if (e2 == null) return e1;
        if (e1.weight < e2.weight || (e1.weight == e2.weight && e1.inNode.index < e2.inNode.index)){
            b_edge et = e1.rson;
            pushDown(e1);
            e1.rson = merge(et, e2);
            if (e1.lson == null) {
                e1.lson = e1.rson;
                e1.rson = null;
            }else if (e1.rson.dist > e1.lson.dist){
                et = e1.rson;
                e1.rson = e1.lson;
                e1.lson = et;
            }
            if (e1.rson == null) {
                e1.dist = 0;
            }else{
                e1.dist = e1.rson.dist + 1;
            }
            return e1;
        }else{
            b_edge et = e2.rson;
            pushDown(e2);
            e2.rson = merge(e1, et);
            if (e2.lson == null) {
                e2.lson = e2.rson;
                e2.rson = null;
            }else if (e2.rson.dist > e2.lson.dist){
                et = e2.rson;
                e2.rson = e2.lson;
                e2.lson = et;
            }
            if (e2.rson == null) {
                e2.dist = 0;
            }else{
                e2.dist = e2.rson.dist + 1;
            }
            return e2;
        }
    }
    static void dismantle(b_node u){
        while (u.parent != null){
            b_node t = u;
            u = u.parent;
            for (int i = 0; i < u.children.size(); i++){
                b_node v = u.children.get(i);
                v.parent = null;
                if (v == t) continue;
                if (v.children.size() > 0){
                    R.add(v);
                }
            }
        }
    }
}
class b_node{
    int index;
    b_edge in; // initially null
    b_node prev; // null
    b_node parent; // null
    b_node ans;
    ArrayList<b_node> children = new ArrayList<>(); // empty
    b_edge Proot;
    b_node(int index){
        this.index = index;
        children = new ArrayList<>();
    }
}

class b_edge{
    int originalWeight;
    b_node outNode;
    b_node inNode;
    int weight;
    b_edge lson;
    b_edge rson;
    int dist;
    int lazy;
    b_edge(b_node outNode, b_node inNode, int weight){
        this.outNode = outNode;
        this.inNode = inNode;
        this.weight = weight;
        this.originalWeight = weight;
        this.dist = 0;
    }
}
class Reader {
    final private int BUFFER_SIZE = 1 << 16;
    private DataInputStream din;
    private byte[] buffer;
    private int bufferPointer, bytesRead;

    public Reader()
    {
        din = new DataInputStream(System.in);
        buffer = new byte[BUFFER_SIZE];
        bufferPointer = bytesRead = 0;
    }

    public Reader(String file_name) throws IOException
    {
        din = new DataInputStream(new FileInputStream(file_name));
        buffer = new byte[BUFFER_SIZE];
        bufferPointer = bytesRead = 0;
    }

    public String readLine() throws IOException
    {
        byte[] buf = new byte[64]; // line length
        int cnt = 0, c;
        while ((c = read()) != -1)
        {
            if (c == '\n')
                break;
            buf[cnt++] = (byte) c;
        }
        return new String(buf, 0, cnt);
    }

    public int nextInt() throws IOException
    {
        int ret = 0;
        byte c = read();
        while (c <= ' ')
            c = read();
        boolean neg = (c == '-');
        if (neg)
            c = read();
        do
        {
            ret = ret * 10 + c - '0';
        }  while ((c = read()) >= '0' && c <= '9');

        if (neg)
            return -ret;
        return ret;
    }

    public long nextLong() throws IOException
    {
        long ret = 0;
        byte c = read();
        while (c <= ' ')
            c = read();
        boolean neg = (c == '-');
        if (neg)
            c = read();
        do {
            ret = ret * 10 + c - '0';
        }
        while ((c = read()) >= '0' && c <= '9');
        if (neg)
            return -ret;
        return ret;
    }

    public double nextDouble() throws IOException
    {
        double ret = 0, div = 1;
        byte c = read();
        while (c <= ' ')
            c = read();
        boolean neg = (c == '-');
        if (neg)
            c = read();

        do {
            ret = ret * 10 + c - '0';
        }
        while ((c = read()) >= '0' && c <= '9');

        if (c == '.')
        {
            while ((c = read()) >= '0' && c <= '9')
            {
                ret += (c - '0') / (div *= 10);
            }
        }

        if (neg)
            return -ret;
        return ret;
    }

    private void fillBuffer() throws IOException
    {
        bytesRead = din.read(buffer, bufferPointer = 0, BUFFER_SIZE);
        if (bytesRead == -1)
            buffer[0] = -1;
    }

    private byte read() throws IOException
    {
        if (bufferPointer == bytesRead)
            fillBuffer();
        return buffer[bufferPointer++];
    }

    public void close() throws IOException
    {
        if (din == null)
            return;
        din.close();
    }
}