import dns.resolver

def recursive_dns_query(domain, query_type):
    # 初始化递归查询的域名和查询类型
    current_domain = domain
    current_query_type = query_type

    while True:
        # 执行 DNS 查询
        try:
            result = dns.resolver.resolve(current_domain, current_query_type)
            for rdata in result:
                print(f"{current_domain} ({current_query_type}): {rdata}")
        except dns.resolver.NoAnswer:
            print(f"No {current_query_type} records found for {current_domain}")
        except dns.resolver.NXDOMAIN:
            print(f"{current_domain} does not exist")
        
        # 获取 CNAME 记录（如果存在）
        try:
            result = dns.resolver.resolve(current_domain, "CNAME")
            for rdata in result:
                current_domain = str(rdata.target)
                current_query_type = query_type
                print(f"CNAME redirect to {current_domain}")
        except dns.resolver.NXDOMAIN:
            break

# 模拟递归查询示例
recursive_dns_query("www.baidu.com", "A")
