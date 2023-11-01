import dns.resolver
# target = input("query target = ")
# Type = input("type = ")
# RD = input("recursive or not = ")
# print(target, Type, RD)
target = "www.baidu.com"
Type = 'a'
RD = '0'
my_resolver = dns.resolver.Resolver()
if RD == '0': # 迭代
    q1 = my_resolver.query(".", 'ns')
    root_ip = q1.response.additional[0][0].to_text()
    print(f"get root ip = {root_ip}, use this to query TLD")

    my_resolver.nameservers=[root_ip]
    q2 = my_resolver.query(target, raise_on_no_answer=False)
    tld_ip = q2.response.additional[0][0].to_text()
    print(f"get tld ip = {tld_ip}, use this to query authority")
    
    my_resolver.nameservers=[tld_ip]
    q3 = my_resolver.query(target, raise_on_no_answer=False)
    authority = q3.response.additional[0][0].to_text()
    print(f"get authority ip = {authority}, use this to query target")

    my_resolver.nameservers=[authority]
    q4 = my_resolver.query(target, Type, raise_on_no_answer=False)
    target_ip = q4.response.answer[0][0].to_text()
    print(f"target = {target_ip}")
else:
    q1 = my_resolver.query(target, Type, raise_on_no_answer=False)
    target_ip = q1.response.answer[0][0].to_text()
    print(f"target = {target_ip}")