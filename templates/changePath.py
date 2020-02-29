import re
filePath = "./templates/index.html"

with open(filePath, "r") as f:
    content = f.read()

pattern = re.compile(r'(href=|src=)(/)(.*?\.(?:js|css|ico))')

# print(pattern.findall(content))


def fun(s):
    group1 = s.group(1)
    # group2 = s.group(2)
    group3 = s.group(3)
    return group1 + "\"" + r"{{url_for('static',filename='" + group3 + r"')}}" + "\""


fixed_content = re.sub(pattern, fun, content)

with open(filePath, "w") as f:
    f.write(fixed_content)
