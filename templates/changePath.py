import re
filePath = "./templates/index.html"

with open(filePath, "r") as f:
    content = f.read()

# content = """    <link href=/assets/css/app.9c1d24ed.css rel=preload as=style>
#     <link href=/assets/css/chunk-vendors.2e31bbc2.css rel=preload as=style>
#     <link href=/assets/js/app.0a10128a.js rel=preload as=script>
#     <link href=/assets/js/chunk-vendors.0195ce98.js rel=preload as=script>
#     <link href=/assets/css/chunk-vendors.2e31bbc2.css rel=stylesheet>
#     <script src=/assets/js/chunk-vendors.0195ce98.js></script>
#     <script src=/assets/js/app.0a10128a.js></script>"""

pattern = re.compile(r'(href=|src=)(/)(assets/.*?\.(?:js|css|ico))')

# print(pattern.findall(content))


def fun(s):
    group1 = s.group(1)
    # group2 = s.group(2)
    group3 = s.group(3)
    return group1 + "\"" + r"{{url_for('static',filename='" + group3 + r"')}}" + "\""


fixed_content = re.sub(pattern, fun, content)

with open(filePath, "w") as f:
    f.write(fixed_content)
