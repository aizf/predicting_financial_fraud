from pyquery import PyQuery as pq

filePath = "./templates/index.html"

doc = pq(filename=filePath)

# print(doc)


def changePath(string):
    return "{{{{url_for('static',filename='{0}')}}}}".format(string[1:])


links = doc("head link")
for link in links.items():
    href = link.attr["href"]
    link.attr["href"] = changePath(href)

scripts = doc("body script")
for script in scripts.items():
    src = script.attr["src"]
    script.attr["src"] = changePath(src)

with open(filePath, "w") as f:
    f.write(str(doc))
# print(doc.html(method='html'))
