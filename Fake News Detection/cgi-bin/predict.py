import cgi
form = cgi.FieldStorage()
doc = form.getvalue('doc')
print(doc)