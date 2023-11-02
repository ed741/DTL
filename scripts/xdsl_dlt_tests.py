from xdsl.dialects import builtin
from xdslDLT import dialect as dlt

str1 = builtin.StringAttr("hi")
str2 = builtin.StringAttr("hi")
str3 = builtin.StringAttr("hii")


set1 = dlt.SetAttr([str1, str2, str3])

set2 = dlt.SetAttr([str2, str3])

print("set1: ", set1)
print("set2: ", set2)
print("set1 == set2: ", set1==set2)

input = '#dlt.set{"hi", "coolio"}'

