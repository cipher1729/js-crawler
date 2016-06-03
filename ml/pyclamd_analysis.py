import pyclamd

cd = pyclamd.ClamdAgnostic()
f = open("SUPERBAD1.txt", 'r')
f_l = f.read()
f_l = f_l.split("###############\n")
for item in f_l:
	if item == "":
		continue
	f1 = open("testFile.js", 'w')
	f1.write(item)
	f1.close()
	obj = cd.scan_file("/home/cipher1729/Desktop/netSec/ml/CS_6262/testFile.js")
	if(obj != None):
		print obj
		print "\n"
f.close()
