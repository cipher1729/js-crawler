from sys import argv

script, start, end, padding = argv;

f= open("alexaUrls.txt")

#maxUrls = 100 
d=0
startUrls=""
comma=","
for line in f:
	if (d >=int(start) and d< int(end)):
		currUrl = line[line.index(comma)+1:len(line)-1]+"\""
		startUrls += "\"http://" + currUrl + ",";
		d=d+1;
	else:
		d=d+1
		continue
for i in range(int(padding)):
	startUrls += "\"http://yelp.com\"" + ","; 
f.close()


f= open("data_crawler/spiders/getDataSpider.py","r")
beforeText = f.read()
#print beforeText
f.seek(0,0)
for line in f:
	#print line	
	if "start_urls" in line:		
		beforeText = beforeText.replace(line,"    start_urls=["+ startUrls+"]\n")
		break;
f.close()

#print beforeText

#f.write(beforeText)
#f.close()
f= open("data_crawler/spiders/getDataSpider.py","w")
#print beforeText
f.write(beforeText)
f.close()
#print startUrls;	
