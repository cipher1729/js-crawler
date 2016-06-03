from urlparse import urlparse
from bs4 import BeautifulSoup
import tldextract

def getUrlStats(urls_this_page):
	countInternalUrls=0;
	countExternalUrls=0;
	allUniqueHostNames=[];	
	
	#GO URL BY URL
	for item in urls_this_page:
		if(item['href'][0] == '/'):
			countInternalUrls+=1;
		else:
			#url is external to the current domain			
			countExternalUrls+=1;
			#add it to the unique hostnames list
			newDomain = tldextract(item).domain
			if newDomain not in allUniqueHostNames:
				allUniqueHostNames.append(newDomain);
		#do something depeneding on type of url , is it an image, video, binary


				



