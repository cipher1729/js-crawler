import urllib2
from bs4 import BeautifulSoup
import pyhash 
from socket import timeout
import thread
import threading
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.dummy import Process
import hashlib
import cchardet
import sys

#reload(sys)  
#sys.setdefaultencoding('utf-8')

def getUnicode(data, new_coding = 'UTF-8'):
	encoding = cchardet.detect(data)['encoding']

	if new_coding.upper() != encoding.upper():
		print "new decoding detected "  + str(encoding)
		data = data.decode(encoding)
		#print "type is " + str(type(data))
	return data



ignore_js=["google","facebook"]
timeOut = 3;
p =ThreadPool(5)
myLock = threading.Lock()

class ScriptItem:

	textHash= "None"
	text="None";
	inIframe= False;
	isInternal= False;
	isIframeHeightZero= False;
	isIframeWidthZero=False;
	parent= None
	sourceUrl= "None"
	
	def __init__(self):
		self.textHash= "None"
		self.text="None";
		self.inIframe= False;
		self.isInternal= False;
		self.isIframeHeightZero= False;
		self.isIframeWidthZero=False;
		self.parent= None
		self.sourceUrl= "None"		


def initPool():
	p =ThreadPool(5) 

def getExtJs(js_url):
	try:
		res = urllib2.urlopen(js_url, None, timeOut)						
		external_js = str(res.read())
		res.close()
		return str(external_js)
	except timeout:
		return "None"
		pass
	except:
		return "None"
		pass
		
def iframeThread(iframeSrc, soup, scriptContainer, oldScripts, domain_name, inIframeIter, inPre):
	if len(iframeSrc)!=0:		
		try:	
			#print "Attempting to fetech iframe" + iframeSrc	
			#totally local iframe source			
			if(iframeSrc[0]=='/'):			
				if(not("www" in iframeSrc)):
					iframeSrc = domain_name[0:len(domain_name)-1]+ iframeSrc;						
				else:
					iframeSrc = "http://"+ iframeSrc[iframeSrc.find("www"):];			
			#this is local but does not being with two slashes			
			elif(not("http" in iframeSrc) and (not("www" in iframeSrc)) ):
				iframeSrc = domain_name+ iframeSrc;		
			#print "Final iframe source " + iframeSrc			
			res= urllib2.urlopen(iframeSrc, None, timeOut)			
			remoteWebpage = str(res.read())
			res.close()
			#print " Fetched iframe"
		except timeout:
			#print "timeout while fetching iframe"
			pass
		except:
			#print "Could not fetch "					
			pass
			 
		else:
			findAllJs(soup,remoteWebpage,scriptContainer, oldScripts,domain_name, True, False)


	
#POST: give back iframe sources
def getAllJs(soup, domain_name, oldScripts):
	print "Crawling domain " + domain_name;	
		

	allScripts= []

	#initialize the hasher
	hasher = pyhash.city_64()

	#get the sources for all the iframes
	iframes_this_page =  soup.find_all('iframe')
	iframe_sources = []	
	for item in iframes_this_page:
		if( not item.has_attr('src') ):
			continue
		iframe_sources.append(item['src']) 

	#init
	
	#get the scripts residing on the home page
	#findAllJs(soup,"home",scriptContainer, oldScripts,domain_name, False, False)

	allThreads=[]
	mainThread = threading.Thread(target = findAllJs,args = (soup,"home",allScripts, oldScripts,domain_name, False, False))	
	allThreads.append(mainThread)
	mainThread.start()

	#get the scripts residing on the iframes	
	#print "...................................................................................POST: IFRAME FETCH......................................"
	
	

	for iframeSrc in iframe_sources:
	#boundary case of iframe URL being not null
		t = Process(target = iframeThread, args =  (iframeSrc, soup, allScripts, oldScripts, domain_name, True, False))
		allThreads.append(t)
		t.start()
		pass
	
	#print "Trying to join all main threads!!"
	for th in allThreads:
		th.join(timeout=2)
		continue
	print "All main threads joined successfuly"
	#print "script container contains " + str(len(allScripts)) +  " scripts"
	return allScripts

#helper function for extracting all JS from a page
def findAllJs(soup, remoteWebPage, allScripts, oldScripts, domain_name, inIframeIter, inPre ):
		
		hasher = pyhash.city_64()
				
		if inIframeIter==False:
			scripts_this_page = soup.find_all('script')
		else:
			print "Processing"
			soup = BeautifulSoup(remoteWebPage,'html.parser')
			scripts_this_page = soup.find_all('script')
		
		allJsThreads=[]		
		allExtJs=[]
		allPos=[]
		allItems=[]
		d=0	
		for item in scripts_this_page:
			
			
			scriptItem= ScriptItem();	
			#this is an external JS		
			if ( item.has_attr('src') ):
				js_url = item['src']
				#print "Attempting to fetch external URL" + js_url;

				#script is external js file but if hosted internally
				if( len(js_url)!=0 and js_url[0]=='/'):
					if((not("www" in js_url)) and js_url[1]=='/'):						
						js_url= domain_name[0:len(domain_name)-1]+ js_url; 
					else:	
						js_url= "http://"+js_url[js_url.find("www"):];						
					#print "new url is " + js_url;	
				elif(not("http" in js_url) and not("www" in js_url)) :
					js_url= domain_name+ js_url; 
					#print "new url is " + js_url;		
				if ".js" in js_url and "google" not in js_url:					
					"""
					currJsThread = threading.Thread(target = jsThread,args = (js_url,scriptContainer, inIframeIter, inPre,item))	
					allJsThreads.append(currJsThread)
					currJsThread.start()
					pass
					"""
					#allPos.append(d)
					allItems.append(item)
					allExtJs.append(js_url)
			#else work as internal scritp
			else:
				scriptItem.isInternal=  True;
				#internal_js = str(item)
				"""
				try:
						scriptItem.text = internal_js
				except UnicodeEncodeError, e:		
					internal_js = internal_js.encode("utf-8")
					scriptItem.text = internal_js
				"""
				#tempItem = getUnicode (str(item))
				scriptItem.text =  unicode(str(item),'utf-8', errors='replace')
				#print "type of text item for internal scripts is " + str(type(scriptItem.text))
	
				scriptItem.isIframeHeightZero= False
				scriptItem.isIframeWidthZero = False
				#is script in iframe?If yes, also check if in zero height or zero with iframe		
				if(inIframeIter==True):
					scriptItem.inIframe = True;
					scriptParents= item.findParents('iframe');
					if(len(scriptParents)>0):				
						if (scriptParents[0].has_attr('height') and  scriptParents[0]['height']==0 ):
							scriptItem.isIframeHeightZero= True; 
						if (scriptParents[0].has_attr('width') and  scriptParents[0]['width']==0 ):
							scriptItem.isIframeWidthZero= True; 	
				#scripts are appended only if there is some valid text for that
				if (scriptItem.text!="None"):
					#add the parent for the internal scripts, src remains none
					scriptItem.sourceUrl = "None"							
					scriptItem.parent = str(item.parent.name)
					scriptItem.textHash = str(hashlib.sha512(scriptItem.text.encode('utf-8')).hexdigest())
					#print "parent  is  "  + str(item.parent.name)
					#print "trying to acquire lock"
					myLock.acquire()
					allScripts.append(scriptItem);
					myLock.release()
					#print "releasing lock"
			d+=1
		
		print "Using thread pool" 
		"""
		for js in allJsThreads:
			try:
				js.join(timeout=0.5)		
				if js.is_alive():
					continue
			except:
				continue
		print "external JS joined !"	
		"""
	
		fetchedExtJs = p.map(getExtJs, allExtJs)
		#p.join(

		#print "number of ext JS is " + str(len(fetchedExtJs))
		
		d=0
		for item in fetchedExtJs:
			#print item	
			if item!="None":
				#print "going to add......................................................."	
				scriptItem= ScriptItem()
				scriptItem.isIframeHeightZero= False
				scriptItem.isIframeWidthZero=False
				scriptItem.isInternal = False;
			
				"""	
				try:
						scriptItem.text = item
				except UnicodeEncodeError, e:	
						tempItem= item.encode("utf-8")
						scriptItem.text = tempItem
				"""
				#tempItem = getUnicode (str(item))
				#tempItem = getUnicode (str(item))
				scriptItem.text =  unicode(str(item),'utf-8',errors='replace')
				#print "type of text item for external scripts is " + str(type(scriptItem.text))
	


				scriptItem.inIframe = inIframeIter
				scriptItem.parent = str(allItems[d].parent.name)
				#print "this one has url  "  + str(allExtJs[d]) 
				scriptItem.sourceUrl = str(allExtJs[d])
				scriptItem.textHash = str(hashlib.sha512(scriptItem.text.encode('utf-8')).hexdigest())	
				myLock.acquire()
				allScripts.append(scriptItem)
				myLock.release()
			d+=1
