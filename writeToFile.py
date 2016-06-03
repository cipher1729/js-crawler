import os
import subprocess
import datetime
import hashlib

def writeToFile(start_urls, website_counter,internal_scripts_src, external_scripts_src):
	external_scripts=[]
	internal_scripts=[]
	#Create a folder for storing javascripts unless already present
	for item in external_scripts_src:
		external_js = item
		external_js_hashname = hashlib.sha512(external_js.encode('utf-8')).hexdigest()
		external_scripts.append(external_js_hashname)
		external_js_hashname = str(external_js_hashname) + ".js"
		#check if this script has already been stored
		#print "Making file " + external_js_hashname	
		f_js = open('/home/group9p1/javascripts/' + start_urls[website_counter] + '/'  + external_js_hashname, 'w' )
		f_js.write(external_js.encode('utf-8'))
		f_js.close()

	for item in internal_scripts_src:
		internal_js = item
		internal_js_hashname = hashlib.sha512( internal_js.encode('utf-8') ).hexdigest()
		internal_scripts.append(internal_js_hashname)
		internal_js_hashname = str(internal_js_hashname) + ".js"
		#check if this script has already been stored
		f_js = open('/home/group9p1/javascripts/' + start_urls[website_counter] + '/'   + internal_js_hashname, 'w' )
		f_js.write(internal_js.encode('utf-8'))
		f_js.close()	
		#print
	
