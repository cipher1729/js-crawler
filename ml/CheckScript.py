import uuid
import virustotal2
import sys
import time 
import os
import json
import itertools


mal_count = 0
file_list = []

def check_half(half,size,file_name):
    v = virustotal2.VirusTotal2(api_key)
    count = 0 
    if(half == True):
        print " Round 1 "
        start = 0
        end = size /2 
    else:
        start = size/2
        end  =  size 
        print "Round 2"

    for script in itertools.islice(script_list , start, end):
        u = str(uuid.uuid1())
        handle = open('./temp/%s'%u, 'w') 
        handle.write(script) 
        handle.close()
        file_list.append(u)
        report = v.scan('./temp/%s'%u,raw = True)
        print "Initial scan"
        count = count + 1 
        if(count == 4):
            count = 0
            time.sleep(65) 
            print " sleep after four requests"
    count = 0
    print "Sleep after all scans"
    time.sleep(100)

    for i,file in enumerate(file_list):
        try:
            rep  =  v.retrieve('./temp/%s'%file,raw ='True')

        except TypeError:
            continue

        print " Retrieving report  ",i
        count = count + 1 
        try:
            repdict = json.loads(str(rep))
        except ValueError:
            print "Json load failed"
            continue

        if(repdict['verbose_msg'] == 'Scan finished, information embedded'):
            if repdict['positives'] > 0:
                global mal_count 
                mal_count = mal_count + 1 
                desc = open('./temp/' + file,'r')
                handle = open(file_name,'a')
                handle.write(desc.read())
                handle.write('###############')
                desc.close()
                handle.close()
                print " Malicious Bitch!  ",mal_count

        if(count == 4):
            count = 0
            print " sleep after four requests"
            time.sleep(65)

        os.remove("./temp/" + file)
    del file_list[:]


 
api_key = "b71069af844053f57aeef323a7fd3293bc2f272d02cd659729c939547f7c2968"
file_name = 'MalJs.txt'
scripts_path  = sys.argv[1] 
if(os.path.exists(scripts_path)):
    pass
else:
    print "Enter valid file"

handle = open(scripts_path, 'r')
string = handle.read()
handle.close()
script_list = string.split('##########\n##########')
script_list = filter(None, script_list)
size =  len(script_list)
if(os.path.exists('temp')):
    pass
else:
    os.mkdir('temp')
check_half(True,size,file_name)
time.sleep(100)
print" Sleeping in between rounds"
check_half(False,size,file_name)
print " TOTAL MALICIOUS SCRIPTS DETECTED = %d", mal_count 
