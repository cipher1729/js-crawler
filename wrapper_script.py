import subprocess


print "STARTING DOCKER\n"
p = subprocess.Popen("docker run -p 3050:6050 scrapinghub/splash", shell=True)
subprocess.call("scrapy crawl getData", shell=True)
print "ENDING DOCKER\n"
p = subprocess.call("docker kill scrapinghub/splash", shell=True)
