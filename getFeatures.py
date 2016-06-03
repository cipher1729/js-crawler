import re
import math
from slimit.parser import Parser
from slimit import ast
from slimit.visitors import nodevisitor

def getEntropy(string):
	prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
	entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
	return entropy

def getFeatures(js):			
	
	nodeMap={}
	stringAssignments=0
	stringModFuncsCount=0;
	eventFuncsCount=0;
	nonPrintableCharactersInStrings=0;
	maxNonPrintableCharactersInStrings=0	
	hexaStrings=0;

	entropySum=0;
	entropyAvg=0;
	stringSum=0
	stringAvg=0
	stringCount=1
	lineSum=0
	lineAvg=0
	lineCount=0;

	wordCount=1;
	
	entropyScript=0; #entropy for the whole script
	evalCount=0;
	setTimeoutCount=0;	
	setIntervalCount=0;
	escapeCount=0;
	unescapeCount =0;
	linkCount=0
	execCount=0
	searchCount=0
	longStrings=0
	maxEntropy=-1
	maxLength=0; #max length for a string
	longVarOrFuncNames=0;	
	suspStrings=0
	domModFuncsCounter=0
	iframeCount=0	#strings with 'iframe' in them
	malTagCount=0
	
	whiteSpaces=0;
	totalJsLength=1; #total no of characters in the js
	keywordsCount =0
	keywordsRatio=0

	evilStrings=["evil","shell","spray","crypt"]
	domModFuncs=["getElementsByTagName","getElementById", "createElement", "createTextNode", "cloneNode","appendChild", "insertBefore", "removeChild", "replaceChild"]
	malTags=["script,object","iframe","embed"]
	keywords =["this","function","class","function*","yield","yield*","new","new.target","super","...obj","delete","void","typeof","in","instanceof","for","while","var","if","else","continue","break","return","with", "switch", "case","default", "label","throw","catch","finally", "alert"]
	

	keywordClasses =[ast.FunctionCall, ast.NewExpr,ast.Null, ast.Number, ast.String, ast.Identifier, ast.Regex, ast.NewExpr, ast.BracketAccessor, ast.DotAccessor,ast.Assign,  ast.GetPropAssign, ast.SetPropAssign, ast.UnaryOp, ast.BinOp, ast.Conditional]	
	stringModFuncs=["concat","replace", "split","toLowerCase", "toUpperCase"]
	eventFuncs =["addEventListener","attachEvent","dispatchEvent","fireEvent"]

	#iniitialize the parser
	parser = Parser()
	tree= None
	try:
		tree = parser.parse(js)
	except Exception,e:
		print "could not generate AST "
		print str(e)
		print "bad script " + str(js)
		ret ={}
		return ret
		#continue
	else:
		pass
		#print tree.to_ecma()	
		#basically go oveer the AST
		#print str(len(nodevisitor.visit(tree)))
		for node in nodevisitor.visit(tree):
		
			if type(node) is not list and node not in nodeMap and hasattr(node, "value"):
				#print "node is "  + str((node))	
				#searching for named functions
				#print "I am a weird node1!!!!!!!!!!!!!!!!!!!"	
				if "eval" in node.value:
					evalCount+=1
				if "setInterval" in node.value:
					setIntervalCount+=1
				if "setTimeout" in node.value:
					setTimeoutCount+=1
				if "escape" in node.value:
					escapeCount+=1
				if "unescape" in node.value:
					unescapeCount+=1
				if "link" in node.value:
					#print "link found!!!"		
					linkCount+=1
				if "search" in node.value:
					searchCount+=1
				if "exec" in node.value:
					execCount+=1
		
				if node.value in domModFuncs:			
					#print "identifier " + str(node.value)			
					domModFuncsCounter+=1
				if node.value in stringModFuncs:			
					#print "identifier " + str(node.value)			
					stringModFuncsCount+=1
				if node.value in eventFuncs:			
					#print "identifier " + str(node.value)			
					eventFuncsCount+=1
		 		
			
				for item in keywords:
					if item in node.value:
						#print "1" + node.value
						keywordsCount+=1
						break
				
				nodeMap[node]= True
		
			elif type(node) is not list and node not in nodeMap and hasattr(node, "identifier") and hasattr(node.identifier, "value"):
			
				#searching for named functions	
				if "eval" in node.identifier.value:
					evalCount+=1
				if "setInterval" in node.identifier.value:
					setIntervalCount+=1
				if "setTimeout" in node.identifier.value:
					setTimeoutCount+=1
				if "escape" in node.identifier.value:
					escapeCount+=1
				if "unescape" in node.identifier.value:
					unescapeCount+=1
				if "link" in node.identifier.value:
					#print "link found!!!"		
					linkCount+=1
				if "search" in node.identifier.value:
					searchCount+=1
				if "exec" in node.identifier.value:
					execCount+=1
		
				if node.identifier.value in domModFuncs:			
					#print "identifier " + str(node.value)			
					domModFuncsCounter+=1
				if node.identifier.value in stringModFuncs:			
					#print "identifier " + str(node.value)			
					stringModFuncsCount+=1
				if node.identifier.value in eventFuncs:			
					#print "identifier " + str(node.value)			
					eventFuncsCount+=1
		 		for item in keywords:
					if item in node.identifier.value:
						#print "2" + node.identifier.value
						keywordsCount+=1
						break
				
				nodeMap[node.identifier]= True
			if type(node) is list:
				for listItem in node:
					#print "I am a weird node!!!!!!!!!!!!"
					if listItem not in nodeMap and hasattr(listItem, "value"):
						if "eval" in listItem.value:
							evalCount+=1
						if "setInterval" in listItem.value:
							setIntervalCount+=1
						if "setTimeout" in listItem.value:
							setTimeoutCount+=1
						if "escape" in listItem.value:
							escapeCount+=1
						if "unescape" in listItem.value:
							unescapeCount+=1
						if "link" in listItem.value:
							#print "link found!!!"		
							linkCount+=1
						if "search" in listItem.value:
							searchCount+=1
						if "exec" in listItem.value:
							execCount+=1
			
						if listItem.value in domModFuncs:			
						#print "identifier " + str(node.value)			
							domModFuncsCounter+=1
						if listItem.value in stringModFuncs:			
							#print "identifier " + str(node.value)			
							stringModFuncsCount+=1
						if listItem.value in eventFuncs:			
							#print "identifier " + str(node.value)			
							eventFuncsCount+=1
					
				
						for item in keywords:
							if item in listItem.value:
								#print "1" + node.value
								keywordsCount+=1
								break
					
						nodeMap[listItem]= True
					
					elif listItem not in nodeMap and hasattr(node, "identifier") and hasattr(node.identifier, "value"):
					
						#searching for named functions	
						if "eval" in listItem.identifier.value:
							evalCount+=1
						if "setInterval" in listItem.identifier.value:
							setIntervalCount+=1
						if "setTimeout" in listItem.identifier.value:
							setTimeoutCount+=1
						if "escape" in listItem.identifier.value:
							escapeCount+=1
						if "unescape" in listItem.identifier.value:
							unescapeCount+=1
						if "link" in listItem.identifier.value:
							#print "link found!!!"		
							linkCount+=1
						if "search" in listItem.identifier.value:
							searchCount+=1
						if "exec" in listItem.identifier.value:
							execCount+=1
				
						if listItem.identifier.value in domModFuncs:			
							#print "identifier " + str(node.value)			
							domModFuncsCounter+=1
						if listItem.identifier.value in stringModFuncs:			
							#print "identifier " + str(node.value)			
							stringModFuncsCount+=1
						if listItem.identifier.value in eventFuncs:			
							#print "identifier " + str(node.value)			
							eventFuncsCount+=1
						for item in keywords:
							if item in listItem.identifier.value:
								#print "2" + node.identifier.value
								keywordsCount+=1
								break
						
						nodeMap[listItem.identifier]= True
					


	
			if isinstance(node, ast.VarDecl):
				#this is a var statement , increase keyword count		
				keywordsCount+=1	
	
			#every other node in the tree I take as a 'word', not sure about this	
			wordCount+=1;

			#entropy analysis and other analysis for strings
			if isinstance(node, ast.String):
				currEntropy = getEntropy(node.value)
				entropySum+= currEntropy
				if(currEntropy >  maxEntropy):
					maxEntropy = currEntropy
				#long strings, greater than 40 characters
				if(len(node.value)> 40):
						longStrings+=1;
				#current, max and avg lengths			
				currLength = len(node.value)
				#print  " current strings " + node.value + "with length is " + str(currLength)
				if(currLength > maxLength):
						maxLength = currLength;	
				stringSum += currLength
		
				#evil strings
				for evil in evilStrings:
						evilRe = re.compile(evil)
						if (len(evilRe.findall(node.value))>0):
							suspStrings+=1;	
				#iframestrings
				iframeRe = re.compile("iframe")
				if (len(iframeRe.findall(node.value))>0):
						iframeCount+=1;	
				#malicious tags
				for malTag in malTags:
						malTagRe = re.compile(malTag)
						if (len(malTagRe.findall(node.value))>0):
							malTagCount+=1;	
				isHexa= True
				nums= ['0','1','2','3','4','5','6','7','8','9']
				nonPrintableCharactersInStrings=0
				for c in node.value:
					if(ord(c)<=31):
						nonPrintableCharactersInStrings+=1;
					if c not in nums and (c>'f' or c>'F'):
						isHexa= False;
				if(nonPrintableCharactersInStrings > maxNonPrintableCharactersInStrings):
					maxNonPrintableCharactersInStrings = nonPrintableCharactersInStrings
				if(isHexa):
						hexaStrings+=1;
				stringCount+=1
	
									
			#long variable or function names
			if isinstance(node, ast.VarDecl) and len(node.identifier.value)> 40:
				longVarOrFuncNames +=1	
			if isinstance(node, ast.Identifier) and len(node.value) > 40:
				longVarOrFuncNames +=1	
			if isinstance(node, ast.Assign) and isinstance(node.right, ast.String): 
				#print "string assignment!!!"			
				stringAssignments+=1
			if isinstance(node, ast.VarDecl) and isinstance(node.initializer, ast.String): 
				#print "string assignment!!!"			
				stringAssignments+=1
		
	
		#no of occurences of eval function of the form 'eval('
		evalRe = re.compile("eval\(")
		setTimeoutRe = re.compile("setTimeout\(")
		setIntervalRe = re.compile("setInterval\(")
	

		#entropy for whole doc
		entropyScript = getEntropy(js)

		
		#split with lines, then with words
		for line in js.split("\n"):
			for c in line:
				if (c==" "):
					whiteSpaces+=1;
				totalJsLength+=1
			lineSum+= len(line);
			lineCount+=1
		
							

		entropyAvg = float(entropySum)/ stringCount;
		lineAvg= float(lineSum)/ lineCount;
		stringAvg = float(stringSum)/stringCount
	
		keywordsRatio = float(keywordsCount) /wordCount;

		ret ={}
		#eval, setTimeout, setInterval, exec and search and others
	 	#print "evalCount : "  + str(evalCount) +  "\n"
		ret["evalCount"] = str(evalCount)


		#print "setInterval :"  + str(setIntervalCount) +  "\n"
		ret["setInterval"] = str(setIntervalCount)
		#print "setintervalcount is "  + str(setIntervalCount)

		#print "setTimeout :"  + str(setTimeoutCount)  +  "\n"
		ret["setTimeout"] = str(setTimeoutCount)

		#print "link : " + str((linkCount)) +  "\n"
		ret["link"] = str(linkCount)

		#print  "search :" + str((searchCount)) +  "\n"
		ret["search"] = str(searchCount)
	
		#print "exec"  + str((execCount)) +  "\n"
		ret["exec" ] = str(execCount)
	
		#print "escape" + str((escapeCount)) +  "\n"
		ret["escape"] = escapeCount
		

		#print "unescape " + str((unescapeCount)) +  "\n"
		ret["unescape"] = unescapeCount


		#ratio	
		#print "keywords"  + str(keywordsCount)
		#print "total words"  + str(wordCount)
		#print "ratio of keywords to words :" + str(keywordsRatio) +  "\n"
		ret["ratio"] = str(keywordsRatio)

		#entropy
		#print "avg entropy is " + str( entropyAvg) +  "\n"
		ret["entropyAvg"]= str(entropyAvg)

		#print "entropy for whole doc is :"  + str(entropyScript)+  "\n"
		ret["entropyScript"] = str(entropyScript)

		#long strings
		#print "long strings :"  + str(longStrings) +  "\n"
		ret["longStrings"] = str(longStrings)		

		#max entropy
		#print "max entropy is " + str(maxEntropy) +  "\n"
		ret["maxEntropy"] =  str(maxEntropy)

		#average string lengths
		#print " avg string size" + str(stringAvg) +  "\n"
		ret["stringAvg"] = str(stringAvg)


		#max string length
		#print " max string length is " + str(maxLength) +  "\n"
		ret["maxLength"] = str(maxLength)
		

		#long variables or function anmes
		#print "long variables or function names : " + str(longVarOrFuncNames) +  "\n"
		ret["longVarFunc"] = str(longVarOrFuncNames)

		#string assignments	
		#print "no of string assignemnts " + str(stringAssignments) +  "\n"
		ret["stringAssignments"] = str(stringAssignments)	

		#string modyifyign functions
		#print " string modiying functions"  + str(stringModFuncsCount) +  "\n"
		ret["stringModFuncsCount"] =  str(stringModFuncsCount)	

		#no of event modiyfyig functions
		#print "event mod funcs " + str(eventFuncsCount) +  "\n"
		ret["eventFuncsCount"] = str(eventFuncsCount)

		#no of DOM modiyfing functions
		#print "DOM modifying funcions " + str(domModFuncsCounter) +  "\n"
		ret["domModFuncsCounter"]  = str(domModFuncsCounter)
	
		#evil strings
		#print "suspicious strings " + str(suspStrings) +  "\n"
		ret["suspStrings"] = str(suspStrings)		

		#white space %
		#print "whitespace " + str(whiteSpaces)
		#print "totalJslength "  + str(totalJsLength)
		#print "whitespaces % : " + str(float(whiteSpaces) * 100/ totalJsLength) +  "\n"
		ret["whiteSpaceRatio"] = str((float(whiteSpaces) * 100/ totalJsLength))

		#shell code	
		#print "hexaStrings " + str(hexaStrings) +  "\n"
		ret["hexaStrings"] = str(hexaStrings)
		
		#print " max non printable characters in strings"  + str(maxNonPrintableCharactersInStrings) +  "\n"
		ret["maxNonPrintableCharactersInStrings"] =  str(maxNonPrintableCharactersInStrings)


		#avg line length
		#print " avg line length " + str(lineAvg) +  "\n"
		ret["lineAvg"] =str( lineAvg)	
	
		#strings with iframes
		#print "strings with iframe in them  :" + str(iframeCount) +  "\n"
		ret["iframeCount"] = str(iframeCount)	

		#strings with malicious tags
		#print " strings with mal tags :" + str(malTagCount) +  "\n"
		ret["malTagCount"] = str(malTagCount)

		#length of the string in characters
		#print "length of string in characters :" + str(len(js))	 +  "\n"
		ret["jsLength"] = str(len(js))
	
		#print "word count "+ str(wordCount) + "\n"
		############################################
		#ratio of keywords to words
		#thresh=1
		############################################
		#functions for deobfuscation
		###########################################
		#code for deobfuscation routine
	
	
		return ret
	
	
