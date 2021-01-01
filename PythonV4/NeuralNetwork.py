import math




def activationFunction(value):
	return 1 / (1 + (math.e ** - value))


class Layer:
	def __init__(self):
		self.__values = []

	def setValues(self, vals, output = False):
		if not output:
			self.__values = vals
			self.__values.append(1)
		else:
			self.__values = vals

	def getValues(self):
		return self.__values

	def toString(self):
		if len(self.__values) > 0:
			out = '[ ' + str(self.__values[0])
			for i in range (1, len(self.__values)):
				out += '\n  ' + str(self.__values[i])

			out += ' ]'
			return out
		else:
			return '[ ]'

class Sample:
	def __init__(self, initialLayer, expectedOutput = []):
		self.__inputLayer = Layer()
		self.__inputLayer.setValues(initialLayer)
		if expectedOutput != []:
			self.__outputLayer = Layer()
			self.__outputLayer.setValues(expectedOutput, True)
		else:
			self.__outputLayer = None
		self.__currentLayers = [self.__inputLayer]
		print(self.__currentLayers)

	def addLayer(self, L):
		self.__currentLayers.append(L)

	def getInputLayer(self):
		return self.__inputLayer

	def getLayer(self, index):
		return self.__currentLayers[index]

	def getPrediction(self):
		lastLayer = self.__currentLayers[len(self.__currentLayers) - 1].getValues()
		return max(range(len(lastLayer)), key=lastLayer.__getitem__)

	def checkTruePrediction(self):
		p = self.getPrediction()
		return True if self.__outputLayer != None and self.__outputLayer.getValues()[p] == 1 else False

	def error(self):
		if self.__outputLayer != None:
			e = [x - y for x,y in zip(self.__currentLayers[len(self.__currentLayers) - 1].getValues(), self.__outputLayer.getValues())]
			return e
		else:
			return 0

	def sumSquaredError(self):
		e = self.error()
		return sum([x**2 for x in e])

	def removePredictedLayers(self):
		self.__currentLayers = self.__currentLayers[:1]




def main():
	pass




if __name__ == '__main__':
	main()