package NN_V3;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Simple Neural network to predict the class of a number in the range of 0 - .25
 * Can now handle multiple samples using average gradient of all samples
 * 
 * @author Ian Gluesing
 *
 */

public class NeuralNetwork {
	
	//Initial weights 
	private static double[][] W1 = {{.15,.25},
									{.2,.3},
									{.35,.35}};

	private static double[][] W2 = {{.4,.5},
									{.45,.55},
									{.6,.6}};
	
	//Total number of predictions to get to the end of the neural network
	private static int numPredictions = 2;
	
	//ArrayList to store the weights and the average gradients
	private static ArrayList<double[][]> W = new ArrayList<double[][]>();
	private static ArrayList<double[][]> aveGradients = new ArrayList<double[][]>();

	
	public static void main(String[] args) {
		Random r = new Random();
		
		W.add(W1);
		W.add(W2);
		
		//For each W, add an empty gradient to the arraylist of gradients
		for(double[][] w: W) {
			aveGradients.add(empty(w.length, w[0].length));
		}
		
		ArrayList<double[]> samples = new ArrayList<double[]>();
		ArrayList<double[]> test = new ArrayList<double[]>();
		
		//Add 20 samples to the samples arraylist
		//10 from the class [1,0] and 10 from the class [0,1]
		for(int i = 0; i < 10; i++) {
			double num1 = ThreadLocalRandom.current().nextDouble(.125, .25);
			double num2 = ThreadLocalRandom.current().nextDouble(0, .125);
			double[] s1 = {num1, 0, 1,0};
			double[] s2 = {num2, 0, 0,1};
			samples.add(s1);
			samples.add(s2);
		}
		
		//Add four test samples
		for(int i = 0; i < 4; i++) {
			double[] s = {r.nextDouble() / 4, 0};
			test.add(s);
		}
		
		//Continually perform gradient descent until the condition that 
		// 80% of the test samples have less than 2% squared error for each class
		boolean go = true;
		for(int i = 0; go; i++) {
			int counter = 0;
			//Loop through each sample and predict the output
			for(double[] s: samples) {
				double[] xVal = Arrays.copyOf(s, s.length - 2);
				double[] output = predict(xVal, 1, numPredictions);
				//Backpropagate for sample
				backPropMain(output, s);
				
				//Check if the error condition is met
				if(Math.pow(s[s.length - 2] - output[0], 2) < .02 && Math.pow(s[s.length - 1] - output[1], 2) < .02) {
					counter++;
				}
			}
			
			go = counter > samples.size() * .8 ? false : true;
			
			//Apply the average of the gradients to the real W matrices and then reset the average
			// gradients to an empty matrix
			for(int j = 0; j < aveGradients.size(); j++) {
				ave(aveGradients.get(j), samples.size());
				updateW(j, aveGradients.get(j), 1);
				//print2D(aveGradients.get(j));
				aveGradients.set(j, empty(W.get(j).length, W.get(j)[0].length));
			}
			
			//Print out the epoch number after every 1000 epochs
			if(i % 1000 == 0) {
				System.out.println(i);
			}
		}
		
		//Go through each test sample and print the prediction
		for(int i = 0; i < test.size(); i++) {
			System.out.println(Arrays.toString(predict(test.get(i), 1, numPredictions)) + " | " + Arrays.toString(test.get(i)));
		}
		System.out.println(Arrays.toString(predict(samples.get(0), 1, numPredictions)) + " | " + Arrays.toString(samples.get(0)));
	}
	
	/**
	 * Somewhat clunky backprop method which will be fixed in the next version
	 * 
	 * @param output
	 * @param s1
	 */
	public static void backPropMain(double[] output, double[] s1) {
		double[] initialLoss = {output[0] - s1[s1.length - 2], output[1] - s1[s1.length - 1]};
		
		double[] inputLayer = Arrays.copyOf(output, output.length + 1);
		inputLayer[inputLayer.length - 1] = 1;

		
		//(Etotal/outO1 == EO1)*(outO1/netO1)  ,   (Etotal/outO2 == EO2)*(outO2/netO2)
		//EO1/netO1   ,   EO2/netO2
		double[] delta = {initialLoss[0] * (output[0] * (1 - output[0])), initialLoss[1] * (output[1] * (1 - output[1]))};
				
		double[] prevOutput = predict(inputLayer, 1, W.size() - 1);
		//outH1 * the delta
		double[][] grad = multiply(prevOutput, delta);
		
		int currentLevel = W.size();
		add(grad, currentLevel);
		backPropSubLevel(currentLevel - 1, delta, inputLayer);
	}
	
	public static void backPropSubLevel(int wLevel, double[] delta, double[] inputLayer) {
		double[][] currentW = W.get(wLevel);
		double[] hiddenLoss = new double[currentW.length];
		for(int i = 0; i < currentW.length; i++) {
			double sum = 0;
			for(int j = 0; j < delta.length; j++) {
				sum += delta[j] * currentW[i][j];
			}
			hiddenLoss[i] = sum;
		}
		
		double[] partialHLoss = new double[hiddenLoss.length];
		double[] outH = predict(inputLayer, wLevel, wLevel);
		
		for(int i = 0; i < partialHLoss.length; i++) {
			partialHLoss[i] = outH[i] * (1 - outH[i]);
		}
		
		
		double[][] grad = new double[currentW.length][currentW[0].length];
		double[] previousLayer = predict(inputLayer, 1, wLevel - 1);
		
		for(int i = 0; i < grad.length; i++) {
			for(int j = 0; j < grad[0].length; j++) {
				grad[i][j] = hiddenLoss[j] * partialHLoss[j] * previousLayer[i];
			}
		}
		
		
		if(wLevel - 1 > 0) {
			backPropSubLevel(wLevel - 1, delta, inputLayer);
		}
		
		add(grad, wLevel);
	}
	
	/**
	 * Returns an empty matrix with equal dimensions to the corresponding W matrix
	 * 
	 * @param rows
	 * @param cols
	 * @return
	 */
	public static double[][] empty(int rows, int cols){
		double[][] empty = new double[rows][cols];
		return empty;
	}
	
	/**
	 * Multiplies two 1D arrays together to get a 2D array
	 * 
	 * @param left
	 * @param right
	 * @return
	 */
	public static double[][] multiply(double[] left, double[] right){
		double[][] out = new double[left.length][right.length];
		
		for(int i = 0; i < left.length; i++) {
			for(int j = 0; j < right.length; j++) {
				out[i][j] = left[i] * right[j];
			}
		}
		
		return out;
	}
	
	/**
	 * Updates the current W matrix with the averaged out gradient for that layer
	 * 
	 * @param w
	 * @param gradient
	 * @param learningRate
	 * @return
	 */
	public static double[][] updateW(int w, double[][] gradient, double learningRate){
		double[][] wMatrix = W.get(w);
		
		for(int i = 0; i < wMatrix.length; i++) {
			for(int j = 0; j < wMatrix[i].length; j++) {
				wMatrix[i][j] = W.get(w)[i][j] - (gradient[i][j] * learningRate);
			}
		}
		
		return wMatrix;
	}
	
	
	/**
	 * Predicts the output starting from the currentLayer all the way to the end layer
	 * 
	 * @param input
	 * @param currentLayer
	 * @param endLayer
	 * @return
	 */
	public static double[] predict(double[] input, int currentLayer, int endLayer) {
		if(currentLayer > endLayer) {
			return input;
		}
		if(currentLayer == 1) {
			input = Arrays.copyOf(input, input.length + 1);
			input[input.length - 1] = 1;
		}
		
		//Create a new output matrix to store the output values for the next layer, also adding the bias term
		double[] output = new double[W.get(currentLayer - 1)[0].length + (currentLayer == numPredictions ? 0 : 1)];
		if(currentLayer != numPredictions) {
			output[output.length - 1] = 1;
		}
		
		
		//Multiply the current layer W matrix by the input layer to get the values for the output layer
		for(int i = 0; i < W.get(currentLayer - 1)[0].length; i++) {
			double sum = 0;
			for(int j = 0; j < W.get(currentLayer - 1).length; j++) {
				sum += W.get(currentLayer - 1)[j][i] * input[j];
			}
			output[i] = activationFunction(sum);
		}
		
		//If the currentLayer index is not equal to the output layer index, we have to predict again, at the next layer
		if(currentLayer != endLayer) {
			return predict(output, currentLayer + 1, endLayer);
		} else {
			return output;
		}
	}
	
	/**
	 * Prints out a 2D matrix in a way that is readable
	 * 
	 * @param output
	 */
	public static void print2D(double[][] output) {
		System.out.println("W matrix.");
		for (int i = 0; i < output.length; i++) {
			for (int j = 0; j < output[0].length; j++) {
				System.out.print(output[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println("\n");
	}
	

	/**
	 * Returns the average of a gradient matrix, given the size of the number of input samples
	 * 
	 * @param gradTotal
	 * @param size
	 */
	public static void ave(double[][] gradTotal, int size) {
		for(int i = 0; i < gradTotal.length; i++) {
			for(int j = 0; j < gradTotal[i].length; j++) {
				gradTotal[i][j] /= size;
			}
		}
	}
	
	/**
	 * Used to add another samples gradient to the current gradient for any layer
	 * 
	 * @param grad
	 * @param level
	 */
	public static void add(double[][] grad, int level) {
		double[][] sumG = aveGradients.get(level - 1);
		
		for(int i = 0; i < sumG.length; i++) {
			for(int j = 0; j < sumG[0].length; j++) {
				sumG[i][j] += grad[i][j];
			}
		}
	}

	/**
	 * Prints out an array in vector for
	 * 
	 * @param output
	 */
	public static void print(double[] output) {
		System.out.print("[ " + output[0] + ",\n");
		for (int i = 1; i < output.length - 1; i++) {
			System.out.println("  " + output[i] + ",");
		}
		System.out.println("  " + output[output.length - 1] + " ]");
	}
	
	/**
	 * Activation function we are using which is the logistic function
	 * 
	 * @param sum
	 * @return
	 */
	public static double activationFunction(double sum) {
		return (double)1 / (1 + Math.pow(Math.E, -sum));
	}
}
