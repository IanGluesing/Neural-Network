//Name of the package that this file is in in Eclipse.
package NN;

/**
 * 
 * @author Ian Gluesing
 *	First attempt at creating a Neural Network without using any libraries
 *	This initial attempt only uses one sample
 *
 *	This program works for just this sample.
 */

public class NN {

	public static void main(String[] args) {
		
		//Input for the neural network, there is no bias anywhere in this version
		double[] inputLayer = {1,
							   1,
							   1};
		
		//There are the vector that will store the outputs from each layer
		//In this version there are two hidden layers, and one output layer
		double[] hiddenOne;
		double[] hiddenTwo;
		double[] output;
		 
		//This is the first weight matrix, this matrix maps the input to the first hidden layer.
		// In order to get the first part of the hidden layer we will do transpose(W1) * inputLayer.
		// So for this W and all of the others, it is in the form of W, not W transpose
		double[][] W1 = {{.1,.1,.1,.1},
				  		{.1,.1,.1,.1},
				  		{.1,.1,.1,.1}};
		
		//This W is a 4x3 matrix, so to get the output for the second hidden layer, we will have to do
		// W transpose time hidden layer one.
		double[][] W2 = {{2,2,2},
						{2,2,2},
						{2,2,2},
						{2,2,2}};
		
		//Similar explanation for this matrix as well
		double[][] W3 = {{1,1},
						  {1,1},
						  {1,1}};
		
		
		//Get initial predictions using the given weights at each layer
		hiddenOne = predict(inputLayer, W1);
		hiddenTwo = predict(hiddenOne, W2);
		output = predict(inputLayer, W3);

		
		//For this sample, lets assume its output should be [0,1]
		int[] y = {0,
				   1};		
		
		//This loop will perform gradient descent on each W matrix ten times
		for(int i = 0; i < 10; i++) {
			//Get the current loss with the currnent W matrices
			double[] initialLoss = {output[0] - y[0], output[1] - y[1]};
			
			//Get the gradient for layer 3 and then update the W matrix
			double[][] gradientW3 = gradient(inputLayer, initialLoss);
			W3 = updateW(W3, gradientW3, 1);
			//Now find the loss for the second hidden layer and then find the gradient and update the second W matrix
			double[] hiddenTwoLoss = loss(hiddenTwo, W3, initialLoss);
			double[][] gradientW2 = gradient(hiddenOne, hiddenTwoLoss);
			W2 = updateW(W2, gradientW2, 1);
			double[] hiddenOneLoss = loss(hiddenOne, W2, hiddenTwoLoss);
			double[][] gradientW1 = gradient(inputLayer, hiddenOneLoss);
			W1 = updateW(W1, gradientW1, 1);
			
			
			//Get the new values at each layer.
			hiddenOne = predict(inputLayer, W1);
			hiddenTwo = predict(hiddenOne, W2);
			output = predict(inputLayer, W3);
		}
		
		print2D(W1);
		print2D(W2);
		print2D(W3);
	
		
		print(output);
		
	}
	
	/**
	 * Simple method to print out an array in a vector like output
	 * 
	 * @param output
	 */
	public static void print(double[] output) {
		System.out.print("[ " + output[0] + ",\n");
		for(int i = 1; i < output.length - 1; i++) {
			System.out.println("  " + output[i] + ",");
		}
		System.out.println("  " + output[output.length - 1] + " ]");
	}
	
	/**
	 * Method to print out a 2d array
	 * 
	 * @param output
	 */
	public static void print2D(double[][] output) {
		System.out.println("W matrix.");
		for(int i = 0; i < output.length; i++) {
			for(int j = 0; j < output[0].length; j++) {
				System.out.print(output[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println("\n");
	}
	
	/**
	 * Function to return the value of the sigmoid function at the given input
	 * 
	 * @param sum
	 * @return Value of activation function at the given value
	 */
	public static double activationFunction(double sum) {
		return (double)1 / (1 + Math.pow(Math.E, -sum));
	}
	
	/**
	 * Method to update the given W matrix weights using the gradient 
	 * 
	 * @param W
	 * @param gradient
	 * @param learningRate
	 * @return Updated W matrix with new weights
	 */
	public static double[][] updateW(double[][] W, double[][] gradient, double learningRate){
		for(int i = 0; i < W.length; i++) {
			for(int j = 0; j < W[i].length; j++) {
				W[i][j] -= learningRate * gradient[i][j];
			}
		}
		
		return W;
	}
	
	/**
	 * Determine the gradient at the given layer by multiplying the x values by the previous loss
	 * 
	 * @param x
	 * @param previousLoss
	 * @return
	 */
	public static double[][] gradient(double[] x, double[] previousLoss){
		double[][] grad = new double[x.length][previousLoss.length];
		
		//We multiply the x layer values by the previous loss, creating a new matrix the size of that layers W
		// This new matrix becomes the gradient that will change the W for that layer.
		for(int i = 0; i < x.length; i++) {
			for(int j = 0; j < previousLoss.length; j++) {
				grad[i][j] = x[i] * previousLoss[j];
			}
		}
		
		return grad;
	}
	
	/**
	 * 	Determine the loss at the given layer
	 * 
	 * @param x
	 * @param W
	 * @param lossPrevious
	 * @return
	 */
	public static double[] loss(double[] x, double[][] W, double[] lossPrevious) {
		//This first for loop determines the values of the derivative of the logistic/sigmoid function
		// with respect to the input
		double[] hadamardLeft = new double[x.length];
		for(int i = 0; i < x.length; i++) {
			hadamardLeft[i] = x[i] * (1 - x[i]);
		}
		
		//Find the loss over each weight in W
		double[] hadamardRight = new double[W.length];
		for(int i = 0; i < W.length; i++) {
			double sum = 0;
			for(int j = 0; j < W[i].length; j++) {
				sum += W[i][j] * lossPrevious[j];
			}
			hadamardRight[i] = sum;
		}
		
		double[] newLoss = new double[x.length];
		
		//We then multiply both sides together to find the new loss for this layer
		for(int i = 0; i < newLoss.length; i++) {
			newLoss[i] = hadamardLeft[i] * hadamardRight[i];
		}
		
		return newLoss;
	}
	
	/**
	 * Method to predict the output at the next layer given the current xValues and their associated weights
	 * 
	 * @param xValues
	 * @param weights
	 * @return the values at the next layer
	 */
	public static double[] predict(double[] xValues, double[][] weights) {
		double[] ans = new double [weights[0].length];
		
		for(int i = 0; i < weights[0].length; i++) {
			double sum = 0;
			//W transpose times x, then sum all of them up
			for(int j = 0; j < weights.length; j++) {
				sum += weights[j][i] * xValues[j];
			}			
			//Evaluate the sigmoid function with the given some and set the next layer value to be this value
			ans[i] = activationFunction(sum);
		}
		
		return ans;
	}
	
}
