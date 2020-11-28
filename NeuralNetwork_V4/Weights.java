package NN_V4;

import java.util.Random;

/**
 * Class made to represent the weights going from one layer to another
 * 
 * @author Ian Gluesing
 *
 */

public class Weights {

	private double[][] weights;
	
	/**
	 * Create a 2D array of weights given the number of rows and columns
	 * The boolean variable empty tells the program whether or not to give the weights random values or 0
	 * 
	 * @param rows
	 * @param columns
	 * @param empty
	 */
	public Weights(int rows, int columns, boolean empty) {
		weights = new double[rows][columns];
		if(empty) {
			empty();
		} else {
			randomize();
		}
	}
	
	/**
	 * This constructor takes in a 2D array of weights and assigns those values to the current weights
	 * 
	 * @param weights
	 */
	public Weights(double[][] weights) {
		this.weights = weights;
	}
	
	/**
	 * This method takes in a layer of values and multiplies them out to the next layer
	 * This matrix multiplication has the weight matrix on the left and the value vector on the right
	 * 
	 * @param in
	 * @return
	 */
	public Layer predict(Layer in) {
		Layer output = new Layer();
		
		//Multiply W transpose by the input layer.
		//We have to multiply by the transpose of W because of the way the weight matrices are set up
		double[] values = new double[weights[0].length];
		for(int i = 0; i < weights[0].length; i++) {
			double sum = 0;
			//Compute the sum for each layer in the weight matrix
			for(int j = 0; j < weights.length; j++) {
				sum += weights[j][i] * in.getValues()[j];
			}
			//Add the sum to the output values after running the sum through the activation function
			values[i] = Functions.activationFunction(sum);
		}
		output.setValues(values, false);
		
		return output;
	}
	
	/**
	 * This matrix will subtract the gradient matrix times the learningrate from the current weights
	 * The input gradient weights should have already been average by the time this method is called
	 * 
	 * @param gradient
	 * @param learningRate
	 */
	public void addGradient(Weights gradient, double learningRate) {
		for(int i = 0; i < weights.length; i++) {
			for(int j = 0; j < weights[i].length; j++) {
				weights[i][j] -= gradient.value(i, j) * learningRate;
			}
		}
	}
	
	/**
	 * This method returns the gradient for a given sample at a given layer
	 * We multiply the input values by the delta to obtain a 2D gradient matrix
	 * 
	 * @param xInput
	 * @param delta
	 * @return
	 */
	public static Weights multiplyGrad(Layer xInput, double[] delta) {
		return multiply(xInput.getValues(), delta);
	}
	
	/**
	 * Multiplies 2 arrays to get a weight matrix of size left.length x right.length
	 * 
	 * @param left
	 * @param right
	 * @return
	 */
	public static Weights multiply(double[] left, double[] right) {
		double[][] w = new double[left.length][right.length];
		for(int i = 0; i < left.length; i++) {
			for(int j = 0; j < right.length; j++) {
				w[i][j] = left[i] * right[j];
			}
		}
		return new Weights(w);
	}
	
	/**
	 * Takes the current weight matrix and averages them out based on the input value
	 * Used for finding the average gradient over multiple samples
	 * 
	 * @param size
	 */
	public void ave(int size) {
		for(int i = 0; i < weights.length; i++) {
			for(int j = 0; j < weights[i].length; j++) {
				this.weights[i][j] /= size;
			}
		}
	}
	
	/**
	 * Adds the input weights to the current weights
	 * Used in finding the average gradient over multiple samples
	 * 
	 * @param w
	 */
	public void add(Weights w) {
		for(int i = 0; i < weights.length; i++) {
			for(int j = 0; j < weights[0].length; j++) {
				this.weights[i][j] += w.weights[i][j];
			}
		}
	}
	
	/**
	 * Empties the current weight matrix, setting it to a new empty matrix with the current number of rows and columns
	 */
	public void empty() {
		this.weights = new double[weights.length][weights[0].length];
	}
	
	/**
	 * When creating a new weight matrix, initially start with random weights
	 */
	private void randomize() {
		Random r = new Random();
		
		for(int i = 0; i < weights.length; i++) {
			for(int j = 0; j < weights[i].length; j++) {
				weights[i][j] = r.nextDouble();
			}
		}
	}
	
	/**
	 * Returns the value of weight at a certain index
	 * 
	 * @param i
	 * @param j
	 * @return
	 */
	public double value(int i, int j) {
		return weights[i][j];
	}
	
	/**
	 * Returns the number of rows of the current weight matrix
	 * Somewhat used for testing purposes
	 * 
	 * @return
	 */
	public int rows() {
		return weights.length;
	}
	
	/**
	 * Used to print out the values of the weights
	 */
	public void print2D() {
		for(int i = 0; i < weights.length; i++) {
			for(int j = 0; j < weights[i].length; j++) {
				System.out.print(weights[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	public String toString() {
		return "Rows: " + weights.length + " | Cols: " + weights[0].length;
	}
	
	
}
