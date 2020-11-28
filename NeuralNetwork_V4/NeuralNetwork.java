package NN_V4;

import java.util.ArrayList;

/**
 * Class to represent our neural network
 * 
 * @author Ian Gluesing
 *
 */

public class NeuralNetwork {

	//Three arraylists to store the weights for this neural network, average gradients, and samples
	private ArrayList<Weights> weights = new ArrayList<Weights>();
	private ArrayList<Weights> aveGradients = new ArrayList<Weights>();
	private ArrayList<Sample> samples = new ArrayList<Sample>();
	private long trainingTime = 0;

	/**
	 * Constructor for this NN taking in the number of non bias nodes in each layer
	 * We make a call to initialize() and emptyGradients() to initialize the weights and 
	 * empty gradients
	 * 
	 * @param nonBias
	 */
	public NeuralNetwork(int[] nonBias) {
		initialize(nonBias);
		emptyGradients(nonBias);
	}
	
	/**
	 * This is the method used for training the neural network
	 */
	public void train() {		
		//Get starting time to compute the number of seconds for training
		long start = System.currentTimeMillis();
		int currentEpoch = 0;
		while(true) {
			currentEpoch++;
			System.out.println("Epoch: " + currentEpoch);
			
			//At the start of each epoch, we will empty the average gradient at each layer
			emptyGradients();
			
			//Counter to keep track of the total number of samples that satisfy the error threshold
			int counter = 0;
			double error = 0;
			for(Sample s: samples) {
				//Loop through each sample and predict the output of that sample at each layer
				predict(s);
				//Increment the counter if the sample passes the error threshold
				counter = s.errorThreshold() ? counter + 1 : counter;
				//Increment the squared error
				error += s.sumSqError();
				//Backpropogate starting at the last level, using the final error as the first delta
				backPropagate(aveGradients.size() - 1, s.error(), s);
			}
			//Calcualtes the average error
			error = error / samples.size();
			
			//If we successfully pass the condition, end the timer, and set the training time in seconds
			if(testEndCondition(counter)) {
				long end = System.currentTimeMillis();
				trainingTime = ((end - start) / 1000);
				return;
			}
			//Print squared error at each epoch.
			System.out.println("Ave squared error: " + error);
		}
	}
	
	/**
	 * This method tests whether or not a preset condition has been met yet
	 * Right now, it tests if 90% of samples are classified correctly
	 * If the test is not met, it updates the weights, using the average gradients
	 * 
	 * @param counter
	 * @return
	 */
	private boolean testEndCondition(int counter) {
		if(counter > samples.size() * .9) {
			return true;
		} else {
			//Loops through the sum of the gradients and then converts them to the average using the size of the samples
			for(Weights w: aveGradients) {
				w.ave(samples.size());
			}
			//Add the average gradients to the current weights
			for(int i = 0; i < weights.size(); i++) {
				weights.get(i).addGradient(aveGradients.get(i), 3);
			}
			return false;
		}
	}
	
	/**
	 * This method takes all of the current W matrices in the average gradients and sets
	 * them back to 0
	 */
	private void emptyGradients() {
		for(Weights w: aveGradients) {
			w.empty();
		}
	}
	
	/**
	 * Backprop method for the neural network
	 * Compute the gradient for each sample at each level and add
	 * that to the sum of the gradients
	 * 
	 * @param wLevel
	 * @param delta
	 * @param s
	 */
	private void backPropagate(int wLevel, double[] delta, Sample s) {
		//Get the weights for the current layer
		Weights currentWeights = weights.get(wLevel);
		
		//Compute the left and right side of the hadamard product to determine the new delta
		double[] left = hadaLeft(s.getLayer(wLevel));
		double[] right = hadaRight(currentWeights, delta);
		
		//Take the dot product of the left and right vectors to get the new delta
		double[] deltaNew = dot(left, right);
		
		//Once we get to the last layer of weights, we stop, otherwise we keep going
		if(wLevel > 0) {
			backPropagate(wLevel - 1, deltaNew, s);
		}
		
		//Compute the gradient for this layer for this sample
		Weights gradient = Weights.multiplyGrad(s.getLayer(wLevel), delta);
		
		//Add the gradient for this layer to the average gradient for this layer for this epoch
		aveGradients.get(wLevel).add(gradient);
	}
	
	/**
	 * Computes and returns the dot product of two vectors
	 * 
	 * @param v1
	 * @param v2
	 * @return
	 */
	private double[] dot(double[] v1, double[] v2) {
		double[] output = new double[v1.length];
		
		for(int i = 0; i < v1.length; i++) {
			output[i] = v1[i] * v2[i];
		}
		return output;
	}
	
	/**
	 * Computes the right half of the hadamard product
	 * This method multiplies the previous delta over each weight in the current weight matrix
	 * 
	 * @param w
	 * @param deltaPrev
	 * @return
	 */
	private double[] hadaRight(Weights w, double[] deltaPrev) {
		double[] output = new double[w.rows()];
		
		for(int i = 0; i < output.length; i++) {
			double sum = 0;
			for(int j = 0; j < deltaPrev.length - 1; j++) {
				sum += w.value(i, j) * deltaPrev[j];
			}
			output[i] = sum;
		}
		return output;
	}
	
	/**
	 * This function computes the left half of the hadamard product to compute the new delta
	 * We calculate f(x)(1 - f(x)) for each layer value, which is the derivative of the activation function
	 * 
	 * @param l
	 * @return
	 */
	private double[] hadaLeft(Layer l) {
		double[] output = new double[l.getValues().length];
		double[] values = l.getValues();
		//Loops through each value in the layer and computes the derivative of the activation function
		for(int i = 0; i < output.length; i++) {
			output[i] = values[i] * (1 - values[i]);
		}
		return output;
	}

	/**
	 * Predicts the output of a sample given the input layer
	 * 
	 * @param input
	 * @return
	 */
	public Layer predict(Sample input) {
		//Set this to be the input layer
		Layer currentLayer = input.getInputLayer();
		//Remove all previous predictions at each layer for this sample
		input.removeAllLayers();
		//Adds the initial input layer
		input.addLayer(input.getInputLayer());
		
		//We then assign the prediction at the next layer to the current layer
		for(Weights w: weights) {
			currentLayer = w.predict(currentLayer);
			//Add the prediction to the input samples layers
			input.addLayer(currentLayer);
		}
		//Once we have gone through all of the predictions, this layer is now the output/final layer
		// so we can set its values to not have the bias term
		currentLayer.setValues(currentLayer.getValues(), true);
		
		return currentLayer;
	}
	
	/**
	 * Add a sample to the neural networks list of samples
	 * 
	 * @param s
	 */
	public void addSample(Sample s) {
		samples.add(s);
	}
	
	/**
	 * Initialize each layer of weights using the number of non bias terms in each layer
	 * 
	 * @param nonBias
	 */
	private void initialize(int[] nonBias) {
		for (int i = 0; i < nonBias.length - 1; i++) {
			weights.add(new Weights(nonBias[i] + 1, nonBias[i + 1], false));
		}
	}
	
	/**
	 * Initialize each layer of gradients to be an empty matrix with the same dimension of each weight matrix
	 * 
	 * @param nonBias
	 */
	private void emptyGradients(int[] nonBias) {
		for(int i = 0; i < nonBias.length - 1; i++) {
			aveGradients.add(new Weights(nonBias[i] + 1, nonBias[i + 1], true));
		}
	}
	
	/**
	 * Returns training time for the neural network
	 * 
	 * @return
	 */
	public long trainingTime() {
		return trainingTime;
	}
	
	public ArrayList<Sample> getSamples(){
		return samples;
	}
}
