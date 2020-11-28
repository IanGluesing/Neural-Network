package NN_V4;

import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

/**
 * 
 * Main class for predicting handwritten numbers from the MNIST dataset.
 * The current implementation reads in a small balanced subset of the training
 * samples and then tests on the entire training set because the output is already labeled.
 * It just makes testing purposes easier.
 * 
 * @author Ian Gluesing
 *
 */

public class Main {
	
	//Neural network object that will represent the main neural network
	private static NeuralNetwork NN;
	//ArrayList to store all of the sample objects, which each store a sample
	private static ArrayList<Sample> samps = new ArrayList<Sample>();

	public static void main(String[] args) {
		//Read the input from the training data file
		readTrainingSet("train.csv");
		
		//Train the neural network given the trianing data we read in
		NN.train();
		
		//Read all of the training samples into the samps ArrayList
		readALL();
		
		
		int count = 0;
		//Predict the output for each sample and then determine if the prediction is correct
		for(Sample s: samps) {
			NN.predict(s);
			if(s.checkTruePrediction()) {
				count++;
			}
		}
		System.out.println("Size of training set: " + NN.getSamples().size() + " | " + NN.getSamples().size() / 10 + " per class");
		System.out.println("Total predictions correct: " + count + " out of " + samps.size());
		System.out.println("Training time: " + NN.trainingTime() / 60.0 + " minutes");
	}
	
	public static void readTrainingSet(String fileName) {
		File f = new File(fileName);
		Scanner in = null;
		
		try {
			in = new Scanner(f);
		} catch(FileNotFoundException e) {
			e.printStackTrace();
		}
		String[] line = in.nextLine().split(",");
		
		//This is an array to store the number of non bias nodes in each layer of the neural network
		int[] nonBias = {line.length - 1, 15, 15, 10};
		//Instantiate a neural network object using the array of number of non bias nodes
		NN = new NeuralNetwork(nonBias);
		
		//This array is to keep track of the number of each sample we have read in so far.
		//Index 0 represents the number of samples that correspond to a hand written zero and so on.
		int[] numTotals = {0,0,0,0,0,0,0,0,0,0};
		
		//Variable to store the number of samples we want to read in from the possible training samples
		int numSamples = 100;// Max 42,000
		while(sum(numTotals) < numSamples) {
			//Get a possible sample from the createEqualSample method
			Sample n = createEqualSample(in, numTotals, numSamples / numTotals.length);
			
			if(n != null) {
				NN.addSample(n);
			}
			
		}
	}
	
	/**
	 * This method is used to read all of the samples from the training set into the sample array list
	 * which will be tested after the NN has been trained
	 */
	private static void readALL() {
		Scanner in = null;
		try {
			in = new Scanner(new File("train.csv"));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		String[] line = in.nextLine().split(",");
		
		while(in.hasNextLine()) {
			line = in.nextLine().split(",");
			
			samps.add(createSample(line));
		}
	}
	
	/**
	 * This method checks to see how many of the next sample is already in the NN training set
	 * If the number is over 10% of the total number of samples, we are not going to create a new one
	 * in order to try and maintain a balanced training set
	 * 
	 * @param in
	 * @param totals
	 * @param numPerSection
	 * @return
	 */
	private static Sample createEqualSample(Scanner in, int[] totals, int numPerSection) {
		String[] Line = in.nextLine().split(",");
		if(totals[Integer.parseInt(Line[0])] >= numPerSection) {
			return null;
		} else {
			totals[Integer.parseInt(Line[0])] += 1;
		}
		
		return createSample(Line);
	}
	
	/**
	 * Returns a new sample based on the line input given to us
	 * 
	 * @param Line
	 * @return
	 */
	private static Sample createSample(String[] Line) {
		double[] y = new double[10];
		//The label of this sample is at the 0th index of the String array
		y[Integer.parseInt(Line[0])] = 1;
		
		double[] x = new double[Line.length - 1];
		
		//We then store all of the input values into the x array
		for(int i = 1; i < Line.length; i++) {
			x[i - 1] = Integer.parseInt(Line[i]) / 10000.0;
		}
		
		return new Sample(x, y);
	}
	
	/**
	 * Determine the sum of all of the elements in an array
	 * 
	 * @param nums
	 * @return
	 */
	private static int sum(int[] nums) {
		int sum = 0;
		for(int i = 0; i < nums.length; i++) {
			sum += nums[i];
		}
		return sum;
	}

}
