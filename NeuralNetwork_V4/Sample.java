package NN_V4;

import java.util.Arrays;
import java.util.ArrayList;

/**
 * Class used to represent a sample
 * 
 * @author Ian Gluesing
 *
 */

public class Sample {

	private Layer xLayer = new Layer();
	private double[] y = null;
	private ArrayList<Layer> values = new ArrayList<Layer>();
	
	/**
	 * Constructor for a sample where we already know what the label for this sample is
	 * 
	 * @param x
	 * @param y
	 */
	public Sample(double[] x, double[] y) {
		//Assign the input layer of this sample to have the values of x
		xLayer.setValues(x, false);
		values.add(xLayer);
		this.y = y;
	}
	
	/**
	 * Constructor for a sample where we do not know what the label for this sample is
	 * 
	 * @param x
	 */
	public Sample(double[] x) {
		xLayer.setValues(x, false);
	}
	
	/**
	 * Used to determine if the error threshold for this sample has been met
	 * Checks to see if the sum of the squared error over each output is less than .05
	 * 
	 * @return
	 */
	public boolean errorThreshold() {
		//Return true if the average squared error over each output value is less than .05
		return sumSqError() / y.length < .05 ? true : false;
	}
	
	/**
	 * Returns the error of each value for this sample
	 * 
	 * @return
	 */
	public double[] error() {
		double[] error = new double[y.length];
		for(int i = 0; i < y.length; i++) {
			error[i] = values.get(values.size() - 1).getValues()[i]- y[i];
		}
		return error;
	}
	
	/**
	 * Returns the sum of the error squared over each possible output for this sample
	 * 
	 * @return
	 */
	public double sumSqError() {
		double error = 0;
		double[] errors = error();
		for(int i = 0; i < y.length; i++) {
			error += Math.pow(errors[i], 2);
		}
		return error;
	}
	
	/**
	 * Returns the index of the highest value in the output layer of this sample
	 * Because we are classifying numbers, the index will represent the predicted number
	 * 
	 * @return
	 */
	public int getPrediction() {
		//Get the predicted values of the last layer for this sample
		double[] vals = values.get(values.size() - 1).getValues();
		double max = 0;
		int index = 0;
		for(int i = 0; i < vals.length; i++) {
			if(vals[i] > max) {
				index = i;
				max = vals[i];
			}
		}
		return index;
	}
	
	/*
	 * If the index of the predicted value is equal to 1, that means there has
	 * been a correct prediction
	 * 
	 */
	public boolean checkTruePrediction() {
		int prediction = getPrediction();
		return y[prediction] == 1;
	}
	
	/**
	 * Returns a certain layers prediction for the given index
	 * 
	 * @param index
	 * @return
	 */
	public Layer getLayer(int index) {
		return values.get(index);
	}
	
	/**
	 * Returns the input layer of the sample, the x values with bias term
	 * 
	 * @return
	 */
	public Layer getInputLayer() {
		return xLayer;
	}
	
	/**
	 * Adds a layer to values at each layer for this sample
	 * 
	 * @param l
	 */
	public void addLayer(Layer l) {
		values.add(l);
	}
	
	/**
	 * Clears the predictions at each layer for this sample
	 */
	public void removeAllLayers() {
		values = new ArrayList<Layer>();
	}
	
	public String toString() {
		if(y != null) {
			return Arrays.toString(y);
		} else {
			return "test sample";
		}
	}
	
}
