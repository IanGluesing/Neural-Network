package NN_V4;

import java.util.Arrays;

/**
 * Class used to represent a layer at each level within the neural network
 * 
 * @author Ian Gluesing
 *
 */

public class Layer {
	
	//Structure to store the values of the current layer
	private double[] values;
	
	public Layer() {
	}
	
	/**
	 * Main method to change the values wihtin a layer.
	 * 
	 * 
	 * @param val
	 * 		Values that will be stored at this layer
	 * @param output
	 * 		To determine if this layer needs a bias term whether or not
	 * 		this layer is the output layer
	 */
	public void setValues(double[] val, boolean output) {
		if(!output){
			//If not the output layer, add a bias term of 1, to the end of this layer
			values = Arrays.copyOf(val, val.length + 1);
			values[values.length - 1] = 1;
		} else {
			//If this layer is the output layer, remove the bias term and assign it to values
			values = Arrays.copyOf(val, val.length - 1);
		}
	}
	
	/**
	 * String representation of the current layer
	 */
	public String toString() {
		String out = "[ " + values[0];
		for(int i = 1; i < values.length; i++) {
			out += "\n  " + values[i];
		}
		out += " ]";
		
		return out;
	}
	
	public double[] getValues() {
		return values;
	}
	
}
