package NN_V4;

/**
 * Simple class that uses the logistic function
 * 
 * @author Ian Gluesing
 *
 */

public class Functions {

	/**
	 * Returns the value of the logistic function at the given x value
	 * 
	 * @param value
	 * @return
	 */
	public static double activationFunction(double value) {
		return (double)1 / (1 + Math.pow(Math.E, -value));
	}
}
