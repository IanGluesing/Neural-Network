package NN_V2;

public class Sample {

	public double[] X = new double[3];
	public double[] y = new double[2];
	
	public Sample(double x1, double x2, double x3, int y1, int y2) {
		X[0] = x1;
		X[1] = x2;
		X[2] = x3;
		y[0] = y1;
		y[1] = y2;
	}
	
	public Sample(double x1, double x2, double x3) {
		X[0] = x1;
		X[1] = x2;
		X[2] = x3;
	}
	
	public String toString() {
		return "[ x1: " + X[0] + " x2: " + X[1] + " x3: " + X[2] + " ]";
	}
}
