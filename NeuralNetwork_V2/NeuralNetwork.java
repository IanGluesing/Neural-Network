package NN_V2;

import java.util.ArrayList;
import java.util.*;
import java.util.Arrays;

public class NeuralNetwork {
	
	public static void main(String[] args) {
		Random r = new Random();
		ArrayList<Sample> samples = new ArrayList<Sample>();
		ArrayList<Sample> test = new ArrayList<Sample>();
		
		for(double i = 0; i < 1000000; i += 1) {
			double num1 = r.nextDouble();
			double num2 = r.nextDouble();
			double num3 = r.nextDouble();
			if(num1 + num2 + num3 > 1.5) {
				samples.add(new Sample(num1, num2, num3, 1, 0));
			} else {
				samples.add(new Sample(num1, num2, num3, 0, 1));
			}
		}
		
		for(int i = 0; i < 10; i++) {
			test.add(new Sample(r.nextDouble(), r.nextDouble(), r.nextDouble()));
		}
		
		double[] hiddenOne;
		double[] hiddenTwo;
		double[] output;


		double[][] W2 = {{2,2,2},
						 {2,2,2},
						 {2,2,2},
						 {2,2,2}};
	
		double[][] W1 = {{.1,.1,.1,.1},
						 {.1,.1,.1,.1},
						 {.1,.1,.1,.1}};

		double[][] W3 = {{1,1},
						 {1,1},
						 {1,1}};
		
		
		
		
		for(int i = 0; i < 10; i++) {	
			for(Sample s: samples) {
				hiddenOne = predict(s.X, W1);
				hiddenTwo = predict(hiddenOne, W2);
				output = predict(hiddenTwo, W3);
				
				
				double[] initialLoss = {output[0] - s.y[0], output[1] - s.y[1]};
				double[] hiddenTwoLoss = loss(hiddenTwo, W3, initialLoss);
				double[] hiddenOneLoss = loss(hiddenOne, W2, hiddenTwoLoss);
				
				double[][] gradientW3 = gradient(hiddenTwo, initialLoss);
				double[][] gradientW2 = gradient(hiddenOne, hiddenTwoLoss);
				double[][] gradientW1 = gradient(s.X, hiddenOneLoss);
				
				W3 = updateW(W3, gradientW3, 1);
				W2 = updateW(W2, gradientW2, 1);		
				W1 = updateW(W1, gradientW1, 1);

			}
			
		}
		
		for(Sample s: test) {
			hiddenOne = predict(s.X, W1);
			hiddenTwo = predict(hiddenOne, W2);
			output = predict(hiddenTwo, W3);
			
			System.out.println(Arrays.toString(output) + " " + s);
		}
		
	}
	
	public static double[] squareLoss(double[] loss) {
		for(int i = 0; i < loss.length; i++) {
			loss[i] = Math.pow(loss[i], 2);
		}
		return loss;
	}
	
	public static double[][] ave(double[][] sumG, double size){
		for(int i = 0; i < sumG.length; i++) {
			for(int j = 0; j < sumG[i].length; j++) {
				sumG[i][j] /= size;
			}
		}
		return sumG;
	}
	
	public static double[][] add(double[][] g, double[][] sumG){
		for(int i = 0; i < sumG.length; i++) {
			for(int j = 0; j < sumG[i].length; j++) {
				sumG[i][j] += g[i][j];
			}
		}
		return sumG;
	}
	

	public static void print(double[] output) {
		System.out.print("[ " + output[0] + ",\n");
		for (int i = 1; i < output.length - 1; i++) {
			System.out.println("  " + output[i] + ",");
		}
		System.out.println("  " + output[output.length - 1] + " ]");
	}

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

	public static double activationFunction(double sum) {
		return (double) 1 / (1 + Math.pow(Math.E, -sum));
	}

	public static double[][] updateW(double[][] W, double[][] gradient, double learningRate) {
		for (int i = 0; i < W.length; i++) {
			for (int j = 0; j < W[i].length; j++) {
				W[i][j] -= learningRate * gradient[i][j];
			}
		}

		return W;
	}

	public static double[][] gradient(double[] x, double[] previousLoss) {
		double[][] grad = new double[x.length][previousLoss.length];

		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < previousLoss.length; j++) {
				grad[i][j] = x[i] * previousLoss[j];
			}
		}

		return grad;
	}

	public static double[] loss(double[] x, double[][] W, double[] lossPrevious) {
		double[] hadamardLeft = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			hadamardLeft[i] = x[i] * (1 - x[i]);
		}

		double[] hadamardRight = new double[W.length];
		for (int i = 0; i < W.length; i++) {
			double sum = 0;
			for (int j = 0; j < W[i].length; j++) {
				sum += W[i][j] * lossPrevious[j];
			}
			hadamardRight[i] = sum;
		}

		double[] newLoss = new double[x.length];

		for (int i = 0; i < newLoss.length; i++) {
			newLoss[i] = hadamardLeft[i] * hadamardRight[i];
		}

		return newLoss;
	}

	public static double[] predict(double[] xValues, double[][] weights) {
		double[] ans = new double[weights[0].length];

		for (int i = 0; i < weights[0].length; i++) {
			double sum = 0;
			for (int j = 0; j < weights.length; j++) {
				sum += weights[j][i] * xValues[j];
			}
			ans[i] = activationFunction(sum);
		}

		return ans;
	}
}
