package com.feeling.deeplearning;

import java.util.Random;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.RandomGeneratorFactory;

/**
 * 
 * this implementation is followed by
 * http://deeplearning.stanford.edu/wiki/index
 * .php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C
 * 
 * @author feeling
 * 
 */
public class NeuralNetworks {

	private double alpha;
	private double lambda;
	private int layerNum;
	private int[] activationLayerNum;

	private int trainSetNum;

	/**
	 * traingingSetInput
	 */
	private double[][] _Y;

	/**
	 * networks param
	 */
	private double[][][] _W;

	/**
	 * bias param
	 */
	private double[] _B;

	/**
	 * activation value
	 */
	private double[][][] _A;

	/**
	 * weighted sum
	 */
	private double[][][] _Z;

	/**
	 * 残差
	 */
	private double[][][] _D;

	private static double innerProduct(double[] a, double[] b) {
		double sum = 0.0;
		for (int i = 0; i < a.length; i++) {
			sum += (a[i] * b[i]);
		}
		return sum;
	}

	private static double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}

	private static double sigmoidDerivative(double z) {
		double sigmoid = sigmoid(z);
		return sigmoid * (1 - sigmoid);
	}

	public NeuralNetworks(int perActivationLayerNum[]) {
		// add input and output layer
		this.layerNum = perActivationLayerNum.length;
		this.activationLayerNum = perActivationLayerNum;
	}

	public void setTrainingSet(double[][] x, double[][] y) {
		this._Y = y;
		// 激活层的第一层实际上就是输入层
		activationLayerNum[0] = x[0].length;
		// 激活层的最后一层实际上就是输出层
		activationLayerNum[activationLayerNum.length - 1] = y[0].length;
		trainSetNum = x.length;
		_A = new double[trainSetNum][][];
		_Z = new double[trainSetNum][][];
		_D = new double[trainSetNum][][];
		// k表示第几个样本点，l表示第几层，i表示每一层里第几个单元。第0层即输入的样本点
		for (int k = 0; k < trainSetNum; k++) {
			_A[k] = new double[layerNum][];
			_Z[k] = new double[layerNum][];
			_D[k] = new double[layerNum][];
			for (int l = 0; l < layerNum; l++) {
				_A[k][l] = new double[activationLayerNum[l]];
				_Z[k][l] = new double[activationLayerNum[l]];
				_D[k][l] = new double[activationLayerNum[l]];
				for (int i = 0; i < activationLayerNum[l]; i++) {
					if (0 == l) {
						_A[k][l][i] = x[k][i];
					} else {
						_A[k][l][i] = 0.0;
					}
					_Z[k][l][i] = 0.0;
					_D[k][l][i] = 0.0;
				}
			}
		}
	}

	public void train() {
		normallyRandomInitialParams(0.0, 0.01);
		while (!isConvergency()) {
			forwardPass();
			backwardPass();
		}
	}

	public void normallyRandomInitialParams(double miu, double sigma) {
		RandomGenerator rg = RandomGeneratorFactory.createRandomGenerator(new Random(System.currentTimeMillis()));
		_W = new double[layerNum - 1][][];
		for (int l = 0; l < layerNum - 1; l++) {
			_W[l] = new double[activationLayerNum[l + 1]][];
			for (int i = 0; i < activationLayerNum[l + 1]; i++) {
				_W[l][i] = new double[activationLayerNum[l]];
				for (int j = 0; j < activationLayerNum[l]; j++) {
					// 将原始为miu为0.5，3sigma为0.5的正态分布，转移到miu，sigma为此处设置的正太分布的随机值
					_W[l][i][j] = (rg.nextGaussian() - 0.5 + miu) * 6.0 * sigma;
				}
			}
		}

		_B = new double[layerNum - 1];

		for (int l = 0; l < layerNum - 1; l++) {
			_B[l] = (rg.nextGaussian() - 0.5 + miu) * 6.0 * sigma;
		}
	}

	private void forwardPass() {
		for (int k = 0; k < trainSetNum; k++) {
			for (int l = 1; l < layerNum; l++) {
				for (int i = 0; i < activationLayerNum[l]; i++) {
					_Z[k][l][i] = innerProduct(_A[k][l - 1], _W[l - 1][i]) + _B[l - 1];
					_A[k][l][i] = sigmoid(_Z[k][l][i]);
				}
			}
		}
	}

	private void backwardPass() {
		updateErrorTerm();
		for (int l = 0; l < layerNum - 1; l++) {

			for (int i = 0; i < activationLayerNum[l + 1]; i++) {
				for (int j = 0; j < activationLayerNum[l]; j++) {
					double sumA = 0.0;
					double sumB = 0.0;
					for (int k = 0; k < trainSetNum; k++) {
						sumA += _A[k][l][j] * _D[k][l + 1][i];
						sumB += _D[k][l + 1][i];
					}
					_W[l][i][j] -= alpha * ((sumA) / trainSetNum + (lambda * _W[l][i][j]));
					_B[l] -= alpha * (sumB / trainSetNum);
				}
			}
		}
	}

	private void updateErrorTerm() {
		for (int k = 0; k < trainSetNum; k++) {
			for (int i = 0; i < activationLayerNum[layerNum - 1]; i++) {
				_D[k][layerNum - 1][i] = (_A[k][layerNum - 1][i] - _Y[k][i]) * _A[k][layerNum - 1][i]
						* (1.0 - _A[k][layerNum - 1][i]);
			}
			for (int l = layerNum - 2; l > 0; l--) {
				for (int i = 0; i < activationLayerNum[l]; i++) {
					for (int j = 0; j < activationLayerNum[l + 1]; j++) {
						_D[k][l][i] += (_W[l][j][i] * _D[k][l + 1][j]);
					}
					_D[k][l][i] *= (_A[k][l][i] * (1.0 - _A[k][l][i]));
				}
			}
		}
	}

	private boolean isConvergency() {
		return true;
	}

	public double[] predict(double[] x) {
		return null;
	}

}
