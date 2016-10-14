package com.feeling.deeplearning;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.RandomGeneratorFactory;

/**
 * 
 * this implementation is followed by
 * http://deeplearning.stanford.edu/wiki/index
 * .php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C,this is a toy code,without
 * inputcheck and multi-thread supported
 * 
 * @author feeling
 * 
 */
public class NeuralNetworks {

	private static final double STOP_CONVERGENCY_CHANGING = 0.00000001;

	private double alpha;
	private double lambda;
	private int layerNum;
	private int[] activationLayerNum;

	private int trainSetNum;
	private double overallCost = Double.MAX_VALUE;

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

	/**
	 * 临时激活值变量，用来应用模型做计算
	 */
	private double[][] _temp_A;

	/**
	 * 临时线性因变量，用来应用模型做计算
	 */
	private double[][] _temp_Z;

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

	/**
	 * 由于f(z) 是sigmoid函数，并且我们已经在前向传导运算中得到了 a^{(l)}_i。那么，使用我们早先推导出的
	 * f'(z)表达式，就可以计算得到 f'(z^{(l)}_i) = a^{(l)}_i (1- a^{(l)}_i)。
	 * 
	 * @param fz
	 * @return
	 */
	private static double fastSigmoidDerivative(double a) {
		return a * (1 - a);
	}

	/**
	 * perActivationLayerNum,
	 * 
	 * @param perActivationLayerNum
	 *            网络里每一层的单元数，一共n层的话，第一层是输入层，中间n-2层是隐藏层，最后一层是输出层
	 * @param alpha
	 * @param lambda
	 */
	public NeuralNetworks(int[] perActivationLayerNum, double alpha, double lambda) {
		// add input and output layer
		this.layerNum = perActivationLayerNum.length;
		this.activationLayerNum = perActivationLayerNum;
		this.alpha = alpha;
		this.lambda = lambda;
	}

	public void setTrainingSet(double[][] x, double[][] y) {
		this._Y = y;
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

		_temp_A = new double[layerNum][];
		_temp_Z = new double[layerNum][];
		for (int l = 0; l < layerNum; l++) {
			_temp_A[l] = new double[activationLayerNum[l]];
			_temp_Z[l] = new double[activationLayerNum[l]];
			for (int i = 0; i < activationLayerNum[l]; i++) {
				_temp_A[l][i] = 0.0;
				_temp_Z[l][i] = 0.0;
			}
		}

	}

	public void train() {
		normallyRandomInitialParams(0.0, 0.01);
		forwardPass();
		int iterCount = 0;
		while (!isConvergency()) {
			iterCount++;
			if (iterCount % 100 == 0) {
				System.out.println("iteration count => " + iterCount);
			}
			backwardPass();
			forwardPass();
		}
		System.out.println("total iteration count => " + iterCount);
	}

	public void normallyRandomInitialParams(double mu, double sigma) {
		RandomGenerator rg = RandomGeneratorFactory.createRandomGenerator(new Random(System.currentTimeMillis()));
		_W = new double[layerNum - 1][][];
		for (int l = 0; l < layerNum - 1; l++) {
			_W[l] = new double[activationLayerNum[l + 1]][];
			for (int i = 0; i < activationLayerNum[l + 1]; i++) {
				_W[l][i] = new double[activationLayerNum[l]];
				for (int j = 0; j < activationLayerNum[l]; j++) {
					// 将原始为mu为0.5，3sigma为0.5的正态分布，转移到mu，sigma为此处设置的正态分布的随机值
					_W[l][i][j] = (rg.nextGaussian() - 0.5 + mu) * 6.0 * sigma;
				}
			}
		}

		_B = new double[layerNum - 1];

		for (int l = 0; l < layerNum - 1; l++) {
			_B[l] = (rg.nextGaussian() - 0.5 + mu) * 6.0 * sigma;
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

	private void predictForwardPass() {
		for (int l = 1; l < layerNum; l++) {
			for (int i = 0; i < activationLayerNum[l]; i++) {
				_temp_Z[l][i] = innerProduct(_temp_A[l - 1], _W[l - 1][i]) + _B[l - 1];
				_temp_A[l][i] = sigmoid(_temp_Z[l][i]);
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
				_D[k][layerNum - 1][i] = (_A[k][layerNum - 1][i] - _Y[k][i])
						* fastSigmoidDerivative(_A[k][layerNum - 1][i]);
				;
			}
			for (int l = layerNum - 2; l > 0; l--) {
				for (int i = 0; i < activationLayerNum[l]; i++) {
					for (int j = 0; j < activationLayerNum[l + 1]; j++) {
						_D[k][l][i] += (_W[l][j][i] * _D[k][l + 1][j]);
					}
					_D[k][l][i] *= fastSigmoidDerivative(_A[k][l][i]);
				}
			}
		}
	}

	private void updateOverallCost() {
		double sumSquaresError = 0.0;
		for (int k = 0; k < trainSetNum; k++) {
			for (int j = 0; j < activationLayerNum[activationLayerNum.length - 1]; j++) {
				sumSquaresError += 0.5 * Math.pow(_A[k][layerNum - 1][j] - _Y[k][j], 2);
			}
		}
		double sumWeightDecay = 0.0;
		for (int l = 0; l < layerNum - 1; l++) {
			for (int i = 0; i < activationLayerNum[l + 1]; i++) {
				for (int j = 0; j < activationLayerNum[l]; j++) {

					sumWeightDecay += Math.pow(_W[l][i][j], 2);
				}
			}
		}
		this.overallCost = sumSquaresError / trainSetNum + this.lambda * sumWeightDecay / 2.0;
	}

	private boolean isConvergency() {
		double lastOverallCost = this.overallCost;
		updateOverallCost();
		if (lastOverallCost - this.overallCost < STOP_CONVERGENCY_CHANGING) {
			return true;
		} else {
			return false;
		}
	}

	public double[] predict(double[] x) {
		for (int i = 0; i < x.length; i++) {
			_temp_A[0][i] = x[i];
		}
		predictForwardPass();
		return Arrays.copyOf(_temp_A[layerNum - 1], activationLayerNum[activationLayerNum.length - 1]);
	}

	public static void main(String[] args) {
		int[] perActivationLayerNum = { 3, 15, 4, 2 };
		double alpha = 0.1;
		double lambda = 0.0;
		NeuralNetworks nn = new NeuralNetworks(perActivationLayerNum, alpha, lambda);
		double[][] train_x = { { 10.0, 5.2, 5.8 }, { 11.0, 4.7, 6.8 }, { 11.0, 4.7, 6.8 }, { 2.0, 15.6, 9.8 },
				{ 2.2, 14.6, 10.8 }, { 2.4, 15.0, 10.0 } };
		double[][] train_y = { { 1.0, 1.05 }, { 1.0, 0.98 }, { 1.0, 0.90 }, { 0.0, -0.01 }, { 0.01, 0.2 },
				{ -0.3, 0.02 } };
		nn.setTrainingSet(train_x, train_y);
		nn.train();
		double[] test_x = { 11.0, 4.98, 6.8 };
		double[] result_y = nn.predict(test_x);
		for (double per_y : result_y) {
			System.out.println("result =>" + per_y);
		}

	}

}