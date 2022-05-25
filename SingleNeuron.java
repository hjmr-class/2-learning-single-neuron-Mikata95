import static java.lang.Math.exp;

public class SingleNeuron {

    private static final int TEACH_NUM = 4;
    private static final int INP_NUM = 2;
    private static final double alpha = 0.1;

    static double[] w = new double[INP_NUM];
    static double[] dw = new double[INP_NUM];
    static double theta;
    static double d_theta;

    static double[][] teachX = {
            { 0, 0 },
            { 0, 1 },
            { 1, 0 },
            { 1, 1 }
    };

    static double[] teachY = { 0, 1, 0, 1 };

    static double randOne() {
        return Math.random();
    }

    static double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double forward(double[] x) {
        double u = 0;
        for (int i = 0; i < INP_NUM; i++) {
            u += w[i] * x[i];
        }
        u += theta;
        return sigmoid(u);
    }

    double funcError() {
        double e = 0;
        for (int t = 0; t < TEACH_NUM; t++) {
            double y = forward(teachX[t]);
            e += 0.5 * (y - teachY[t]) * (y - teachY[t]);
        }
        return e;
    }

    void clearDw() {
        for (int i = 0; i < INP_NUM; i++) {
            dw[i] = 0;
        }
        d_theta = 0;
    }

    void calcDw() {
        for (int t = 0; t < TEACH_NUM; t++) {
            double[] xt = teachX[t];

            double y = forward(xt);
            double y_hat = teachY[t];
            for (int i = 0; i < INP_NUM; i++) {
                dw[i] += (y - y_hat) * y * (1 - y) * xt[i];
            }
            d_theta += (y - y_hat) * y * (1 - y);
        }
    }

    void initW() {
        for (int i = 0; i < INP_NUM; i++) {
            w[i] = randOne() * 2 - 1.0;
        }
        theta = randOne() * 2 - 1.0;
    }

    void updateW() {
        for (int i = 0; i < INP_NUM; i++) {
            w[i] -= alpha * dw[i];
        }
        theta -= alpha * d_theta;
    }

    public static void main(String[] args) {
        SingleNeuron neuron = new SingleNeuron();
        int t = 0, loop = 0;

        neuron.initW();
        for (t = 0; t < TEACH_NUM; t++) {
            double y = neuron.forward(teachX[t]);
            System.out.format("%d: y = %f <--> y_hat = %f\n", t, y, teachY[t]);
        }
        for (loop = 0; loop < 100000; loop++) {
            if (loop % 1000 == 0) {
                System.out.format("%d, %f\n", loop, neuron.funcError());
            }
            neuron.clearDw();
            neuron.calcDw();
            neuron.updateW();
        }
        System.out.format("%d, %f\n", loop, neuron.funcError());

        for (t = 0; t < TEACH_NUM; t++) {
            double y = neuron.forward(teachX[t]);
            System.out.format("%d: y = %f <--> y_hat = %f\n", t, y, teachY[t]);
        }
    }
}
