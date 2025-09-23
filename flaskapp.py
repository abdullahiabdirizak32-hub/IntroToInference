from flask import Flask, request, jsonify, send_file
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import io

app = Flask(__name__)

class ClimateAnomalyAnalyzer:
    def __init__(self, mu, sigma, X):
        self.mu = mu
        self.sigma = sigma
        self.X = X
        self.z = self.compute_zscore()
        self.p_less, self.p_greater = self.compute_probabilities()

    def compute_zscore(self):
        return (self.X - self.mu) / self.sigma

    def compute_probabilities(self):
        p_less = norm.cdf(self.z)
        p_greater = 1 - p_less
        return p_less, p_greater

    def plot_distribution(self):
        x_vals = np.linspace(self.mu - 4*self.sigma, self.mu + 4*self.sigma, 1000)
        y_vals = norm.pdf(x_vals, self.mu, self.sigma)

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label='Normal Distribution')
        ax.fill_between(x_vals, y_vals, where=(x_vals > self.X), color='red', alpha=0.5, label=f'P(X > {self.X})')
        ax.axvline(self.X, color='black', linestyle='--', label=f'X = {self.X}')
        ax.set_title('Climate Anomaly Distribution')
        ax.set_xlabel('Anomaly Value')
        ax.set_ylabel('Probability Density')
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

@app.route('/analyze')
def analyze():
    try:
        mu = float(request.args.get('mu'))
        sigma = float(request.args.get('sigma'))
        X = float(request.args.get('X'))

        analyzer = ClimateAnomalyAnalyzer(mu, sigma, X)
        response = {
            "Z-score": round(analyzer.z, 4),
            "P(X <= X)": round(analyzer.p_less, 4),
            "P(X > X)": round(analyzer.p_greater, 4)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/plot')
def plot():
    try:
        mu = float(request.args.get('mu'))
        sigma = float(request.args.get('sigma'))
        X = float(request.args.get('X'))

        analyzer = ClimateAnomalyAnalyzer(mu, sigma, X)
        plot_buf = analyzer.plot_distribution()
        return send_file(plot_buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/')
def home():
    return '''
    <h2>Welcome to Climate Anomaly Analyzer 🌍</h2>
    <p>Try accessing <a href="/analyze?mu=0.5&sigma=0.2&X=0.9">/analyze</a> or <a href="/plot?mu=0.5&sigma=0.2&X=0.9">/plot</a></p>
    '''

if __name__ == '__main__':
    app.run(debug=True)
