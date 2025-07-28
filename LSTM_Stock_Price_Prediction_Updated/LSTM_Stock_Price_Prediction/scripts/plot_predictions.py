import matplotlib.pyplot as plt

def plot_results(actual, predicted):
    plt.figure(figsize=(12,6))
    plt.plot(actual, color='blue', label='Actual')
    plt.plot(predicted, color='red', label='Predicted')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()