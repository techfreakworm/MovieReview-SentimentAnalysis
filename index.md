<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.13.3/dist/tf.min.js"> </script>

<script>
  function predictSentiment(){
	document.getElementById("predictionResult").innerHTML = "We're getting there"
  }
</script>

# Sentiment Analysis on IMDB Movie reviews
### Algorithm: Multilayered Perceptron (MLP) - Keras Sequential

<input type="text" id="reviewString">
<button onclick="predictSentiment();">Predict Sentiment</button>

<p id="predictionResult"></p>