<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.13.3/dist/tf.min.js"> </script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
<script>    
    async function loadModelAndWordIndex(){
        document.getElementById("result").innerHTML = "Loading Model, please wait..."
        this.model = await tf.loadModel('kerasMLPTfjs/model.json')
        const wordIndexJson = await fetch('word_index_data.json')
        const wordIndexData = await wordIndexJson.json();
        this.wordIndex = wordIndexData['word_index']
        document.getElementById("result").innerHTML = "Model loaded, type in your review and hit predict. Happy Predicting! :)"
    }

    async function predictSentiment(text){
        const inputText = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
        
        const inputBuffer = tf.buffer([1, 100], 'int32');
        
        for (let i = 0; i < inputText.length; ++i) {
            const word = inputText[i];
            inputBuffer.set(this.wordIndex[word] + 3, 0, i);
        }
        
        const input = inputBuffer.toTensor();
        
        document.getElementById("result").innerHTML = "Running inference..."
        const predictOut = this.model.predict(input);
        const score = predictOut.dataSync()[0];
        predictOut.dispose();
        if (score>0.5){
            document.getElementById("result").innerHTML = "Positive review, score: " + score 
        }
        else if (score<0.5){
            document.getElementById("result").innerHTML = "Negative review, score: " + score 
        }
        else{
            document.getElementById("result").innerHTML = "Something went wrong";
        }
    }


    $( document ).ready(function() {
        loadModelAndWordIndex();
        
        $('#reviewText').keyup(function(){
            var keyed = $(this).val();
            predictSentiment(keyed)
        });
    });
</script>

# Sentiment Analysis on IMDB Movie reviews
### Algorithm: Multilayered Perceptron (MLP) - Keras Sequential
<div>
    <textarea rows="5" cols="70" id="reviewText">Type your review here!</textarea>
</div>
<div>
    <p id="result"></p>
</div>