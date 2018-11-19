<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.3/Chart.bundle.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.13.3/dist/tf.min.js"> </script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
<script>    
    var predictionNumber = 0
    var config
    
    async function loadModels(){
        $('#info').text("Loading Model, please wait...")
        this.CNNModel = await tf.loadModel('kerasMLPTfjs/model.json')
        this.LSTMModel = await tf.loadModel('kerasLSTM/model.json')
        this.model = {
            "CNN": this.CNNModel,
            "LSTM": this.LSTMModel
        }
    }
    
    async function loadWordIndex(){
        $('#info').text("Loading word Index, please wait...")
        const wordIndexJson = await fetch('word_index_data.json')
        this.wordIndex = await wordIndexJson.json();
    }
    
    async function predictSentiment(){
        const inputText = $('#reviewText').val().trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
        
        const inputBuffer = tf.buffer([1, 100], 'int32');
        
        for (let i = 0; i < inputText.length; ++i) {
            const word = inputText[i];
            inputBuffer.set(this.wordIndex[word] + 3, 0, 100-inputText.length+i);
        }
        
        const input = inputBuffer.toTensor();

        predictionNumber = predictionNumber + 1
        config.data.labels.push(predictionNumber)

        $("#info").text("Running inference...")
        for(var key in this.model)
        {
            const predictOut = this.model[key].predict(input)
            const score = predictOut.dataSync()[0]
            predictOut.dispose()
            updatePredictionResults(key, score)
            updateComparisonChart(key, score)
        }
        $("#info").text("Inference Complete!")
    }
    
    
    async function updatePredictionResults(element, score)
    
    {
        let elementID = '#' + element + 'result'
        if (score>0.5){
            $(elementID).text("Positive review, Confidence: "+ score*100 + " %" )
        }
        else if (score<0.5){
            $(elementID).text("Negative review, Confidence: " + (1-score)*100 + " %")
        }
        else{
            $(elementID).text("Something went wrong")
        }
    }
    
    async function updateComparisonChart(element, score){
        if (element == 'CNN'){
            config.data.datasets[0].data.push(score)
        }
        else{
            config.data.datasets[1].data.push(score)
        }    
        window.myLine.update()
    }
    
    async function init(){
        await loadModels()
        await loadWordIndex()
        initChart()
        $('#info').text("Model and word index loaded, type in your review and hit predict to predict sentiment. Happy Predicting! :)")
        $('#predictDiv').css("display", "block")
    }
    
    async function initChart(){
        config = {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CNN',
                    backgroundColor: '#ff0000',
                    borderColor: "#ff0000",
                    data: [],
                    fill: false,
                }, {
                    label: 'LSTM',
                    fill: false,
                    backgroundColor: '#00ff00',
                    borderColor: '#00ff00',
                    data: [],
                }]
            },
            options: {
                responsive: true,
                title: {
                    display: true,
                    text: 'Realtime comparison'
                },
                tooltips: {
                    mode: 'index',
                    intersect: false,
                },
                hover: {
                    mode: 'nearest',
                    intersect: true
                },
                scales: {
                    xAxes: [{
                        display: true,
                        scaleLabel: {
                            display: true,
                            labelString: 'Predictions'
                        }
                    }],
                    yAxes: [{
                        display: true,
                        scaleLabel: {
                            display: true,
                            labelString: 'Score'
                        }
                    }]
                }
            }
        };

        var ctx = document.getElementById('chartCanvas').getContext('2d');
        window.myLine = new Chart(ctx, config);
        
    }
    
    $( document ).ready(init);
</script>
    
<div id="predictDiv" style="display:none;">
    <div>
        <textarea rows="5" cols="70" id="reviewText" placeholder="Type your review here!"></textarea>
    </div>
    <button onclick="predictSentiment();" id="predictButton">Predict</button>
</div>
<div>
    <h4>Info:</h4>
    <p id="info"></p>
</div>
<div>
    <h4>CNN:</h4>
    <p id="CNNresult"></p>
</div>
<div>
    <h4>LSTM:</h4>
    <p id="LSTMresult"></p>
</div>

<div id="chartDiv">
    <canvas id="chartCanvas"></canvas>
</div>