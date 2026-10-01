# 5-minute video script

**0:00 Intro (30s)** Name, course, and the goal: a neural network that predicts sentiment live and explains its decision.

**0:30 Problem and dataset (45s)** Show the PPT slides 2-3. IMDB, 50,000 reviews, 10,000-word vocabulary.

**1:15 Architecture (60s)** Slide 4 or the Model insights tab. Embedding turns words into vectors, Bi-LSTM reads the sentence both ways, dense layers decide, sigmoid gives a probability. Mention dropout and early stopping.

**2:15 Live demo (120s)** Open the app.
- Click "Glowing review": point at the gauge and teal words.
- Click "Scathing review": coral words.
- Type live: "The film was good" then add "not" and show the flip.
- Click "Plot twist": show that the network handles "expected terrible but surprisingly good".

**4:15 Insights tab (30s)** Accuracy, training curves (no big gap between train and validation), confusion matrix.

**4:45 Deployment and close (15s)** GitHub and Streamlit Cloud links, one-line conclusion.
