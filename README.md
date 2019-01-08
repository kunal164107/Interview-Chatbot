# Interview-Chatbot

An Interview Chatbot powered by Deep Learning and trained on dataset consisting of Interview Question(Computer Science Domain). 
Built on TensorFlow v1.11.0 and tensorLayer v1.11.1 and Python v3.6 and Trained the model on NVIDIA DGX-1 V100.

You can download the dataset from [here](https://drive.google.com/open?id=1WPY3HB3BlXD-Pflk-CbN60_rTQ8eMNlt).
Here is a sample chat transcript. Bot replies with "Out of Context Question " whenever user ask question from a different domain.<br>
![](Images/Capture.JPG)<br><br>
Preety good response right!!

# Usage
Step 1: Install required libraries<br>
Step 2: Clone the project<br>
Step 3: Train the model **python main.py --batch-size 32 --num-epochs 1000 -lr 0.001**<br>
Step 4: Run the model **python app.py**<br>

You Can install the trained model from [here](google.com)<br>
# MethodoLogy
1. **Prepare the Dataset:**<br>
First we need to prepare the dataset. We had prepared the dataset of question for the subjects like Data Structures, Algorithms, Operating System. The dataset contain the question and answers of these subjects. The better the dataset, the more accurate and efficient conversational results can be obtained.<br><br>
2. **Pre-Processing:**
  * Lowercase all the charcters and remove unwanted charcter like - or # or $ etc. 
  * Filter the dataset with max question length and max answer length Here we are use 20 for both qmax and amax.
  * Tokenization and Vectorization
  * Add zero padding 
  * Split into train,validation,test data<br><br>
3. **Creation of LSTM,Encoder and Decoder Model:**<br>
LSTM are a special kind of RNN which are capable of learning long-term dependencies. Encoder-Decoder model contains two parts- encoder which takes the vector representation of input sequence and maps it to an encoded representation of the input. This is then used by decoder to generate output.<br><br>

4. **Train and Save Model:**<br>
We trained the model with 1000 epochs and batch size of 32, Learning rate-0.001, word embedding size was set to 1024, we took categorical cross entropy as our loss function and optimiser used was AdamOptimizer. We got the best results with these parameters. We trained and tested our model on NVIDIA DGX-1 V100. Training accuracy obtained was approximately 99% and validation accuracy of about 80%.<br><br>

5. **Testing:**<br>
Finally the user can input questions or speak question by clicking Speak now button and bot will reply the answer of the Question. The results obtained are satisfactory according to review analysis.<br>

6. **Submitted By:** [Kunal Kumar](https://github.com/kunal164107), [Syed Muhamed Nihall](https://github.com/neoNihall/), [Vvsmanideep](https://github.com/manideepvvs356), Pachipulusu Vamshi Krishna.<br><br> The Project was done under the guidance of [Mr. Tejalal](https://github.com/tejalal) and Dr. Vipul Kumar Mishra<br><br>

Thanks to [Suriyadeepan Ramamoorthy](https://github.com/suriyadeepan) for his [practical_seq2seq](https://github.com/suriyadeepan/practical_seq2seq) repo, which this repo is based on<br><br>
