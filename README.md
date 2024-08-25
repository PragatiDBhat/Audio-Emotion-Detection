# Decoding Emotions Through Speech Using LSTM

## Introduction

Humans communicate not only by exchanging linguistic information with others but also through paralinguistic and non-linguistic information. Para-linguistic information includes intention, attitude, and style. Non-linguistic information includes sentiment and emotion. Sentiment can be regarded as a sub-category of emotion. It is close to valence in dimensional emotion.  ​

Sentiment analysis system helps companies improve their products and services based on genuine and specific customer feedback.​
Sentiment analysis finds applications in:​
1. Human-Computer Interaction (HCI)​
2. Customer Service and Market Research​
3. Security and Law Enforcement: Deception Detection​
4. Virtual Assistants​


## Dataset Description

### *Toronto emotional speech set (TESS)​*

A collection of 200 target words was articulated within the carrier phrase "Say the word _" by two actresses, aged 26 and 64. Recordings were captured for each actress expressing seven distinct emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutrality.​

The dataset comprises a total of 2800 audio files, with each actress and their emotional expressions organized into separate folders. Within each folder, the 200 target words are presented in WAV format, providing a comprehensive and structured dataset for analysis.​


## Data Preprocessing
### Data Collection​
- Walking through the directory structure of the dataset and collecting file paths and labels. ​
- Extracting emotion labels from file names and creating lists for paths and labels. ​
- Constructing a DataFrame (df) to organize this information.​

### Label Encoding​
- Using LabelEncoder from sklearn to convert categorical emotion labels into numerical representations.​
- Creating a new column 'label_encoded' in the DataFrame for encoded labels.​

### Feature Extraction (MFCC):​
- Defining a function extract_mfcc that utilizes the librosa library to extract Mel-frequency cepstral coefficients (MFCCs) from each audio file.
- Applying this function to each file path in the 'speech' column of the DataFrame to generate features.​
- Converting the extracted features (X_mfcc) into a numpy array and expanding its dimensions to accommodate the model's input shape.​
​
## Methods Available

### ​LSTM (Long Short-Term Memory):​
LSTMs are a type of recurrent neural network (RNN) designed to capture long-term dependencies in sequential data.​ LSTMs can be employed to analyze sequential patterns in speech signals. For instance, they can model the temporal dependencies in a sequence of audio features extracted from speech signals to recognize patterns associated with different emotions.​

### CNN (Convolutional Neural Network):​
CNNs can be used to extract hierarchical and spatial features from spectrograms or other time-frequency representations of speech signals. This helps in capturing relevant patterns and relationships between different frequency components associated with emotional content in speech.​ CNNs can leverage pre-trained models on large datasets (e.g., ImageNet) and fine-tune them for speech emotion recognition. This transfer learning approach can be especially useful when dealing with limited labeled data, helping the model to generalize better to various emotional expressions.​

​### Random Forest:​
Usage in Emotion Recognition: Random Forest can be applied to classify speech features extracted from audio signals. It is particularly useful when dealing with a combination of different types of features (acoustic, prosodic, linguistic) as it can handle both categorical and numerical data. Random Forest provides a measure of feature importance. In the context of emotion recognition, this feature importance analysis can help identify which acoustic or linguistic features contribute most significantly to the model's decision, providing valuable insights into the factors influencing emotional classification.​

### SVM (Support Vector Machine):​
Usage in Emotion Recognition: SVM can be employed for binary or multiclass classification of emotional states based on extracted speech features. It works well when the decision boundary between different emotion classes is not linear and requires a more complex separation.​ SVMs can utilize different kernel functions (e.g., radial basis function kernel) to transform the input space, making it easier to separate complex patterns. The choice of the kernel function in SVM allows for flexibility in capturing non-linear relationships within the high-dimensional feature space of speech signals. ​

## Results 
We have tried different machine learning models and have got the resukts as shown in below table:

---

| Model                            | Training Accuracy (%) | Testing Accuracy (%) |
|----------------------------------|-----------------------|----------------------|
| Long Short Term Memory (LSTM)    | 98                    | 97                   |
| Random Forest                    | 95                    | 84                   |
| Random Forest and CNN            | 98                    | 93                   |
| Support Vector Machine (SVM)     | 86                    | 87                   |
| SVM and CNN                      | 97                    | 94                   |
| Convolutional Neural Network (CNN)| 99                   | 91                   |

---

​
## Conclusion
- LSTM and CNN models demonstrated impressive emotion recognition accuracy of 98% and 99%, showcasing their ability to capture time-based patterns and spatial features in audio data.​
- The results underscore the importance of leveraging deep learning structures for intricate sentiment analysis, emphasizing the significance of these models in understanding emotion-driven audio signals.​
- Random Forest and SVM models exhibited varying performances, serving as a reminder to carefully consider the specific qualities of each model and their compatibility with the complexities of emotion-driven audio signals.​
- Graphical representations, especially for the LSTM model, revealed consistent improvement in accuracy and reduction in training loss, providing valuable insights into the learning dynamics of the model.​

​
​
