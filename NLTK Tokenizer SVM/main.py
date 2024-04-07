from data_loading import load_data
from preprocessing import tokenize_data
from data_splitting import split_data
from training import train_model
from evaluation import evaluate_model
from results import display_results

# Load data
file_path = r"D:\ML\CVD_Models\NLTK Tokenizer SVM\Unbiased_cwe476_Data.csv"
cwe476_Data = load_data(file_path)

# Preprocessing
x_input = tokenize_data(cwe476_Data)
y_input = cwe476_Data['Label']

X_train, X_test, y_train, y_test = split_data(x_input, y_input)

# Training
svm_model= train_model(X_train, y_train)

# Evaluation
y_pred = evaluate_model(svm_model, X_test, y_test)

# Display and save results
display_results(y_test, y_pred)

