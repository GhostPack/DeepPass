from flask import Flask, render_template, url_for, request, redirect, jsonify
from multiprocessing import Process
from flask_bootstrap import Bootstrap
import os, pickle, json, re, requests, zipfile, shutil, json, uuid
import numpy as np

app = Flask(__name__, template_folder='Templates')
Bootstrap(app)

# the deep learning password model served by tensorflow/serving
MODEL_URI = 'http://tensorflow:8501/v1/models/password:predict'

# the Tika docker serve point
TIKA_URI = 'http://tika:9998/tika'

# extracted from the fit Keras tokenizer so we don't need Keras/Tensorflow as a requirement
CHAR_DICT = {'<UNK>': 1, 'e': 2, 'i': 3, 'a': 4, 'n': 5, 't': 6, 'r': 7, 'o': 8, 's': 9, 'c': 10, 'l': 11, 'A': 12, 'E': 13, 'd': 14, 'u': 15, 'm': 16, 'p': 17, 'I': 18, 'S': 19, 'R': 20, 'O': 21, 'N': 22, 'g': 23, 'T': 24, '-': 25, 'L': 26, 'h': 27, 'y': 28, 'C': 29, 'b': 30, 'f': 31, 'M': 32, 'v': 33, 'D': 34, '1': 35, 'U': 36, 'H': 37, 'P': 38, 'k': 39, '2': 40, '0': 41, 'B': 42, 'G': 43, 'w': 44, 'Y': 45, 'K': 46, '3': 47, '9': 48, 'F': 49, '.': 50, ',': 51, '4': 52, '8': 53, 'V': 54, '5': 55, '7': 56, '6': 57, 'W': 58, 'j': 59, 'x': 60, 'z': 61, 'J': 62, 'q': 63, 'Z': 64, '_': 65, "'": 66, ':': 67, 'X': 68, 'Q': 69, '/': 70, ')': 71, '(': 72, '"': 73, '!': 74, ';': 75, '*': 76, '@': 77, '\\': 78, ']': 79, '?': 80, '[': 81, '<': 82, '>': 83, '=': 84, '#': 85, '&': 86, '$': 87, '+': 88, '%': 89, '`': 90, '~': 91, '^': 92, '{': 93, '}': 94, '|': 95}

# regex for 7-32 character mixed alphanumeric + special char
PASSWORD_REGEX = re.compile('^(?=.*[0-9])(?=.*[a-zA-Z])(?=.*[~!@#$%^&*_\-+=`|\()\{\}[\]:;"\'<>,.?\/])(?=.*\d).{7,32}$')

# the number of words around matches to display
CONTEXT_WORDS = 5


"""
Routes
"""

# web interface
@app.route('/', methods=['GET','POST'])
def index():
    
    if not os.path.exists("static"):
        os.makedirs("static")

    if request.method == 'POST':
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':

            custom_regex = None
            if(request.form['regex_terms']):
                regex_terms = request.form['regex_terms']
                r = f".*({regex_terms.replace(',', '|')}).*"
                custom_regex = re.compile(r, flags=re.IGNORECASE)

            file_path = os.path.join('static', uploaded_file.filename)
            print(f"file_path: {file_path}")

            uploaded_file.save(file_path)

            if(uploaded_file.filename.endswith(".zip")):
                results = process_zip(file_path, custom_regex)
            else:
                results = [ process_document(file_path, custom_regex) ]

            background_remove(file_path)

            return render_template('result.html', results = results)

    return render_template('index.html')


# API endpoint
@app.route('/api/passwords', methods=['PUT', 'POST', 'PUT'])
def passwords():

    if not os.path.exists("static"):
        os.makedirs("static")

    uploaded_file = request.files['file']
    if uploaded_file.filename != '':

        file_path = os.path.join('static', uploaded_file.filename)
        uploaded_file.save(file_path)

        if(uploaded_file.filename.endswith(".zip")):
            results = process_zip(file_path)
        else:
            results = [ process_document(file_path) ]

        background_remove(file_path)

    else:
        results = "error"

    return json.dumps(results, cls=NumpyEncoder)


"""
Helpers
"""

def process_zip(file_path, custom_regex=None):
    """
    Extracts a zip file of multiple documents and processes each.

    :param file_path: The path of the document to process.
    :return: A list containing the result dictionaries from process_document().
    """

    results = []

    # generate a random folder for extraction
    zip_folder = f"static/{uuid.uuid4().hex}/"

    # extract all the files from the zip
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(zip_folder)

    # process each file extracted
    for root, dirs, files in os.walk("static"):
        for file in files:
            full_path = os.path.join(root, file)
            if not file.endswith(".zip") and not file.startswith("~") and not file.startswith(".") and os.path.getsize(full_path) != 0:
                try:
                    results.append(process_document(full_path, custom_regex))
                except:
                    print(f"[!] Error processing '{full_path}'")

    # cleanup the extracted folder
    shutil.rmtree(zip_folder)

    return results


def process_document(file_path, custom_regex=None):
    """
    Processes a file through the NN/regex.

    :param file_path: The path of the document to process.
    :return: A dictionary containing the file name, model password candidates, and regex password candidates.
    """

    results = []

    print(f"[*] Processing: '{file_path[7:]}'")

    # extract the text using Tika
    text = extract_text(file_path)

    n = 50000

    # chunkify things if needed because of the memory limits for the deep learning model (TODO: double check this)
    chunks = [text[i:i+n] for i in range(0, len(text), n)]
    model_password_candidates = list()

    for chunk in chunks:
        # extract password candidates from the model
        model_password_candidates.extend(extract_passwords_model(chunk))

    print(f"[*] Number of model password candidates: {len(model_password_candidates)}")

    # extract password candidates from the regex
    regex_password_candidates = extract_terms_regex(text, PASSWORD_REGEX)

    print(f"[*] Number of regex password candidates: {len(regex_password_candidates)}")

    # check for any custom regex
    custom_regex_matches = None
    if(custom_regex):
        custom_regex_matches = extract_terms_regex(text, custom_regex)

    result = {
        'file_name': os.path.basename(file_path),
        'model_password_candidates': model_password_candidates,
        'regex_password_candidates': regex_password_candidates,
        'custom_regex_matches': custom_regex_matches
    }

    return result


def tokenize(word):
    """
    Tokenizes and pads the supplied word to the proper length.

    :param word: The word to tokenize.
    :return: A 32 val numpy array of representing the character tokenization of the word.
    """
    
    global CHAR_DICT

    if len(word) < 7 or len(word) > 32:
        return [0]*32
    else:
        seq = [CHAR_DICT[char] if char in CHAR_DICT else 1 for char in word]
        seq.extend([0] * (32-len(seq)))
        return seq


def extract_passwords_model(document):
    """
    Tokenizes the input text, submits it to the served model, and returns words that might be passwords.

    :param document: The string representing the text of a single document.
    :return: A list of any possible passwords.
    """

    global MODEL_URI
    global CONTEXT_WORDS

    # extract whitespace stripped words
    words = np.array([word.strip() for word in document.split()])

    # turn the input words into sequences and pad them to 32
    tokenized_words = [tokenize(word) for word in words]
    #print(f"[*] tokenized_words: {tokenized_words}")

    # post the tokenized words to the served model
    data = json.dumps({
        'inputs': tokenized_words
    })
    response = requests.post(MODEL_URI, data=data.encode('utf-8'), timeout=60)
    result = json.loads(response.text)
    pred = np.array(result['outputs'])
    #print(f"[*] pred: {pred}")

    # properly cast the predictions for each word
    pred = (pred > 0.5).astype("int32")

    # use the predictions as indicies for the words to return
    positive_indicies = np.where(pred == 1)
    passwords = list()

    for x in positive_indicies[0]:
        left_context = words[:x][-CONTEXT_WORDS:]
        password = words[x]
        right_context = words[x+1:CONTEXT_WORDS+x+1]

        result = {
            "left_context": left_context,
            "password": password,
            "right_context": right_context
        }

        passwords.append(result)

    return passwords


def extract_terms_regex(document, regex):
    """
    extract_terms_regex Runs the supplied compiled regex against extracted documents.

    :param document: The string representing the text of a single document.
    :return: A list of terms matching the regex.
    """

    global CONTEXT_WORDS

    words = np.array([word.strip() for word in document.split()])
    passwords = list()

    for x, word in enumerate(words):
        if regex.match(word):
            left_context = words[:x][-CONTEXT_WORDS:]
            password = words[x]
            right_context = words[x+1:CONTEXT_WORDS+x+1]

            result = {
                "left_context": left_context,
                "password": password,
                "right_context": right_context
            }

            passwords.append(result)

    return passwords


def extract_text(file_path):
    """
    extract_text Extracts plaintext from a document path using Tika.

    :param file_path: The path of the file to extract text from.
    :return: ASCII-encoded plaintext extracted from the document.
    """

    with open(file_path, 'rb') as f:
        resp = requests.put(TIKA_URI, f, headers={'Accept': 'text/plain'})
        if(resp.status_code == 200):
            return resp.text.strip().encode("ascii","ignore").decode()


def background_remove(path):
    """
    Backgroundable helper to remove the uploaded file.
    """
    task = Process(target=rm(path))
    task.start()


def rm(path):
    os.remove(path)


class NumpyEncoder(json.JSONEncoder):
    """
    Helper to JSONify numpy arrays.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    app.run(debug = True)
