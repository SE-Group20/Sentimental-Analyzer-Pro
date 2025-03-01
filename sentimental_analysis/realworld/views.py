import base64
import csv
import io
import json
import os
import shutil
import subprocess
import sys
from io import StringIO

import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import cv2
import langdetect
import nltk
import speech_recognition as sr
from deepface import DeepFace
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render
from django.template.defaulttags import register
from django.views.decorators.csrf import csrf_exempt
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pydub import AudioSegment
from realworld.fb_scrap import *
from realworld.newsScraper import *
from realworld.reddit_scrap import *
from realworld.twitter_scrap import *
from realworld.utilityFunctions import *
from spanish_nlp import classifiers
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
import re

from django.core.cache import cache
from googletrans import Translator
from transformers import pipeline

# force langdetect to be deterministic:
from langdetect import DetectorFactory
DetectorFactory.seed = 0


def pdfparser(data):
    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data = retstr.getvalue()

    text_file = open("Output.txt", "w", encoding="utf-8")
    text_file.write(data)

    text_file = open("Output.txt", 'r', encoding="utf-8")
    a = ""
    for x in text_file:
        if len(x) > 2:
            b = x.split()
            for i in b:
                a += " " + i
    final_comment = a.split('.')
    return final_comment

@login_required
def analysis(request):
    return render(request, 'realworld/index.html')

def get_clean_text(text):
    text = removeLinks(text)
    text = stripEmojis(text)
    text = removeSpecialChar(text)
    text = stripPunctuations(text)
    text = stripExtraWhiteSpaces(text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english')).union(['the', 'a', 'an', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'ought', 'it', 'they', 'them', 'their', 'theirs', 'themselves', 'he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'i', 'me', 'my', 'mine', 'myself'])
    stop_words.add('rt')
    stop_words.add('')
    newtokens = [item for item, pos_tag in pos_tag(tokens) if item.lower() not in stop_words and pos_tag in ['NN', 'VB', 'JJ', 'RB']]

    textclean = ' '.join(newtokens)
    return textclean

def detailed_analysis(result):
    result_dict = {"pos": 0, "neu": 0, "neg": 0}
    neg_count = 0
    pos_count = 0
    neu_count = 0
    total_count = len(result)

    for item in result:
        cleantext = get_clean_text(str(item))
        # print(cleantext)
        sentiment = sentiment_scores(cleantext)
        pos_count += sentiment['pos']
        neu_count += sentiment['neu']
        neg_count += sentiment['neg']
    total = pos_count + neu_count + neg_count
    if(total>0):
        pos_ratio = (pos_count/total)
        neu_ratio = (neu_count/total)
        neg_ratio = (neg_count/total)
        result_dict['pos'] = pos_ratio
        result_dict['neu'] = neu_ratio
        result_dict['neg'] = neg_ratio
    return result_dict

def detailed_analysis_sentence(result):
    sia = SentimentIntensityAnalyzer()
    result_dict = {}
    result_dict['compound'] = sia.polarity_scores(result)['compound']
    return result_dict

def input(request):
    if request.method == 'POST':
        file = request.FILES['document']
        fs = FileSystemStorage()
        fs.save(file.name, file)
        pathname = 'sentimental_analysis/media/'
        extension_name = file.name
        extension_name = extension_name[len(extension_name) - 3:]
        path = pathname + file.name
        destination_folder = 'sentimental_analysis/media/document/'
        shutil.copy(path, destination_folder)
        useFile = destination_folder + file.name
        result = {}
        finalText = ''
        if extension_name == 'pdf':
            value = pdfparser(useFile)
            result = detailed_analysis(value)
            finalText = result
        elif extension_name == 'txt':
            text_file = open(useFile, 'r', encoding="utf-8")
            a = ""
            for x in text_file:
                if len(x) > 2:
                    b = x.split()
                    for i in b:
                        a += " " + i
            final_comment = a.split('.')
            text_file.close()
            finalText = final_comment
            result = detailed_analysis(final_comment)
        folder_path = 'sentimental_analysis/media/'
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        return render(request, 'realworld/results.html', {'sentiment': result, 'text': finalText, 'reviewsRatio': {}, 'totalReviews': 1, 'showReviewsRatio': False})
    else:
        note = "Please Enter the Document you want to analyze"
        return render(request, 'realworld/home.html', {'note': note})

def inputimage(request):
    if request.method == 'POST':
        file = request.FILES['document']
        fs = FileSystemStorage()
        fs.save(file.name, file)
        pathname = 'sentimental_analysis/media/'
        extension_name = file.name
        extension_name = extension_name[len(extension_name) - 3:]
        path = pathname + file.name
        destination_folder = 'sentimental_analysis/media/document/'
        shutil.copy(path, destination_folder)
        useFile = destination_folder + file.name
        image = cv2.imread(useFile)
        detected_emotion = DeepFace.analyze(image)

        emotions_dict = {'happy': 0.0, 'sad': 0.0, 'neutral': 0.0}
        for emotion in detected_emotion:
            emotion_scores = emotion['emotion']
            happy_score = emotion_scores['happy']
            sad_score = emotion_scores['sad']
            neutral_score = emotion_scores['neutral']

            emotions_dict['happy'] += happy_score
            emotions_dict['sad'] += sad_score
            emotions_dict['neutral'] += neutral_score

        total_score = sum(emotions_dict.values())
        if total_score > 0:
            for emotion in emotions_dict:
                emotions_dict[emotion] /= total_score

        print(emotions_dict)
        finalText = max(emotions_dict, key=emotions_dict.get)
        return render(request, 'realworld/resultsimage.html',
                      {'sentiment': emotions_dict, 'text': finalText, 'analyzed_image_path': useFile})


def productanalysis(request):
    if request.method == 'POST':
        blogname = request.POST.get("blogname", "")

        text_file = open(
            "Amazon_Comments_Scrapper/amazon_reviews_scraping/amazon_reviews_scraping/spiders/ProductAnalysis.txt", "w")
        text_file.write(blogname)
        text_file.close()

        spider_path = r'Amazon_Comments_Scrapper/amazon_reviews_scraping/amazon_reviews_scraping/spiders/amazon_review.py'
        output_file = r'Amazon_Comments_Scrapper/amazon_reviews_scraping/amazon_reviews_scraping/spiders/reviews.json'
        command = f"scrapy runspider \"{spider_path}\" -o \"{output_file}\" "
        result = subprocess.run(command, shell=True)

        if result.returncode == 0:
            print("Scrapy spider executed successfully.")
        else:
            print("Error executing Scrapy spider.")

        with open(r'Amazon_Comments_Scrapper/amazon_reviews_scraping/amazon_reviews_scraping/spiders/reviews.json',
                  'r') as json_file:
            json_data = json.load(json_file)
        reviews = []
        reviews2 = {
            "pos": 0,
            "neu": 0,
            "neg": 0,
        }
        for item in json_data:
            reviews.append(item['Review'])
            r = detailed_analysis_sentence(item['Review'])
            if(r != {}):
                st = item['Stars']
                if(st is not None):
                    stars = int(float(st))
                    if(stars != -1):
                        if(stars >= 4):
                            r['compound'] += 0.1
                        elif(stars >= 2):
                           continue
                        else:
                            r['compound'] -= 0.1
                if(r['compound'] > 0.4):
                    reviews2['pos'] += 1
                elif(r['compound'] < -0.4):
                    reviews2['neg'] += 1
                else:
                    reviews2['neu'] +=1
        finalText = reviews
        totalReviews = reviews2['pos'] + reviews2['neu'] + reviews2['neg']
        result = detailed_analysis(reviews)
        return render(request, 'realworld/results.html', {'sentiment': result, 'text' : finalText, 'reviewsRatio': reviews2, 'totalReviews': totalReviews, 'showReviewsRatio': True})

    else:
        note = "Please Enter the product blog link for analysis"
        return render(request, 'realworld/productanalysis.html', {'note': note})


# Text sentiment Analysis - Detect Language and use corresponding model for sentiment score

nltk.download('vader_lexicon')

def detect_language(texts: str | list[str]):
    """Detects the language of the given text using spaCy."""
    detected_languages = []
    if isinstance(texts, str):
        texts = [texts]
    for text in texts:
        try:
            lang = langdetect.detect(text)
            detected_languages.append(lang)
        except Exception as e:
            logging.error(f"Error detecting language: {e}")
            detected_languages.append("unknown")
    # Determine the most common detected language
    return max(set(detected_languages), key=detected_languages.count)

def analyze_sentiment(text, language):
    """Performs sentiment analysis based on the detected language."""
    if language == "en":
        scores = sentiment_analyzer_scores(text)
        return {
            "pos": scores["pos"],
            "neu": scores["neu"],
            "neg": scores["neg"]
        }
    elif language == "es":  # Spanish using Spanish NLP Classifier
        sc = classifiers.SpanishClassifier(model_name="sentiment_analysis")
        result_classifier = sc.predict(text)
        print(result_classifier)
        return {
            "pos": result_classifier.get("positive", 0.0),
            "neu": result_classifier.get("neutral", 0.0),
            "neg": result_classifier.get("negative", 0.0)
        }
    else:
        translator  = Translator() #Use the imported google translator
        english_text = translator.translate(text, src=language, dest='en')
        translated_text = english_text.text

        scores = sentiment_analyzer_scores(translated_text)
        return {
            "pos": scores["pos"],
            "neu": scores["neu"],
            "neg": scores["neg"]
        }

def create_word_correlation_heatmap(text):
    # 1. Text Preprocessing
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha() and word not in stop_words]

    # 2. Word Co-occurrence Matrix
    vocabulary = sorted(list(set(words)))
    co_occurrence_matrix = pd.DataFrame(0, index=vocabulary, columns=vocabulary)

    window_size = 4  # Adjust window size as needed

    for i in range(len(words)):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                co_occurrence_matrix.loc[words[i], words[j]] += 1



    # 3. Correlation Matrix
    correlation_matrix = co_occurrence_matrix.corr()


    if correlation_matrix.shape[0] == 0:
        logging.warning("No sentences found for correlation matrix.")
        return None

    if np.isnan(correlation_matrix).any().any():
        logging.warning("NaN values found in correlation matrix. Replacing with 0.")
        correlation_matrix.fillna(0, inplace=True)

    # 4. Heatmap Visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Word Correlation Heatmap')
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return image_base64


def textanalysis(request):
    """Performs sentiment analysis for the single line text"""
    if request.method == 'POST':
        text_data = request.POST.get("textField", "")
        final_comment = text_data.split('.')
        result = {}
        finalText = final_comment
        image_base64 = create_word_correlation_heatmap(text_data)

        if detect_language(final_comment):
            result = detailed_analysis(final_comment)
        else:
            sc = classifiers.SpanishClassifier(model_name="sentiment_analysis")
            result_string = ' '.join(final_comment)
            result_classifier = sc.predict(result_string)
            result = {
                'pos': result_classifier.get('positive', 0.0),
                'neu': result_classifier.get('neutral', 0.0),
                'neg': result_classifier.get('negative', 0.0)
            }
        print("Sentiment Scores:", sentiment_scores)  # Debugging

        return render(request, 'realworld/results.html', {'sentiment': result, 'text' : finalText, 'reviewsRatio': {}, 'totalReviews': 1, 'showReviewsRatio': False,
                'heatmap_image': image_base64})


        # Making sure sentences with floating point numbers are getting split correctly
        # Sentences are split by sentence-ending delimiters . ? !
        sentences =  re.split(r'(?<!\d)[.!?]+(?!\d)', text_data)
        sentences = [s.strip() for s in sentences if s.strip()]

        detected_language = detect_language(sentences)
        print(detected_language)
        sentiment_result = analyze_sentiment(text_data, detected_language)

        return render(request, 'realworld/results.html', {'sentiment': sentiment_result,
                                                          'text' : sentences,
                                                          "language": detected_language.upper(),
                                                          'reviewsRatio': {},
                                                          'totalReviews': 1,
                                                          'showReviewsRatio': False
                                                          })
    else:
        note = "Enter the Text to be analysed!"
        return render(request, 'realworld/textanalysis.html', {'note': note})

def create_sentence_correlation_heatmap(texts):
    all_sentences = []
    for text in texts:
        sentences = sent_tokenize(text)
        all_sentences.extend(sentences)

    vocabulary = sorted(list(set(all_sentences)))
    co_occurrence_matrix = pd.DataFrame(0, index=vocabulary, columns=vocabulary)

    window_size = len(texts) #co-occur if in same batch.

    for i in range(len(all_sentences)):
        for j in range(len(all_sentences)):
            if i != j:
                if all_sentences[i] in sent_tokenize(texts[i//len(sent_tokenize(texts[0]))]) and all_sentences[j] in sent_tokenize(texts[j//len(sent_tokenize(texts[0]))]):
                    co_occurrence_matrix.loc[all_sentences[i], all_sentences[j]] += 1

    correlation_matrix = co_occurrence_matrix.corr()
    if correlation_matrix.shape[0] == 0:
        logging.warning("No sentences found for correlation matrix.")
        return None

    if np.isnan(correlation_matrix).any().any():
        logging.warning("NaN values found in correlation matrix. Replacing with 0.")
        correlation_matrix.fillna(0, inplace=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Sentence Correlation Heatmap')
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    print("Sentiment Sentence Scores:", correlation_matrix)  # Debugging
    return image_base64

def batch_analysis(request):
    """Performs sentiment analysis for multiple text lines"""

    if request.method == 'POST':
        texts_orig = request.POST.get("batchTextField", "")
        texts = texts_orig.split('\n')
        texts = [t.strip() for t in texts if t.strip()]

        # Initialize aggregate sentiment scores
        total_sentiment = {
            'pos': 0.0,
            'neg': 0.0,
            'neu': 0.0
        }

        # Process each text
        individual_results = {}  # Changed from list to dictionary
        for idx, text in enumerate(texts):
            final_comment = text.split('.')
            if detect_language(final_comment):
                result = detailed_analysis(final_comment)
            else:
                sc = classifiers.SpanishClassifier(model_name="sentiment_analysis")
                result_string = ' '.join(final_comment)
                result_classifier = sc.predict(result_string)
                result = {
                    'pos': result_classifier.get('positive', 0.0),
                    'neg': result_classifier.get('negative', 0.0),
                    'neu': result_classifier.get('neutral', 0.0)
                }

            # Add to totals
            total_sentiment['pos'] += result['pos']
            total_sentiment['neg'] += result['neg']
            total_sentiment['neu'] += result['neu']

            # Store individual results with index as key
            individual_results[str(idx)] = {
                'text': text,
                'sentiment': result
            }

        # Calculate average sentiment
        num_texts = len(texts) or 1
        avg_sentiment = {
            'pos': total_sentiment['pos'] / num_texts,
            'neg': total_sentiment['neg'] / num_texts,
            'neu': total_sentiment['neu'] / num_texts
        }

        heatmap_image = create_sentence_correlation_heatmap(texts)

        # optional field: get as csv
        if request.POST.get('download_csv'):
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="sentiment_analysis_results.csv"'

            writer = csv.writer(response)
            writer.writerow(['Index', 'Text', 'Positive', 'Negative', 'Neutral'])
            for idx, item in individual_results.items():
                writer.writerow([idx, item['text'], item['sentiment']['pos'], item['sentiment']['neg'], item['sentiment']['neu']])

            return response

        return render(request, 'realworld/results.html', {
            'sentiment': avg_sentiment,
            'text': texts,
            'reviewsRatio': individual_results,  # Now a dictionary
            'totalReviews': len(texts),
            'showReviewsRatio': True,
            'heatmap_image': heatmap_image,
            'texts_orig': request.POST.get("batchTextField", ""),
            'is_batch': True,
            'texts_orig': request.POST.get("batchTextField", ""),
            'is_batch': True,
            'heatmap_image': heatmap_image,
            'texts_orig': request.POST.get("batchTextField", ""),
            'is_batch': True
        })
    return render(request, 'realworld/batch_analysis.html')

# End of text sentiment analysis

def fbanalysis(request):
    if request.method == 'POST':
        current_directory = os.path.dirname(__file__)
        result = fb_sentiment_score()

        csv_file_fb = 'fb_sentiment.csv'
        csv_file_path = os.path.join(current_directory, csv_file_fb)

        # Open the CSV file and read its content
        with open(csv_file_path, 'r') as csv_file:
            # Use DictReader to read CSV data into a list of dictionaries
            csv_reader = csv.DictReader(csv_file)
            data = [row for row in csv_reader]

        text_dict = {"reviews": data}
        print("text_dict:", text_dict["reviews"])
        # Convert the list of dictionaries to a JSON array
        json_data = json.dumps(text_dict, indent=2)

        reviews = []

        for item in text_dict["reviews"]:
            #print("item :",item)
            reviews.append(item["FBPost"])
        finalText = reviews


        return render(request, 'realworld/results.html', {'sentiment': result, 'text' : finalText, 'reviewsRatio': {}, 'totalReviews': 1, 'showReviewsRatio': False})
    else:
        note = "Please Enter the product blog link for analysis"
        return render(request, 'realworld/productanalysis.html', {'note': note})

def twitteranalysis(request):
    if request.method == 'POST':
        current_directory = os.path.dirname(__file__)
        result = twitter_sentiment_score()

        csv_file_fb = 'twitt.csv'
        csv_file_path = os.path.join(current_directory, csv_file_fb)

        # Open the CSV file and read its content
        with open(csv_file_path, 'r') as csv_file:
            # Use DictReader to read CSV data into a list of dictionaries
            csv_reader = csv.DictReader(csv_file)
            data = [row for row in csv_reader]

        text_dict = {"reviews": data}
        print("text_dict:", text_dict["reviews"])
        # Convert the list of dictionaries to a JSON array
        json_data = json.dumps(text_dict, indent=2)

        reviews = []

        for item in text_dict["reviews"]:
            #print("item :",item)
            reviews.append(item["review"])
        finalText = reviews


        return render(request, 'realworld/results.html', {'sentiment': result, 'text' : finalText, 'reviewsRatio': {}, 'totalReviews': 1, 'showReviewsRatio': False})
    else:
        note = "Please Enter the product blog link for analysis"
        return render(request, 'realworld/productanalysis.html', {'note': note})

def redditanalysis(request):
    if request.method == 'POST':
        blogname = request.POST.get("blogname", "")  # Get the Reddit post URL from the form
        fetched_data = fetch_reddit_post(blogname)  # Fetch the Reddit post details

        # Combine the fetched data (title, body, comments) into a single list for analysis
        data = [fetched_data["title"], fetched_data["body"]] + fetched_data["comments"]
        # Perform sentiment analysis
        result = reddit_sentiment_score(data)

        # Combine the title, body, and comments into a single list for displaying on the results page
        reviews = [f"Title: {fetched_data['title']}", f"Body: {fetched_data['body']}"] + fetched_data["comments"]

        return render(request, 'realworld/results.html', {
            'sentiment': result,  # Sentiment analysis result
            'text': reviews,      # Display the text analyzed (title, body, comments)
            'reviewsRatio': {},   # Placeholder (optional, for further analysis)
            'totalReviews': len(reviews),  # Total number of items analyzed
            'showReviewsRatio': False
        })
    else:
        note = "Enter the Reddit post URL for analysis"
        return render(request, 'realworld/redditanalysis.html', {'note': note})

def audioanalysis(request):
    if request.method == 'POST':
        file = request.FILES['audioFile']
        fs = FileSystemStorage()
        fs.save(file.name, file)
        pathname = "sentimental_analysis/media/"
        extension_name = file.name
        extension_name = extension_name[len(extension_name) - 3:]
        path = pathname + file.name
        result = {}
        destination_folder = 'sentimental_analysis/media/audio/'
        shutil.copy(path, destination_folder)
        useFile = destination_folder + file.name
        text = speech_to_text(useFile)
        finalText = text
        result = detailed_analysis(text)

        folder_path = 'sentimental_analysis/media/'
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return render(request, 'realworld/results.html', {'sentiment': result, 'text' : finalText, 'reviewsRatio': {}, 'totalReviews': 1, 'showReviewsRatio': False})
    else:
        note = "Please Enter the audio file you want to analyze"
        return render(request, 'realworld/audio.html', {'note': note})

def livespeechanalysis(request):
    if request.method == 'POST':
        my_file_handle = open(
            'sentimental_analysis/realworld/recordedAudio.txt')
        audioFile = my_file_handle.read()
        result = {}
        text = speech_to_text(audioFile)

        finalText = text
        result = detailed_analysis(text)
        folder_path = 'sentimental_analysis/media/recordedAudio/'
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return render(request, 'realworld/results.html', {'sentiment': result, 'text' : finalText, 'reviewsRatio': {}, 'totalReviews': 1, 'showReviewsRatio': False})


@csrf_exempt
def recordaudio(request):
    if request.method == 'POST':
        audio_file = request.FILES['liveaudioFile']
        fs = FileSystemStorage()
        fs.save(audio_file.name, audio_file)
        folder_path = 'sentimental_analysis/media/'
        files = os.listdir(folder_path)

        pathname = "sentimental_analysis/media/"
        extension_name = audio_file.name
        extension_name = extension_name[len(extension_name) - 3:]
        path = pathname + audio_file.name
        audioName = audio_file.name
        destination_folder = 'sentimental_analysis/media/recordedAudio/'
        shutil.copy(path, destination_folder)
        useFile = destination_folder + audioName
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        audio = AudioSegment.from_file(useFile)
        audio = audio.set_sample_width(2)
        audio = audio.set_frame_rate(44100)
        audio = audio.set_channels(1)
        audio.export(useFile, format='wav')

        text_file = open("sentimental_analysis/realworld/recordedAudio.txt", "w")
        text_file.write(useFile)
        text_file.close()
        response = HttpResponse('Success! This is a 200 response.', content_type='text/plain', status=200)
        return response

def newsanalysis(request):
    if request.method == 'POST':
        topicname = request.POST.get("topicname", "")
        scrapNews(topicname, 10)

        json_data = cache.get(NEWS_JSON_CACHE_KEY)

        news = []
        for item in json_data:
            news.append(item['Summary'])

        # cached_sentiment, cached_text = analysis_cache.get_analysis(topicname, news)
        cached_sentiment, cached_text = None, None
        maybe_cached_sentiment: str|None = cache.get(f"{topicname}_sentiment")
        if maybe_cached_sentiment:
            cached_sentiment, cached_text = maybe_cached_sentiment
            cached_sentiment = json.loads(cached_sentiment)
            cached_text = json.loads(cached_text)


        if cached_sentiment and cached_text:
            print('loaded sentiment')
            return render(request, 'realworld/results.html', {
                'sentiment': cached_sentiment,
                'text': cached_text,
                'reviewsRatio': {},
                'totalReviews': 1,
                'showReviewsRatio': False
            })

        finalText = news
        result = detailed_analysis(news)
        print('cached sentiment')
        # analysis_cache.set_analysis(topicname, news, result, finalText)
        cache.set(f"{topicname}_sentiment", (json.dumps(result), json.dumps(finalText)))


        return render(request, 'realworld/results.html', {'sentiment': result, 'text' : finalText, 'reviewsRatio': {}, 'totalReviews': 1, 'showReviewsRatio': False})

    else:
        return render(request, 'realworld/index.html')

def speech_to_text(filename):
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return text


def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    print(score)
    return score


@register.filter(name='get_item')
def get_item(dictionary, key):
    return dictionary.get(key, 0)
